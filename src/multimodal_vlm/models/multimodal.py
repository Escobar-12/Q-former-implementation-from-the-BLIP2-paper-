import torch
import torch.nn as nn
import torch.nn.functional as F
import accelerate

from scripts.train import outputs


class MultiModal(nn.Module):
    def __init__(self, vit, vit_processor, llm, llm_tokenizer, qformer):
        super().__init__()
        self.vit = vit
        self.llm = llm
        self.q_former = qformer
        self.llm_tokenizer = llm_tokenizer
        self.vit_processor = vit_processor

        ## making sure qformer is trainable
        for param in self.q_former.parameters():
            param.requires_grad = True

        ## freezing llm
        for param in self.llm.parameters():
            param.requires_grad = False

        ## freezing vit
        for param in self.vit.parameters():
            param.requires_grad = False


        self.llm_embedding_layer = self.llm.get_input_embeddings()

    def forward(self, *, image_inputs, bert_inputs, bert_input_mask=None, input_ids=None, attention_mask=None, stage:int = 1):
        ## ViT output
        with torch.no_grad():
            vision_features = self.vit(pixel_values=image_inputs)
        image_embeds = vision_features.last_hidden_state

        ## ---------- Stage 1 ----------
        if stage == 1:
            ## Q-Former output
            model_output, model_loss = self.q_former(image_embeds=image_embeds, bert_inputs=bert_inputs,
                                                               bert_input_mask=bert_input_mask)
            # qformer_output = qformer_output.to(dtype=torch.bfloat16)

            # Extract query tokens from Qformer output
            # if isinstance(qformer_output, dict):
            #     query_tokens = qformer_output['query_output']  # (B, 32, 1024)
            # else:
            #     query_tokens = qformer_output  # Fallback


        ## ---------- Stage 2 ----------
        else:
            query_tokens, model_loss = self.q_former.s2forward(image_embeds=image_embeds)
            if input_ids is not None:
                # Get LLM embeddings for text
                text_embeds = self.llm_embedding_layer(input_ids).to(dtype=torch.bfloat16)  # (B, seq_len, hidden_size)

                # Concatenate query tokens with text embeddings
                llm_input_embeds = torch.cat([query_tokens, text_embeds], dim=1)

                # Adjust attention mask to include query tokens
                if attention_mask is not None:
                    query_mask = torch.ones(
                        query_tokens.size(1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    query_mask = query_mask.expand(B, -1)
                    llm_attention_mask = torch.cat([query_mask, attention_mask], dim=1)
                else:
                    llm_attention_mask = torch.ones(
                        llm_input_embeds.size(1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    llm_attention_mask = llm_attention_mask.expand(B, -1)
            else:
                llm_input_embeds = query_tokens
                llm_attention_mask = torch.ones(
                    query_tokens.size(1),
                    dtype=torch.long,
                    device="cuda"
                )
                llm_attention_mask = llm_attention_mask.expand(B, -1)

            with torch.no_grad():
                model_output = self.llm(
                    inputs_embeds=llm_input_embeds,
                    attention_mask=llm_attention_mask,
                    return_dict=True,
                )

            model_loss = model_output.loss

        return {
            "model_output": model_output,
            "model_loss": model_loss,
        }

    def generate(self, *, image, text):
        self.vit.eval()
        self.llm.eval()
        self.q_former.eval()

        messages = [
            {"role": "user", "content": f"{text}"}
        ]

        # ViT pass
        with torch.no_grad():
            image_input = self.vit_processor(images=image, return_tensors="pt").to(device="cuda")
            vision_features = self.vit(pixel_values=image_input["pixel_values"])
            image_embeds = vision_features.last_hidden_state

            # extract query tokens
            query_tokens = self.q_former.s2forward(image_embeds)

            # feed to LLM
            llm_input = self.llm_tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=True).to(device="cuda")
            llm_embed_layer = self.llm.get_input_embeddings()
            llm__text_embeds = llm_embed_layer(llm_input)
            llm_embeddings = torch.cat([query_tokens, llm__text_embeds], dim=1)

            attention_mask = torch.ones(
                llm_embeddings.shape[:2], dtype=torch.long, device="cuda"
            )

            llm_input_len = llm_embeddings.shape[1]
            generated_ids = self.llm.generate(inputs_embeds=llm_embeddings, attention_mask=attention_mask , max_new_tokens=30)
            model_output = self.llm_tokenizer.batch_decode(generated_ids[:, llm_input_len:], skip_spacial_token=True)[0]

        return model_output