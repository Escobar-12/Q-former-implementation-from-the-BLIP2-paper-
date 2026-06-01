import torch
import torch.nn as nn
import torch.nn.functional as F
import accelerate

class Trainer():
    def __init__(self, model, train_dataloader, val_dataloader=None, epochs=1, lr=1e-3, early_stop_step_size=0, device="cuda"):
        super().__init__()
        self.model = model
        self.dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.lr = lr
        self.device = device

        # early stoping
        self.early_stop_step_size = early_stop_step_size
        self.es_steps = 0

        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr,
                                           weight_decay=0.001)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.85)

        self.train_losses = []
        self.val_losses = []

        self.last_val_loss = None
        self.best_val_loss = float("inf")
        self.avg_val_loss = float("inf")

    def train(self, stage:int=1, save_model=False, device="cuda"):
        self.model.train()


        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_idx, batch in enumerate(self.dataloader):
                image_inputs = batch["pixel_values"].to(self.device)  # (B, C, H, W)
                bert_inputs = batch["BertInputs"].to(self.device)  # (B, bert_len)
                bert_input_mask = batch["BertInputMask"].to(self.device) # (B, bert_len)
                llm_input_ids = batch["llm_input_ids"].to(self.device)  # (B, llm_seq_len)
                llm_attention_mask = batch["llm_attention_mask"].to(self.device)

                outputs = self.model(image_inputs=image_inputs, bert_inputs=bert_inputs, bert_input_mask=bert_input_mask, input_ids=llm_input_ids,
                                     attention_mask=llm_attention_mask, stage=stage)

                model_loss = outputs["model_loss"]


                normalized_model_loss = model_loss / len(self.dataloader)
                normalized_model_loss.backward()

                # Tracking
                epoch_loss += normalized_model_loss


            # Validation Loss
            if self.val_dataloader is not None:
                val_loss = 0
                for batch_idx, batch in enumerate(self.val_dataloader):

                    val_image_inputs = batch["pixel_values"].to(self.device)  # (B, C, H, W)
                    val_bert_inputs = batch["BertInputs"].to(self.device)  # (B, bert_len)
                    val_llm_input_ids = batch["llm_input_ids"].to(self.device)  # (B, llm_seq_len)
                    val_llm_attention_mask = batch["llm_attention_mask"].to(self.device)

                    with torch.no_grad():
                        outputs = self.model(image_inputs=val_image_inputs, bert_inputs=val_bert_inputs,
                                             input_ids=val_llm_input_ids, attention_mask=val_llm_attention_mask)
                    model_loss = outputs["model_loss"]
                    normalized_val_loss = model_loss / len(self.val_dataloader)
                    val_loss += normalized_val_loss

                self.avg_val_loss = val_loss
                self.val_losses.append(self.avg_val_loss)

                ## calculate early stop
                if self.early_stop_step_size != 0 and self.last_val_loss is not None:
                    if self.es_steps >= self.early_stop_step_size:
                        break
                    if self.last_val_loss < self.avg_val_loss:
                        self.es_steps += 1

                print(f'\n=== Epoch {epoch + 1}/{self.epochs} Complete ===')
                print(f'Average Validation Loss: {self.avg_val_loss:.4f}  -  ', end="")

            ## Update
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Step Lr Scheduler
            self.scheduler.step()

            # End of epoch
            self.train_losses.append(epoch_loss)

            print(f'Average Training Loss: {epoch_loss:.4f}  -  ', end="")
            print(f'lr: {self.scheduler.get_lr()}\n')

            if save_model and self.val_dataloader is not None and self.best_val_loss > self.avg_val_loss:
                torch.save(self.model.q_former.state_dict(), f"q_former_model_Stage{stage}.pth")
                self.best_val_loss = self.avg_val_loss

            ## set previous loss
            self.last_val_loss = self.avg_val_loss

    def continue_training(self, saved_model_path:str, stage:int=1, save_model=False):
        self.model.q_former.load_state_dict(torch.load(saved_model_path))
        self.train(stage, save_model)

    def history(self):
        return self.train_losses, self.val_losses
