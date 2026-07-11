import os
import json
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import f1_score

from config import Config
from dataset import GoEmotionsDataset
from losses import AsymmetricLoss, MultiLabelSupervisedContrastiveLoss
from model import ImprovedRoberta
from metrics import Metrics

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.SAVE_DIR, exist_ok=True)
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
        self.device = cfg.DEVICE
        self.tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL_NAME)
        self.model = None
        self.criterion = AsymmetricLoss(gamma_pos=cfg.ASL_GAMMA_POS, gamma_neg=cfg.ASL_GAMMA_NEG, clip=cfg.ASL_CLIP)
        self.contrastive = MultiLabelSupervisedContrastiveLoss(temperature=cfg.CONTRASTIVE_TAU) if cfg.CONTRASTIVE_WEIGHT > 0 else None
        self.thresholds = None

    def load_data(self):
        ds = load_dataset("google-research-datasets/go_emotions", "simplified")
        self.train_texts = ds['train']['text']; self.train_labels = ds['train']['labels']
        self.val_texts = ds['validation']['text']; self.val_labels = ds['validation']['labels']
        self.test_texts = ds['test']['text']; self.test_labels = ds['test']['labels']
        
        self.train_dataset = GoEmotionsDataset(self.train_texts, self.train_labels, self.tokenizer, self.cfg.MAX_LENGTH)
        self.val_dataset = GoEmotionsDataset(self.val_texts, self.val_labels, self.tokenizer, self.cfg.MAX_LENGTH)
        self.test_dataset = GoEmotionsDataset(self.test_texts, self.test_labels, self.tokenizer, self.cfg.MAX_LENGTH)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=True, drop_last=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=False)

    def initialize_model(self):
        self.model = ImprovedRoberta(self.cfg.MODEL_NAME, self.cfg.NUM_LABELS).to(self.device)
        
        if self.cfg.FREEZE_EPOCHS > 0:
            for p in self.model.roberta.parameters():
                p.requires_grad = False
                
        base_lr = self.cfg.LEARNING_RATE
        decay = float(self.cfg.LAYER_WISE_LR_DECAY)
        roberta = self.model.roberta
        num_layers = len(roberta.encoder.layer)

        param_groups = []
        emb_params = [p for p in roberta.embeddings.parameters()]
        param_groups.append({'params': emb_params, 'lr': base_lr * (decay ** (num_layers + 1)), 'weight_decay': 0.01})

        for i, layer in enumerate(roberta.encoder.layer):
            layer_idx_from_top = num_layers - 1 - i
            lr_scale = decay ** layer_idx_from_top
            params = [p for p in layer.parameters()]
            param_groups.append({'params': params, 'lr': base_lr * lr_scale, 'weight_decay': 0.01})

        other_roberta = [p for n, p in roberta.named_parameters() if 'encoder.layer' not in n and 'embeddings' not in n]
        if other_roberta:
            param_groups.append({'params': other_roberta, 'lr': base_lr * (decay ** num_layers), 'weight_decay': 0.01})

        head_params = [p for n, p in self.model.named_parameters() if 'roberta' not in n]
        param_groups.append({'params': head_params, 'lr': base_lr * 3.0, 'weight_decay': 0.01})

        self.optimizer = AdamW(param_groups)
        total_steps = max(1, len(self.train_loader) * self.cfg.NUM_EPOCHS)
        warmup_steps = max(1, int(self.cfg.WARMUP_RATIO * total_steps))
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0.0
        
        if epoch_idx == self.cfg.FREEZE_EPOCHS:
            for p in self.model.roberta.parameters():
                p.requires_grad = True
            print("Unfroze RoBERTa backbone.")

        for batch in tqdm(self.train_loader, desc=f"Train E{epoch_idx+1}"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_projection=True)
            logits = out['logits']
            pooled = out['pooled']
            loss = self.criterion(logits, labels)

            if self.contrastive is not None:
                proj = out['proj']
                c_loss = self.contrastive(proj, labels)
                loss = loss + self.cfg.CONTRASTIVE_WEIGHT * c_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        self.model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_projection=False)
                probs = torch.sigmoid(out['logits'])
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        preds = (all_probs >= 0.5).astype(int)
        metrics = Metrics.calc(all_labels, preds, Config.EMOTION_LABELS)
        return metrics, all_probs, all_labels

    def optimize_thresholds(self, val_probs, val_labels):
        n_labels = val_labels.shape[1]
        thresholds = np.zeros(n_labels)
        for i in range(n_labels):
            best_t = 0.5; best_f1 = 0.0
            for t in np.arange(0.1, 0.9, 0.02):
                p = (val_probs[:, i] >= t).astype(int)
                if p.sum() == 0:
                    continue
                f1 = f1_score(val_labels[:, i], p, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1; best_t = t
            thresholds[i] = best_t
        self.thresholds = thresholds
        return thresholds

    def save_model(self):
        torch.save({'model_state_dict': self.model.state_dict()}, os.path.join(self.cfg.SAVE_DIR, 'model.pth'))
        self.tokenizer.save_pretrained(self.cfg.SAVE_DIR)

    def load_model(self):
        ck = os.path.join(self.cfg.SAVE_DIR, 'model.pth')
        if not os.path.exists(ck):
            raise FileNotFoundError("No checkpoint")
        state = torch.load(ck, map_location=self.device)
        if self.model is None:
            self.model = ImprovedRoberta(self.cfg.MODEL_NAME, self.cfg.NUM_LABELS).to(self.device)
        self.model.load_state_dict(state['model_state_dict'])

    def full_train(self):
        self.load_data()
        self.initialize_model()
        best_f1 = 0.0
        
        for epoch in range(self.cfg.NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{self.cfg.NUM_EPOCHS}")
            tr_loss = self.train_epoch(epoch)
            print("Train loss:", tr_loss)
            
            val_metrics, val_probs, val_labels = self.evaluate(self.val_loader)
            print("Val metrics:", val_metrics)
            
            if val_metrics['f1_macro'] > best_f1:
                best_f1 = val_metrics['f1_macro']
                self.save_model()
                print("Saved best model.")
                
        _, val_probs, val_labels = self.evaluate(self.val_loader)
        thr = self.optimize_thresholds(val_probs, val_labels)
        np.save(os.path.join(self.cfg.SAVE_DIR, 'optimal_thresholds.npy'), thr)
        
        self.load_model()
        test_metrics, test_probs, test_labels = self.evaluate(self.test_loader)
        preds = (test_probs >= self.thresholds).astype(int)
        final_metrics = Metrics.calc(test_labels, preds, Config.EMOTION_LABELS)
        
        with open(os.path.join(self.cfg.RESULTS_DIR, 'test_results.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2)
            
        print("Final test metrics:", final_metrics)
        return final_metrics