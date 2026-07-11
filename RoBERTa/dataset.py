import torch
from torch.utils.data import Dataset
from typing import List
from config import Config

class GoEmotionsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labs = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        label_vector = torch.zeros(Config.NUM_LABELS, dtype=torch.float)
        for l in labs:
            if 0 <= l < Config.NUM_LABELS:
                label_vector[l] = 1.0
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': label_vector
        }