import torch
import torch.nn as nn
from transformers import RobertaModel

class ImprovedRoberta(nn.Module):
    def __init__(self, model_name: str, num_labels: int, proj_dim: int = 256, hidden_intermediate: int = 512, dropout=0.2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size

        self.attn_proj = nn.Linear(hidden_size, 1)

        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_intermediate),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_intermediate, num_labels)
        )

        self.label_emb = nn.Parameter(torch.randn(num_labels, hidden_size) * 0.02)

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, input_ids, attention_mask, return_projection=False):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        h = out.last_hidden_state  

        attn_scores = self.attn_proj(h).squeeze(-1)            
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        attn_w = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(attn_w * h, dim=1)                 

        logits_head = self.classifier_head(pooled)            
        logits_label = torch.matmul(pooled, self.label_emb.t())  
        logits = logits_head + logits_label

        outputs = {'logits': logits, 'pooled': pooled}
        if return_projection:
            proj = self.proj(pooled)
            outputs['proj'] = proj
        return outputs