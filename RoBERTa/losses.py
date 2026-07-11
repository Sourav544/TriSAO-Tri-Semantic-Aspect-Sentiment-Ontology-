import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8, reduction='mean'):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        prob = torch.sigmoid(logits)
        if self.clip is not None and self.clip > 0:
            pt_neg = torch.clamp(prob + self.clip, max=1.0)
        else:
            pt_neg = prob

        pos_term = targets * torch.log(prob.clamp(min=self.eps))
        neg_term = (1 - targets) * torch.log((1 - pt_neg).clamp(min=self.eps))

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pos_weight = (1 - prob) ** self.gamma_pos
            neg_weight = (prob) ** self.gamma_neg
            pos_term = pos_weight * pos_term
            neg_term = neg_weight * neg_term

        loss = - (pos_term + neg_term)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class MultiLabelSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        device = features.device
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]
        sim_mat = torch.matmul(features, features.T) / self.temperature

        label_eq = (labels @ labels.T) > 0
        mask = label_eq.fill_diagonal_(False)

        sim_exp = torch.exp(sim_mat - torch.max(sim_mat, dim=1, keepdim=True)[0].detach())
        denom = sim_exp.sum(dim=1) - torch.exp(torch.diag(sim_mat))
        
        loss = 0.0
        valid_count = 0
        for i in range(batch_size):
            pos_idx = mask[i]
            if pos_idx.sum() == 0:
                continue
            numer = sim_exp[i, pos_idx].sum()
            denom_i = denom[i].clamp(min=self.eps)
            loss_i = - torch.log( (numer / denom_i).clamp(min=self.eps) )
            loss += loss_i
            valid_count += 1
            
        if valid_count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss / valid_count