import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # alpha can be a scalar or a tensor with shape (num_classes,)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: logits tensor of shape (batch_size, num_classes)
        :param targets: ground truth labels tensor of shape (batch_size,)
        """
        # Compute softmax over the classes axis
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Gather the probabilities of the correct class
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))
        probs = torch.sum(probs * targets_one_hot, dim=1)
        log_probs = torch.sum(log_probs * targets_one_hot, dim=1)

        # Compute the focal loss components
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha = torch.tensor([self.alpha] * inputs.size(1)).to(inputs.device)
            else:
                alpha = self.alpha.to(inputs.device)
            alpha = torch.gather(alpha, 0, targets)
        else:
            alpha = 1.0
        
        loss = -alpha * (1 - probs) ** self.gamma * log_probs

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
