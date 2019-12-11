
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator for adversarial training and multi-task learning
    """
    def __init__(self, feat_dim, num_class):
        self.linear = nn.Linear(feat_dim, num_class)
    
    def forward(self, inputs):
        """
        args:
            inputs: B x H
        output:
            predictions: B x C
        """
        predictions = self.linear(inputs)
        return predictions
