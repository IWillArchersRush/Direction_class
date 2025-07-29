import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class DirectionClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        base = mobilenet_v2(weights='IMAGENET1K_V1')
        self.feature_extractor = base.features  # (B, 1280, 7, 7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, 1280, 1, 1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)  # over sequence
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x)        # (B*T, 1280, 7, 7)
        feats = self.pool(feats).squeeze(-1).squeeze(-1)  # (B*T, 1280)
        feats = feats.view(B, T, -1)             # (B, T, 1280)
        pooled = self.temporal_pool(feats.transpose(1, 2)).squeeze(-1)  # (B, 1280)
        out = self.classifier(pooled)            # (B, num_classes)
        return out
