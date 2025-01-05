import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ResNet50, self).__init__()
        
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        
        base_model.fc = nn.Identity()  
        self.encoder = base_model
        
      
        self.embedding_layer = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        # Convert 1-channel (grayscale) to 3-channel (RGB)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Resize to 224x224 for ResNet input
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Pass through ResNet and embedding layer
        features = self.encoder(x)
        embedding = self.embedding_layer(features)
        return embedding
