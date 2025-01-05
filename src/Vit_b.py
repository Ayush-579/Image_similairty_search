import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViT(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ViT, self).__init__()
        
       
        base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        
        base_model.heads = nn.Identity()  
        self.encoder = base_model
        
       
        self.embedding_layer = nn.Linear(768, embedding_dim)

    def forward(self, x):
    
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            features = self.encoder(x) 
            embedding = self.embedding_layer(features)  
    
        return embedding
