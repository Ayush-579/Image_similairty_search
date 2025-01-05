import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VGG16(nn.Module):
    def __init__(self, embedding_dim=128):
        super(VGG16, self).__init__()
        
        
        base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        base_model.classifier = nn.Identity()
        self.encoder = base_model.features  
        
        #
        self.embedding_layer = nn.Linear(512 * 7 * 7, embedding_dim)  

    def forward(self, x):
        
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features using the VGG16 encoder
        features = self.encoder(x)
        
       
        features = features.view(features.size(0), -1)
        
      
        embedding = self.embedding_layer(features)
        
        return embedding
