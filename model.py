import torch
import torch.nn as nn
import torchvision.models as models

class BrainTumorModel(nn.Module):
    def __init__(self, num_classes=2):
        super(BrainTumorModel, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
def get_model(num_classes=2, device='cuda'):
    model = BrainTumorModel(num_classes=num_classes)
    model = model.to(device)
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)
    
def load_model(path, num_classes=2, device='cuda'):
    model = get_model(num_classes=num_classes, device=device)
    model.load_state_dict(torch.load(path))
    return model