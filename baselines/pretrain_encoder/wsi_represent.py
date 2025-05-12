import sys, torch
sys.path.append("your/path")
import torch.nn as nn

class WSILinearClassifier(nn.Module):
    def __init__(self, backbone='dino', num_classes=2): 
        super(WSILinearClassifier, self).__init__()
        assert backbone in ['resnet', 'vit', 'dino'], "Only ResNet, ViT and DiNOv2 supported for now."

        if backbone == 'resnet':
            self.feature_dim = 2048
        elif backbone == 'vit':
            self.feature_dim = 768
        elif backbone == 'dino':
            self.feature_dim = 768
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
    


if __name__ == "__main__":
    model = WSILinearClassifier(backbone='dino', num_classes=2)
    dummy_input = torch.randn(4, 768)
    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)   # torch.Size([4, 768])
    print("Output shape:", output.shape)       # torch.Size([4, 2])
    print("Output logits:\n", output)