import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class PrognosticModel(nn.Module):
    def __init__(self, num_classes=3):  # 3 classes: good, moderate, poor prognosis
        super(PrognosticModel, self).__init__()
        
        # Use pre-trained ResNet50 as backbone
        self.backbone = resnet50(pretrained=True)
        
        # Modify first conv layer to accept single channel input
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom head for prognosis prediction
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        output = self.head(features)
        return F.softmax(output, dim=1)

class PrognosticTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
    def train_step(self, images, labels):
        self.model.train()
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, dataloader):
        self.model.eval()
        total = 0
        correct = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def predict(self, images):
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
        return predicted
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 