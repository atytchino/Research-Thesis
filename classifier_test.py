import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchmetrics import F1Score
import numpy as np

# Data preprocessing and augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#load the two datasets and image loaders
test_dataset = datasets.ImageFolder(root='archive(1)/Testing Data/Testing Data', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load ResNet50 model, loss function, optimizer
model = models.resnet50(weights=ResNet50_Weights.DEFAULT) 
model.load_state_dict(torch.load('trained_model.pth', map_location=device, weights_only=True))
model.to(device)

#Loss Function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Evaluate mode
model.eval()

# Initialize F1 Score metric
f1_metric = F1Score(num_classes=len(test_dataset.classes), average='macro', task='multiclass')

# Testing loop
testing_loss = 0.0  
correct = 0.0
total = 0.0
truth_labels = []
predicted_labels = []
criterion = torch.nn.CrossEntropyLoss()

# Disable gradient during test stage
with torch.no_grad():
    # testing loop
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  
        loss=criterion(outputs, labels)
        testing_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        truth_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

        #add to F1 score
        f1_metric(predicted, labels)

# Accuracy calculation
testing_loss /= len(test_loader)
testing_accuracy = correct / total

# Compute F1 score
f1_score = f1_metric.compute()

# Print validation results
print(f'Validation Loss: {testing_loss:.4f}, Validation Accuracy: {testing_accuracy:.2%}, F1 Score: {f1_score:.4f}')
