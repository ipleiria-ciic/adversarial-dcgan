import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import numpy as np
import random
import matplotlib.pyplot as plt

class UniversalAdversarialPerturbation:
    def __init__(self, model, epsilon=0.03, num_iterations=1000, alpha=1.0):
        self.model = model
        self.epsilon = epsilon  # The magnitude of the perturbation
        self.num_iterations = num_iterations  # Number of iterations for optimization
        self.alpha = alpha  # Perturbation strength scaling factor
    
    def generate_uap(self, data_loader, device):
        # Initialize the perturbation with zeros (same shape as input images)
        uap = torch.zeros_like(next(iter(data_loader))[0]).to(device)

        uap.requires_grad = True
        
        optimizer = optim.Adam([uap], lr=self.alpha)
        
        for i in range(self.num_iterations):
            total_loss = 0
            for images, _ in data_loader:
                images = images.to(device)
                
                # Create perturbed images
                perturbed_images = images + uap
                perturbed_images = torch.clamp(perturbed_images, 0, 1)
                
                # Forward pass
                outputs = self.model(perturbed_images)
                loss = F.cross_entropy(outputs, torch.argmax(outputs, dim=1))
                
                # Backward pass and update the perturbation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.num_iterations}, Loss: {total_loss}")
        
        return uap.detach()

    def apply_uap(self, images, uap):
        perturbed_images = images + uap
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        return perturbed_images


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

model = torchvision.models.resnet18(pretrained=True)
model = model.to(device)
model.eval()  # Set the model to evaluation mode

