import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import itertools

# Set up logging to log all output to a file
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters and combinations for tuning
batch_sizes = [16]
learning_rates = [0.001]
num_epochs = 50
num_classes = 16  # AMD and DME
patience = 5  # for early stopping

# Dataset paths
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Function to convert grayscale image to 3 channels (RGB-like format)
def grayscale_to_rgb(x):
    return x.view(1, 224, 224).expand(3, -1, -1)  # Convert 1 channel to 3 channels by repeating the grayscale image

# Define transformations for the training and testing sets
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale input
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),           # Rotation
    transforms.RandomAffine(degrees=0, shear=20),  # Shearing
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Scaling
    transforms.ToTensor(),
    transforms.Lambda(grayscale_to_rgb),  # Convert grayscale to 3 channels
    transforms.Normalize(mean=[0.485] * 3, std=[0.229] * 3)  # Normalize for 3 channels
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale input
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Lambda(grayscale_to_rgb),  # Convert grayscale to 3 channels
    transforms.Normalize(mean=[0.485] * 3, std=[0.229] * 3)  # Normalize for 3 channels
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

# Split the training data into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image

# Modified train transformations for visualization with names
train_transform_with_names = [
    ('Grayscale', transforms.Grayscale(num_output_channels=1)),
    ('Resize (224x224)', transforms.Resize((224, 224))),
    ('Random Horizontal Flip', transforms.RandomHorizontalFlip(p=1.0)),
    ('Random Rotation (30°)', transforms.RandomRotation(30)),
    ('Random Affine (Shear 20°)', transforms.RandomAffine(degrees=0, shear=20)),
    ('Random Resized Crop', transforms.RandomResizedCrop(224, scale=(0.8, 1.0))),
    ('To Tensor', transforms.ToTensor()),
    ('Convert Grayscale to RGB', transforms.Lambda(grayscale_to_rgb)),
    ('Normalize', transforms.Normalize(mean=[0.485] * 3, std=[0.229] * 3))
]

# Function to apply individual transformations sequentially
def apply_transform(image, transform):
    if isinstance(image, torch.Tensor):
        image = F.to_pil_image(image)
    return transform(image)

# Function to visualize augmentations with names and save the image
def visualize_augmentations_with_names(dataset, num_samples=2, num_augmentations=5, save_path='augmentations_visualized.png'):
    fig, axes = plt.subplots(num_samples, num_augmentations + 1, figsize=(15, 7))

    for i in range(num_samples):
        # Load an image and its label from the dataset
        image, label = dataset[i]

        # Plot the original image in the first column
        axes[i, 0].imshow(image.permute(1, 2, 0))  # Convert tensor to HWC format
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Apply augmentations and plot in the next columns
        for j in range(1, num_augmentations + 1):
            aug_name, aug_transform = train_transform_with_names[j+1]  # Skip Grayscale and Resize for visualization

            # Apply individual transformations
            augmented_image = apply_transform(image, aug_transform)
            
            # If tensor, convert back to PIL image for visualization
            if isinstance(augmented_image, torch.Tensor):
                augmented_image = F.to_pil_image(augmented_image)
            
            # Plot the augmented image
            axes[i, j].imshow(augmented_image)
            axes[i, j].set_title(aug_name)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)  # Save the figure as an image file
    plt.show()

# Visualize augmentations and save the image
visualize_augmentations_with_names(train_dataset, num_samples=2, num_augmentations=5, save_path='augmentations_visualized.png')


# Define the model using ConvNext-Tiny pretrained on ImageNet
class ConvNextTiny(nn.Module):
    def __init__(self, num_classes):
        super(ConvNextTiny, self).__init__()
        self.model = models.convnext_tiny(pretrained=True)  # Using convnext pretrained on ImageNet
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)  # Modify the fully connected layer
    
    def forward(self, x):
        return self.model(x)

# Training function with early stopping and logging
def train_model(batch_size, learning_rate):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = ConvNextTiny(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    checkpoint_path = f'model_bs{batch_size}_lr{learning_rate}.pth'

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        if early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Training
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        logger.info(f'Epoch [{epoch+1}/{num_epochs}] Complete. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping and model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)  # Save the best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (Batch Size: {batch_size}, LR: {learning_rate})')
    plt.legend()
    plt.savefig(f'loss_plot_bs{batch_size}_lr{learning_rate}.png')
    plt.show()

# Evaluation function
def evaluate_model(batch_size, learning_rate):
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = ConvNextTiny(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(f'model_bs{batch_size}_lr{learning_rate}.pth'))

    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    logger.info(f'Accuracy on test dataset (Batch Size: {batch_size}, LR: {learning_rate}): {accuracy:.2f}%')

    # Save the test accuracy
    with open('test_accuracy.txt', 'a') as f:
        f.write(f'Batch Size: {batch_size}, Learning Rate: {learning_rate}, Test Accuracy: {accuracy:.2f}%\n')

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.dataset.classes)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix (Batch Size: {batch_size}, LR: {learning_rate})")
    plt.savefig(f'confusion_matrix_bs{batch_size}_lr{learning_rate}.png')
    plt.show()

# Run hyperparameter tuning
for batch_size, learning_rate in itertools.product(batch_sizes, learning_rates):
    train_model(batch_size, learning_rate)
    evaluate_model(batch_size, learning_rate)
