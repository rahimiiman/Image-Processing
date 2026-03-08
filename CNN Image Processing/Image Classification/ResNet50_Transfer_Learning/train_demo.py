import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt


### Hyperparameters
num_classes = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 15

#overfit prevention hyperparameters
overfitting_threshold = 0.1
dropout_rate = 0.2
weight_decay = 1e-4

### Load Data
Folder_PATH = r"E:\Machine_Learning\Machine Vision\Image Processing\CNN Image Processing\Image Classification\dataset_flowers"

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load the full dataset
full_dataset = datasets.ImageFolder(root=Folder_PATH, transform=transform)

# Calculate split sizes (60% train, 20% val, 20% test)
total_size = len(full_dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Print dataset info
print(f"Number of classes: {len(full_dataset.classes)}")
print(f"Classes: {full_dataset.classes}")
print(f"Total images: {total_size}")
print(f"Train set: {train_size} images")
print(f"Validation set: {val_size} images")
print(f"Test set: {test_size} images")


### Model Setup
model = resnet50(pretrained=True)
# Modify the final layer for our number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, verbose=True
)


### Evaluation Function (for validation and testing)
def evaluate_model(model, data_loader, criterion, return_predictions=False):
    """
    Evaluate the model on a given dataset
    """
    model.eval()
    eval_loss = 0.0
    eval_correct = 0
    eval_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            eval_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            eval_total += labels.size(0)
            eval_correct += (predicted == labels).sum().item()
            
            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    eval_acc = 100 * eval_correct / eval_total
    eval_loss = eval_loss / len(data_loader)
    
    if return_predictions:
        return eval_acc, eval_loss, all_preds, all_labels
    return eval_acc, eval_loss


### Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, patience=3):
    """
    Train the model with monitoring, early stopping, and learning rate scheduling
    """
    best_val_acc = 0.0
    patience_counter = 0
    
    # Training history for plotting
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Progress bar for training batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=True)
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar with current metrics
            current_loss = train_loss / (train_total / BATCH_SIZE + 1)
            pbar.set_postfix({'loss': f"{current_loss:.4f}"})
        
        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        scheduler.step(loss)  # Step the scheduler based on training loss
        print(f"\scheduler step - current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        # Validation phase with progress bar
        val_acc, val_loss = evaluate_model(model, val_loader, criterion)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print detailed metrics
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"⚠ No improvement. Patience: {patience_counter}/{patience}")
        
        # Learning rate scheduler step
        scheduler.step(val_acc)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\n{'='*60}")
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}\n")
    
    return history



### Testing Function
def test_model(model, test_loader, criterion):
    """
    Test the model on the test set with detailed report
    """
    test_acc, test_loss, all_preds, all_labels = evaluate_model(model, test_loader, criterion, return_predictions=True)
    
    print("Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Per-class accuracy
    class_names = full_dataset.classes
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return test_acc, test_loss


# Example usage:
if __name__ == "__main__":
    # Train the model
    print("Starting training...\n")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, patience=3)
    
    # Plot training history
    print("\nPlotting training history...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100)
    print("Training history plot saved as 'training_history.png'")
    plt.show()
    
    # Load best model for testing
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Test the model
    print("\nTesting the model...")
    test_accuracy, test_loss = test_model(model, test_loader, criterion)

