import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from torchvision import transforms
import matplotlib.pyplot as plt


class VirusDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(self.root_dir)
        self.data = []

        for class_ in self.classes:
            class_dir = os.path.join(self.root_dir, class_)
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                if os.path.isfile(file_path): 
                    self.data.append((class_, file_path))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, img_path = self.data[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, self.classes.index(label)


class ConvNet(nn.Module):
    def __init__(self, num_classes, num_layers, activation):
        super(ConvNet, self).__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList()  
        self.activation = activation

        # Funciones de activacion
        if self.activation == 'relu':
            activation_function = nn.ReLU()
        elif self.activation == 'sigmoid':
            activation_function = nn.Sigmoid()
        elif self.activation == 'tanh':
            activation_function = nn.Tanh()
        else:
            raise ValueError("Invalid activation function. Choose from 'relu', 'sigmoid', 'tanh'.")

        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(16),
                    activation_function,
                    nn.MaxPool2d(kernel_size=2, stride=2)))
            elif i == 1:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(32),
                    activation_function,
                    nn.MaxPool2d(kernel_size=2, stride=2)))
            else:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(32),
                    activation_function,
                    nn.MaxPool2d(kernel_size=2, stride=2)))
        self.fc = None 

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        if self.fc is None: 
            n_features = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc = nn.Linear(n_features, self.num_classes)
            
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(model, criterion, optimizer_choice, lr, dataloader, num_epochs):
    model.train()
    losses = []
    accuracies = []

    #Aqui se puede escoger entre adam y sgd
    if optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer choice. Choose from 'adam', 'sgd'.")
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Loss and accuracy calculation
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions.double() / len(dataloader.dataset)
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, epoch_acc))

    return losses, accuracies


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),  
    transforms.ToTensor()
])

train_dataset = VirusDataset(root_dir='virus/dataset/augmented_train', transform=transform)
validation_dataset = VirusDataset(root_dir='virus/dataset/validation', transform=transform)
test_dataset = VirusDataset(root_dir='virus/dataset/test', transform=transform)  # Define the test dataset



train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Cambiar aqui
num_layers_list = [3]   

activations = ['relu','sigmoid','tanh']
optimizers = ['adam','sgd']
learning_rates = [0.1]

