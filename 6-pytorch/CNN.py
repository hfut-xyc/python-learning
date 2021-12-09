import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
NUM_EPOCH = 5
BATCH_SIZE = 100
LEARNING_RATE = 0.001

# MNIST dataset
train_data = torchvision.datasets.MNIST(
    root='../data',    
    train=True, 
    transform=transforms.ToTensor(),
    download=False
)

test_data = torchvision.datasets.MNIST(
    root='../data', 
    train=False, 
    transform=transforms.ToTensor()
)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, 
    BATCH_SIZE=BATCH_SIZE, 
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data, 
    BATCH_SIZE=BATCH_SIZE, 
    shuffle=False
)

# Convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7 * 7 * 32, 10)
         
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = CNN().to(device)

# Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
total_step = len(train_loader)
for epoch in range(NUM_EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        output = model(x)
        loss = loss_func(output, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 100 == 0:
            print ('Epoch [{}], Step [{}], Loss: {:.4f}'.format(epoch+1, step+1, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')