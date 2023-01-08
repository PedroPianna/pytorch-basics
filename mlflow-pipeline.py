import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18 , ResNet18_Weights, efficientnet_b0, EfficientNet_B0_Weights
from net import Net
import mlflow

# connecting to mlflow server
mlflow.set_tracking_uri('file:///home/mlruns')

# check of GPU acceleration as available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# Load transformer
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load dataset and define dataloaders 
batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# net = resnet18(ResNet18_Weights.auto)
# net = efficientnet_b0(EfficientNet_B0_Weights.auto)
net = Net()
net.to(device)

epochs = 50
lr = 0.0001

# loss and optimizer 
import torch.optim as optim
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)


# start mlflow run
mlflow.set_experiment("CIFAR10")
with mlflow.start_run(run_name='custom_net') as run:
    
    mlflow.log_param("epochs", epochs)
    mlflow.log_param('learning_rate', lr)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("optimizer", optimizer)
    mlflow.log_param("criterion", criterion)
    mlflow.log_param("classes", classes)

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # train loop for the CNN 
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
            
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Accuracy: {100 * round(correct / total, ndigits=4)}%")
        mlflow.log_metric('test_accuracy', 100 * round(correct / total, ndigits=4))

        # saving model at each epoch
        PATH = './custom_net_cifar.pth'
        torch.save(net.state_dict(), PATH)

print('Finished Training')

# save model
torch.save(net.state_dict(), PATH)