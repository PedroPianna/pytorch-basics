import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from net import Net
import mlflow
import utils

PATH = 'models/convnet_large_cifar10.pth'
MLRUNS_FOLDER = 'file:///home/mlruns'

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
# net = densenet121(DenseNet121_Weights.auto)
# net = vgg11(VGG11_Weights.auto)
net = Net()
# net = convnext_base(ConvNeXt_Base_Weights.auto)
# net = convnext_large(ConvNeXt_Large_Weights.auto)
net.to(device)

epochs = 75
lr = 0.0001

# loss and optimizer 
import torch.optim as optim
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# connecting to mlflow server
mlflow.set_tracking_uri(MLRUNS_FOLDER)
# start mlflow run
mlflow.set_experiment("CIFAR10")
with mlflow.start_run(run_name='resnet18') as run:
    mlflow.log_param("epochs", epochs)
    mlflow.log_param('learning_rate', lr)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("optimizer", optimizer)
    mlflow.log_param("criterion", criterion)

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # train loop for the CNN 
    for epoch in range(epochs):  # loop over the dataset multiple times

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
            
        train_loss, train_accuracy = utils.validation(net, trainloader, criterion)
        test_loss, test_accuracy = utils.validation(net, testloader, criterion)

        # saving model at each epoch
        torch.save(net.state_dict(), PATH)
        mlflow.pytorch.log_model(net,'file:/home/mlruns')

        metrics = {
            "training_loss":train_loss,
            "train_accuracy":train_accuracy,
            "test_loss":test_loss,
            "test_accuracy":test_accuracy,
            "learning_rate":optimizer.param_groups[0]['lr'],
        }

        mlflow.log_metrics(metrics, step=epoch)

print('Finished Training')

# save final model
torch.save(net.state_dict(), PATH + '_final')