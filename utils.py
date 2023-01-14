import torch

def validation(model, validloader, criterion):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        correct, total = 0, 0
        for data in validloader:
            images, labels = data[0], data[1]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * round(float(correct / total), 3)
    
    return loss, accuracy