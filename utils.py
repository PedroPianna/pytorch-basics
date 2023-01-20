import torch

def validation(model, validloader, criterion):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        total_loss = 0
        correct, total = 0, 0
        for data in validloader:
            images, labels = data[0], data[1]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * round(float(correct / total), 3)
        loss = total_loss / len(validloader.sampler)
    return loss, accuracy