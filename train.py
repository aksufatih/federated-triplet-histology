import torch


def train_triplet(dataloader, model, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for data in dataloader:
        images, labels, pids = data[0].to(device), data[1].to(device), data[2]

        optimizer.zero_grad()

        outputs = model(images).flatten(start_dim=1)

        loss = criterion(outputs, labels, pids)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)

    return epoch_loss, 0


def train_softmax(dataloader, model, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    cnt = 0
    for data in dataloader:
        images, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        epoch_acc += torch.sum(preds == labels).cpu().item()
        cnt += images.size(0)

    epoch_loss /= cnt
    epoch_acc /= cnt

    return epoch_loss, epoch_acc
