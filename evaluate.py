import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import torch


@torch.no_grad()
def eval_triplet(trainloader, testloader, model, criterion, device):
    model.eval()
    test_loss = 0
    features = {x: torch.FloatTensor().to(device) for x in ['train', 'test']}
    labels = {x: torch.LongTensor().to(device) for x in ['train', 'test']}
    pids = []
    dataloader = {'train': trainloader, 'test': testloader}
    for phase in ['train', 'test']:
        for data in dataloader[phase]:
            images, label, pid = data[0].to(device), data[1].to(device), data[2]

            outputs = model(images).flatten(start_dim=1)

            features[phase] = torch.cat((features[phase], outputs))
            labels[phase] = torch.cat((labels[phase], label))

            if phase == 'test':
                loss = criterion(outputs, label, pid)
                test_loss += loss.item()
                pids.extend(pid)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features['train'].cpu(), labels['train'].cpu())
    scores = knn.predict_proba(features['test'].cpu())
    preds = np.argmax(scores, axis=1)

    test_loss /= len(dataloader['test'])

    return test_loss, labels['test'].cpu().numpy(), scores, preds, np.array(pids)


@torch.no_grad()
def eval_softmax(dataloader, model, criterion, device):
    model.eval()
    scores = torch.FloatTensor().to(device)
    labels = torch.LongTensor().to(device)
    pids = []
    test_loss = 0

    for data in dataloader:
        images, label, pid = data[0].to(device), data[1].to(device), data[2]

        outputs = model(images)
        loss = criterion(outputs, label)

        scores = torch.cat((scores, torch.nn.functional.softmax(outputs, dim=1)))
        labels = torch.cat((labels, label))
        pids.extend(pid)

        test_loss += loss.item()

    test_loss /= len(dataloader)
    preds = torch.argmax(scores, dim=1)

    return test_loss, labels.cpu().numpy(), scores.cpu().numpy(), preds.cpu().numpy(), np.array(pids)
