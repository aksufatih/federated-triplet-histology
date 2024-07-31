import csv
import os
import random

import nrrd
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch
from torch.masked import masked_tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms.functional as tvf
import wandb


def init_model(model_name, out_features, freeze=False):

    if model_name == 'shufflenet':
        model = models.shufflenet_v2_x1_0(weights='ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1')
        if out_features == -1:
            model = torch.nn.Sequential(*list(model.children())[:-1], nn.AdaptiveAvgPool2d(1))
            freeze_until = '5'
        else:
            model.fc = nn.Linear(model.fc.in_features, out_features)
            freeze_until = 'conv5'

    elif model_name == 'resnet50':
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        if out_features == -1:
            model = torch.nn.Sequential(*list(model.children())[:-1])
            freeze_until = '7'
        else:
            model.fc = nn.Linear(model.fc.in_features, out_features)
            freeze_until = 'layer4'

    elif model_name == 'resnet18':
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        if out_features == -1:
            model = torch.nn.Sequential(*list(model.children())[:-1])
            freeze_until = '7'
        else:
            model.fc = nn.Linear(model.fc.in_features, out_features)
            freeze_until = 'layer4'

    else:
        raise Exception('Not implemented yet!')

    if freeze:
        for name, param in model.named_parameters():
            if freeze_until in name:
                break
            param.requires_grad = False

    return model


def log_results(phase, loss, pids, labels, scores, preds, args, fold, model, class_num, oe=0, ie=0):
    for based_on in ['patches', 'patients']:
        if based_on == 'patients':
            unique_ids = np.unique(pids)
            labels = np.array([labels[pids == pid].mean().astype(int) for pid in unique_ids])
            scores = np.array([scores[pids == pid].mean(axis=0) for pid in unique_ids])
            preds = np.array([1 if preds[pids == pid].mean() >= 0.5 else 0 for pid in unique_ids])

        acc = np.sum(preds == labels) / len(preds)
        if class_num == 2:
            auc = roc_auc_score(labels, scores[:, 1])
            wandb.log({f'Model{model}_{phase}_{based_on}_loss': loss, f'Model{model}_{phase}_{based_on}_acc': acc,
                       f'Model_{model}_{phase}_{based_on}_auc': auc, 'custom_step': oe * args.inner_epochs + ie})
        else:
            auc = 'N/A'
            wandb.log({f'Model{model}_{phase}_{based_on}_loss': loss, f'Model{model}_{phase}_{based_on}_acc': acc,
                       'custom_step': oe * args.inner_epochs + ie})

        if phase == 'val':
            return acc

        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        sensitivity = tp/(tp+fn+1e-15)
        specificity = tn/(tn+fp+1e-15)
        precision = tp/(tp+fp+1e-15)
        recall = sensitivity
        f1 = 2*precision*recall/(precision+recall+1e-15)
        gmean = np.sqrt(sensitivity*specificity)
        if not os.path.exists('./results'):
            os.mkdir('./results')
        is_file = os.path.isfile(f'./results/{args.result_path}.csv')
        with open(f'./results/{args.result_path}.csv', 'a', newline='') as csvfile:
            fieldnames = ['exp', 'fold', 'model', 'based_on', 'test_acc', 'test_auc', 'tn', 'fp', 'fn', 'tp',
                          'sensitivity', 'specificity', 'precision', 'recall', 'f1', 'gmean']

            logger = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not is_file:
                logger.writeheader()
            logger.writerow({'exp': args.exp,
                             'fold': fold,
                             'model': model,
                             'based_on': based_on,
                             'test_acc': acc,
                             'test_auc': auc,
                             'tn': tn,
                             'fp': fp,
                             'fn': fn,
                             'tp': tp,
                             'sensitivity': sensitivity,
                             'specificity': specificity,
                             'precision': precision,
                             'recall': recall,
                             'f1': f1,
                             'gmean': gmean})



class CustomTripletLoss:
    def __init__(self, margin, device):
        self.m = margin
        self.device = device

    def __call__(self, outputs, labels, pids):

        dist = torch.cdist(outputs, outputs)
        pids = np.array(pids)
        diff_pids = (pids[:, None] != pids[None, :])
        positives = (labels[:, None] == labels[None, :])#.fill_diagonal_(False)
        negatives = (labels[:, None] != labels[None, :])
        positives = positives * torch.from_numpy(diff_pids).to(self.device)

        px = torch.argmin(masked_tensor(dist.clone().detach(), positives), dim=1).to_tensor(0).to(torch.int64)
        nx = torch.argmin(masked_tensor(dist.clone().detach(), negatives), dim=1).to_tensor(0).to(torch.int64)

        dp = dist[torch.arange(dist.size(0)), px]
        dn = dist[torch.arange(dist.size(0)), nx]

        loss = torch.mean(F.relu(dp - dn + self.m))

        return loss


class CustomDataset(Dataset):

    def __init__(self, root, df, patch_size=64, transform=None):

        self.root = root
        self.df = df
        self.class_num = df.y.nunique()
        self.patch_size = patch_size
        self.transform = transform
        self.center_info = pd.read_csv(os.path.join(root, 'center_info.csv'))

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        img, _ = nrrd.read(os.path.join(self.root, self.df.iloc[idx]['X']))
        label = self.df.iloc[idx]['y']
        id = self.df.iloc[idx]['id']
        aug = self.df.iloc[idx]['aug']
        cnt_pt = self.center_info.loc[self.center_info['X'] == self.df.iloc[idx]['X']]['center'].item()
        cnt_pt = (int(cnt_pt.split(',')[0]), int(cnt_pt.split(',')[1]))

        img = np.clip(img, -600, 400)
        img = (img + 600) / 1000
        img = tvf.to_tensor(img).to(dtype=torch.float32)

        if 'rt' in aug:
            img = tvf.rotate(img, angle=random.randint(-90, 90), center=[cnt_pt[0], cnt_pt[1]])

        if 'tr' in aug:
            translations = random.choices(range(10), k=2)
            img = tvf.affine(img, angle=0, translate=translations, scale=1, shear=0)

        patch = tvf.crop(img, cnt_pt[1]-self.patch_size//2, cnt_pt[0]-self.patch_size//2, self.patch_size, self.patch_size)

        if 'hf' in aug:
            patch = tvf.hflip(patch)

        if 'vf' in aug:
            patch = tvf.vflip(patch)

        patch = patch.repeat((3, 1, 1))

        return patch, label, id
