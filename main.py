import argparse
import copy

import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import wandb

from evaluate import eval_triplet, eval_softmax
from train import train_triplet, train_softmax
from utils import CustomDataset, CustomTripletLoss, log_results, init_model

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def run(args):

    for fold in range(args.folds):

        wandb.init(project='federated', name=f'E{args.exp}_F{fold}', config=args)
        wandb.define_metric("custom_step")
        wandb.define_metric("Model*", step_metric="custom_step")
        dataset = {}
        dataloader = {}
        model = {}
        best_model_weights = {}
        best_acc = {}
        optimizer = {}
        criterion = {'test': CustomTripletLoss(0.5, device) if args.mode == 'triplet' else torch.nn.CrossEntropyLoss()}
        scheduler = {}
        class_num = 0
        phases = ['train', 'val', 'test'] if args.val else ['train', 'test']

        # Initialize datasets and dataloaders
        for n in range(args.number_of_datasets):
            best_acc[n] = 0
            dataset[n] = {}
            dataloader[n] = {}
            for p in phases:
                df = pd.read_excel(args.fold_paths[n], sheet_name=f'Fold{fold + 1}_{p}', keep_default_na=False)
                dataset[n][p] = CustomDataset(args.roots[n], df)
                dataloader[n][p] = DataLoader(dataset[n][p], batch_size=args.batch_size, shuffle=p != 'test', pin_memory=True)
                class_num = max(class_num, dataset[n][p].class_num)

        if args.mode == 'softmax':
            args.feature_size = class_num

        # Initialize models, optimizers, losses and schedulers
        for n in range(args.number_of_datasets):
            model[n] = init_model(args.model_name, args.feature_size, args.freeze).to(device)
            optimizer[n] = torch.optim.AdamW(model[n].parameters(), lr=args.lr, weight_decay=0.001)
            criterion[n] = CustomTripletLoss(0.5, device) if args.mode == 'triplet' else torch.nn.CrossEntropyLoss()
            scheduler[n] = StepLR(optimizer[n], step_size=max(1, args.outer_epochs//5))

        # Federation rounds
        for oe in range(args.outer_epochs):
            SD = {}
            # Training phase
            for n in range(args.number_of_datasets):

                for ie in range(args.inner_epochs):
                    if args.mode == 'triplet':

                        loss, acc = train_triplet(dataloader[n]['train'], model[n], optimizer[n], criterion[n], device)
                    else:
                        loss, acc = train_softmax(dataloader[n]['train'], model[n], optimizer[n], criterion[n], device)
                    print(f'Epoch {oe*args.inner_epochs+ie} Model{n}_train_loss: {loss}, Model{n}_train_acc: {acc}')
                    wandb.log({f'Model{n}_train_loss': loss, f'Model{n}_train_acc': acc, 'custom_step': oe * args.inner_epochs + ie})

                SD[n] = copy.deepcopy(model[n].state_dict())

            # Average weights
            if not args.train_separately:
                SD['test'] = copy.deepcopy(model[0].state_dict())

                for key in model[0].state_dict():
                    param_list = [SD[n][key] for n in range(args.number_of_datasets)]
                    SD['test'][key] = sum(param_list) / len(param_list)

                for n in range(args.number_of_datasets):
                    model[n].load_state_dict(SD['test'])

            # Validation phase
            if args.val:
                for n in range(args.number_of_datasets):

                    if args.mode == 'triplet':
                        loss, labels, scores, preds, pids = eval_triplet(dataloader[n]['train'], dataloader[n]['val'], model[n], criterion['test'], device)
                    else:
                        loss, labels, scores, preds, pids = eval_softmax(dataloader[n]['val'], model[n], criterion['test'], device)

                    acc = log_results('val', loss, pids, labels, scores, preds, args, fold, n, class_num, oe, ie)

                    if acc >= best_acc[n]:
                        best_model_weights[n] = copy.deepcopy(model[n].state_dict())
                        best_acc[n] = acc
                    scheduler[n].step()
            else:
                for n in range(args.number_of_datasets):
                    best_model_weights[n] = copy.deepcopy(model[n].state_dict())
                    scheduler[n].step()

        # Test phase
        for n in range(args.number_of_datasets):

            model[n].load_state_dict(best_model_weights[n])

            if args.mode == 'triplet':
                loss, labels, scores, preds, pids = eval_triplet(dataloader[n]['train'], dataloader[n]['test'], model[n],
                                                           criterion['test'], device)
            else:
                loss, labels, scores, preds, pids = eval_softmax(dataloader[n]['test'], model[n], criterion['test'], device)

            log_results('test', loss, pids, labels, scores, preds, args, fold, n, class_num)

        wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-rp', '--roots', nargs='*', required=True, help='Path to root folders of the datasets')
    parser.add_argument('-fp', '--fold_paths', nargs='*', required=True, help='Path to fold files')
    parser.add_argument('-resp', '--result_path', required=True, help='Results filename')
    parser.add_argument('-n', '--number_of_datasets', default=2, type=int, help='Number of datasets')
    parser.add_argument('-f', '--folds', default=5, type=int, help='Number of folds')
    parser.add_argument('-bs', '--batch_size', default=32, type=int, help='Number of folds')
    parser.add_argument('-lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('-oe', '--outer_epochs', default=50, type=int, help='Outer epochs')
    parser.add_argument('-ie', '--inner_epochs', default=5, type=int, help='Inner epochs')
    parser.add_argument('-m', '--model_name', default='resnet18', help='Name of the model')
    parser.add_argument('-mode', default='triplet', help='softmax / triplet')
    parser.add_argument('-fs', '--feature_size', default=32, type=int, help='Size of the output feature vector')
    parser.add_argument('-exp', required=True, type=int, help='Exp ID')
    parser.add_argument('-val', action='store_true', help='Validation')
    parser.add_argument('-freeze', action='store_true', help='Freeze layers')
    parser.add_argument('-train_separately', action='store_true', help='Train separately')
    args = parser.parse_args()

    run(args)
