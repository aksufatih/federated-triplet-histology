import os
import random
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, train_test_split

basic_aug = ['rt', 'tr', 'hf', 'vf']
augmentations = ['']
for r in range(1, len(basic_aug) + 1):
    for combo in combinations(basic_aug, r):
        augmentations.append('_'.join(combo))
# OVERRIDE MANUALLY
augmentations = ['rt_tr', 'rt_tr_hf', 'rt_tr_vf', 'rt_tr_hf_vf', 'rt_hf_vf', 'tr_hf_vf', 'rt_hf']


def prepare_histology(dname, root, label_path, out_root, desired_len=1000, clinical_path=None, class_no=2, nfolds=3, cut_off=100, n_slices=None):
    if cut_off is not None:
        out_path = os.path.join(out_root, f"histology_{nfolds}fold_{desired_len * class_no}_c{cut_off}.xlsx")
    elif n_slices is not None:
        out_path = os.path.join(out_root, f"histology_{nfolds}fold_{desired_len * class_no}_{n_slices}slices.xlsx")
    info = pd.read_csv(os.path.join(root, 'center_info.csv'))
    file_list = os.listdir(root)
    file_list.remove('center_info.csv')

    if dname == 'NSCLC-Radiomics':
        clinical = pd.read_csv(clinical_path)
        file_list = list(filter(
            lambda x: clinical[clinical['PatientID'] == x.split('_')[0]]['Histology'].item() in ['adenocarcinoma',
                                                                                                 'squamous cell carcinoma'],
            file_list))
    if cut_off is not None:
        file_list = list(filter(lambda x: int(info.loc[info['X'] == x]['area'].item()) > cut_off, file_list))

    pids = list(map(lambda x: x.split('_')[0], file_list))
    unique_pid = list(set(pids))

    if n_slices is not None:
        file_list = list(map(lambda x: '_'.join((x.split('_')[0], x.split('_')[1].split('.')[0].zfill(2)))+'.nrrd', file_list))
        temp_list = []
        for pid in unique_pid:
            pt_files = list(filter(lambda x: pid in x, file_list))
            pt_files.sort()
            if len(pt_files) <= n_slices:
                temp_list.extend(pt_files)
            else:
                temp_list.extend(pt_files[((len(pt_files) - n_slices) // 2):((len(pt_files)-n_slices) // 2) + n_slices])

        file_list = temp_list
        file_list = list(map(lambda x: '_'.join((x.split('_')[0], x.split('_')[1].split('.')[0].lstrip('0')))+'.nrrd', file_list))
    label_df = pd.read_csv(label_path)

    pid_labels = list(map(lambda x: label_df.loc[label_df['X'] == x]['y'].item(), unique_pid))
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
    with pd.ExcelWriter(out_path, engine='openpyxl', mode='w') as writer:
        for fold, (train_idx, test_idx) in enumerate(skf.split(unique_pid, pid_labels)):
            train_id, val_id = train_test_split(np.array(unique_pid)[train_idx],
                                                test_size=1 / (nfolds - 1),
                                                # To make the ratio of val and test sets equal
                                                stratify=np.array(pid_labels)[train_idx])
            test_id = np.array(unique_pid)[test_idx]

            train_files = list(filter(lambda x: x.split('_')[0] in train_id, file_list))
            val_files = list(filter(lambda x: x.split('_')[0] in val_id, file_list))
            test_files = list(filter(lambda x: x.split('_')[0] in test_id, file_list))

            train_labels = list(map(lambda x: label_df.loc[label_df['X'] == x.split('_')[0]]['y'].item(), train_files))
            val_labels = list(map(lambda x: label_df.loc[label_df['X'] == x.split('_')[0]]['y'].item(), val_files))
            test_labels = list(map(lambda x: label_df.loc[label_df['X'] == x.split('_')[0]]['y'].item(), test_files))

            val_pids = list(map(lambda x: x.split('_')[0], val_files))
            test_pids = list(map(lambda x: x.split('_')[0], test_files))

            class_dict = {}
            new_dict = {}
            aug_dict = {}

            for c in range(class_no):
                class_dict[c] = [f for f, label in zip(train_files, train_labels) if label == c]
                new_dict[c] = []
                aug_dict[c] = []

            for c in range(class_no):
                for i in range(desired_len // len(class_dict[c])):
                    new_dict[c].extend(class_dict[c])
                    aug_dict[c].extend([augmentations[i]] * len(class_dict[c]))
                new_dict[c].extend(random.sample(class_dict[c], desired_len % len(class_dict[c])))
                aug_dict[c].extend([augmentations[i + 1]] * (desired_len % len(class_dict[c])))

            new_files = []
            new_aug = []
            for c in range(class_no):
                new_files.extend(new_dict[c])
                new_aug.extend(aug_dict[c])
            new_labels = list(map(lambda x: label_df.loc[label_df['X'] == x.split('_')[0]]['y'].item(), new_files))
            new_pids = list(map(lambda x: x.split('_')[0], new_files))
            df_dict = {'train': pd.DataFrame({'X': new_files, 'y': new_labels, 'id': new_pids, 'aug': new_aug}),
                       'val': pd.DataFrame({'X': val_files, 'y': val_labels, 'id': val_pids, 'aug': ''}),
                       'test': pd.DataFrame({'X': test_files, 'y': test_labels, 'id': test_pids, 'aug': ''})}

            for phase in ['train', 'val', 'test']:
                df_dict[phase].to_excel(writer, sheet_name=f'Fold{fold + 1}_{phase}', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dn', '--dname', help='Name of the dataset')
    parser.add_argument('-dp', '--data_path', required=True, help='Path to data files')
    parser.add_argument('-lp', '--label_path', required=True, help='Path to label file')
    parser.add_argument('-cp', '--clinical_path', help='Path to clinical data file')
    parser.add_argument('-op', '--out_path', required=True, help='Path to folder of output file')
    parser.add_argument('-dl', '--desired_len', type=int, help='Desired length of the dataset')
    args = parser.parse_args()

    prepare_histology(args.dname, args.data_path, args.label_path, args.out_path, args.desired_len, args.clinical_path)

