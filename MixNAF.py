import glob, os, sys, pdb, time
import argparse
import numpy
import pandas as pd
import numpy as np
import cv2
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import config
from copy import deepcopy

# MixNAF
symptom_class = ['dr', 'normal', 'hm', 'disc swelling and haemorrhage', 'disc swelling', 'um',
                 'pm', 'maculopathy', 'cataract', 'vkh', 'optic atrophy', 'osteoma', 'rao',
                 'mocd', 'amd', 'serous-exudative rd']  # 窄角

pathology_class = ['microaneurysm', 'laser spots', 'exudates', 'retinal haemorrhage', 'tf', 'vessel tortuosity',
                   'disc swelling and haemorrhage', 'disc swelling', 'gray-brown retinal mass',
                   'ppa', 'drusen', 'cataract', 'serous-exudative rd', 'retinal folds', 'macular edema',
                   'optic atrophy',
                   'retinal opacities', 'cotton-wool spots', 'macular haemorrhage',
                   'hyperpigmentary']  # 窄角


class MixNAF_TrainDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir

        self.transform = transform

        # full dataframe including train_val and test set
        self.df = self.get_df()
        print('self.df.shape: {}'.format(self.df.shape))

        self.all_classes, self.all_classes_dict, self.all_Pathology_classes, self.all_Pathology_classes_dict = self.choose_the_indices()
        print('\nself.all_classes_dict: {}'.format(self.all_classes_dict))
        print('\nself.all_Pathology_classes_dict: {}'.format(self.all_Pathology_classes_dict))

    def __getitem__(self, index):
        row = self.df.iloc[index, :]

        img = cv2.imread(row['image_links'])
        labels = str.split(row['Symptom Labels'], ',')
        Pathology_labels = str.split(row['Pathology Labels'], ',')

        target = torch.zeros(len(self.all_classes))
        for lab in labels:
            lab = lab.strip().lower()
            if lab in symptom_class:
                lab_idx = self.all_classes.index(lab)
                target[lab_idx] = 1

        Pathology_target = torch.zeros(len(self.all_Pathology_classes))
        for lab in Pathology_labels:
            lab = lab.strip().lower()
            if lab in pathology_class:
                lab_idx = self.all_Pathology_classes.index(lab)
                Pathology_target[lab_idx] = 1

        if self.transform is not None:
            img = self.transform(img)

        return img, Pathology_target, target

    def choose_the_indices(self):

        max_examples_per_class = 10000  # its the maximum number of examples that would be sampled in the training set for any class
        all_classes = {}
        all_Pathology_classes = {}
        length = len(self.df)
        print('\nSampling the huuuge training dataset')
        for i in tqdm(list(np.random.choice(range(length), length, replace=False))):
            temp = str.split(self.df.iloc[i, :]['Symptom Labels'], ',')
            temp_Pathology = str.split(self.df.iloc[i, :]['Pathology Labels'], ',')

            temp = list(set(temp))
            temp_Pathology = list(set(temp_Pathology))

            # choose if multiple labels
            if len(temp) > 1:
                bool_lis = [False] * len(temp)
                # check if any label crosses the upper limit
                for idx, t in enumerate(temp):
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t in all_classes:
                        if all_classes[t] < max_examples_per_class:  # 500
                            bool_lis[idx] = True
                    else:
                        bool_lis[idx] = True
                # if all lables under upper limit, append
                if sum(bool_lis) == len(temp):
                    # maintain count
                    for t in temp:
                        t = t.strip().lower()
                        if t == '':
                            continue
                        if t not in symptom_class:
                            continue
                        if t not in all_classes:
                            all_classes[t] = 1
                        else:
                            all_classes[t] += 1
            else:  # these are single label images
                for t in temp:
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t not in symptom_class:
                        continue
                    if t not in all_classes:
                        all_classes[t] = 1
                    else:
                        if all_classes[t] < max_examples_per_class:  # 500
                            all_classes[t] += 1
            if len(temp_Pathology) > 1:
                bool_lis = [False] * len(temp_Pathology)
                # check if any label crosses the upper limit
                for idx, t in enumerate(temp_Pathology):
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t in all_Pathology_classes:
                        if all_Pathology_classes[t] < max_examples_per_class:  # 500
                            bool_lis[idx] = True
                    else:
                        bool_lis[idx] = True
                    # if all lables under upper limit, append
                    if sum(bool_lis) == len(temp_Pathology):
                        # maintain count
                        for t in temp_Pathology:
                            t = t.strip().lower()
                            if t == '':
                                continue
                            if t not in pathology_class:
                                continue
                            if t not in all_Pathology_classes:
                                all_Pathology_classes[t] = 1
                            else:
                                all_Pathology_classes[t] += 1
            else:  # these are single label images
                for t in temp_Pathology:
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t not in pathology_class:
                        continue
                    if t not in all_Pathology_classes:
                        all_Pathology_classes[t] = 1
                    else:
                        if all_Pathology_classes[t] < max_examples_per_class:  # 500
                            all_Pathology_classes[t] += 1

        return sorted(list(all_classes)), all_classes, sorted(list(all_Pathology_classes)), all_Pathology_classes

    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_train.csv')
        print('\n{} found: {}'.format(csv_path, os.path.exists(csv_path)))

        all_xray_df = pd.read_csv(csv_path)

        df = pd.DataFrame()
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, '*', '*'))]

        df['Image Index'] = df['image_links'].apply(lambda x: x.split('/')[-1])
        merged_df = df.merge(all_xray_df, how='inner', on=['Image Index'])
        merged_df = merged_df[['image_links', 'Symptom Labels', 'Pathology Labels']]
        return merged_df

    def __len__(self):
        return len(self.df)


class MixNAF_ValDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir

        self.transform = transform

        # full dataframe including train_val and test set
        self.df = self.get_df()
        print('self.df.shape: {}'.format(self.df.shape))

        self.all_classes, self.all_classes_dict, self.all_Pathology_classes, self.all_Pathology_classes_dict = self.choose_the_indices()

        print('\nself.val_all_classes_dict: {}'.format(self.all_classes_dict))
        print('\nself.val_all_Pathology_classes_dict: {}'.format(self.all_Pathology_classes_dict))

    def __getitem__(self, index):
        row = self.df.iloc[index, :]

        img = cv2.imread(row['image_links'])
        labels = str.split(row['Symptom Labels'], ',')
        Pathology_labels = str.split(row['Pathology Labels'], ',')
        name = row['Image Index']

        target = torch.zeros(len(self.all_classes))
        for lab in labels:
            lab = lab.strip().lower()
            if lab in symptom_class:
                lab_idx = self.all_classes.index(lab)
                target[lab_idx] = 1

        Pathology_target = torch.zeros(len(self.all_Pathology_classes))
        for lab in Pathology_labels:
            lab = lab.strip().lower()
            if lab in pathology_class:
                lab_idx = self.all_Pathology_classes.index(lab)
                Pathology_target[lab_idx] = 1

        if self.transform is not None:
            img = self.transform(img)

        return img, Pathology_target, target, name

    def choose_the_indices(self):

        max_examples_per_class = 10000  # its the maximum number of examples that would be sampled in the training set for any class

        all_classes = {}
        all_Pathology_classes = {}

        length = len(self.df)
        print('\nSampling the huuuge training dataset')
        for i in tqdm(list(np.random.choice(range(length), length, replace=False))):
            temp = str.split(self.df.iloc[i, :]['Symptom Labels'], ',')
            temp_Pathology = str.split(self.df.iloc[i, :]['Pathology Labels'], ',')

            temp = list(set(temp))
            temp_Pathology = list(set(temp_Pathology))

            # choose if multiple labels
            if len(temp) > 1:
                bool_lis = [False] * len(temp)
                # check if any label crosses the upper limit
                for idx, t in enumerate(temp):
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t in all_classes:
                        if all_classes[t] < max_examples_per_class:  # 500
                            bool_lis[idx] = True
                    else:
                        bool_lis[idx] = True
                # if all lables under upper limit, append
                if sum(bool_lis) == len(temp):
                    # maintain count
                    for t in temp:
                        t = t.strip().lower()
                        if t == '':
                            continue
                        if t not in symptom_class:
                            continue
                        if t not in all_classes:
                            all_classes[t] = 1
                        else:
                            all_classes[t] += 1
            else:  # these are single label images
                for t in temp:
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t not in symptom_class:
                        continue
                    if t not in all_classes:
                        all_classes[t] = 1
                    else:
                        if all_classes[t] < max_examples_per_class:  # 500
                            all_classes[t] += 1
            if len(temp_Pathology) > 1:
                bool_lis = [False] * len(temp_Pathology)
                # check if any label crosses the upper limit
                for idx, t in enumerate(temp_Pathology):
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t in all_Pathology_classes:
                        if all_Pathology_classes[t] < max_examples_per_class:  # 500
                            bool_lis[idx] = True
                    else:
                        bool_lis[idx] = True
                    # if all lables under upper limit, append
                    if sum(bool_lis) == len(temp_Pathology):
                        # maintain count
                        for t in temp_Pathology:
                            t = t.strip().lower()
                            if t == '':
                                continue
                            if t not in pathology_class:
                                continue
                            if t not in all_Pathology_classes:
                                all_Pathology_classes[t] = 1
                            else:
                                all_Pathology_classes[t] += 1
            else:  # these are single label images
                for t in temp_Pathology:
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t not in pathology_class:
                        continue
                    if t not in all_Pathology_classes:
                        all_Pathology_classes[t] = 1
                    else:
                        if all_Pathology_classes[t] < max_examples_per_class:  # 500
                            all_Pathology_classes[t] += 1

        return sorted(list(all_classes)), all_classes, sorted(list(all_Pathology_classes)), all_Pathology_classes

    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_val.csv')
        print('\n{} found: {}'.format(csv_path, os.path.exists(csv_path)))

        all_xray_df = pd.read_csv(csv_path)

        df = pd.DataFrame()
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, '*', '*'))]

        df['Image Index'] = df['image_links'].apply(lambda x: x.split('/')[-1])
        merged_df = df.merge(all_xray_df, how='inner', on=['Image Index'])
        merged_df = merged_df[['image_links', 'Symptom Labels', 'Pathology Labels', 'Image Index']]
        return merged_df

    def __len__(self):
        return len(self.df)
