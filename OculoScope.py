import glob, os, sys, pdb, time

import numpy
import pandas as pd
import numpy as np
import cv2
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import argparse
import config
from copy import deepcopy

# OculoScope
symptom_class = ['isolated drusen', 'normal', 'laser spots', 'ah', 'floaters', 'vitreous opacity', 'aneurysms',
                 'vasculitis', 'maculopathy', 'brvo', 'cataract', 'choroidal diseases', 'fundus neoplasm',
                 'coats', 'optic abnormalities', 'crvo', 'pdr', 'erm', 'fevr', 'hm', 'trd', 'fibrosis',
                 'lens dislocation', 'mh', 'myelinated nerve fiber', 'pm',
                 'peripheral retinal degeneration', 'rd', 'retinal breaks', 'retinal white dots', 'rp',
                 'silicone oil', 'surgery-air', 'surgery-band:buckle', 'surgery-medicine', 'vkh',
                 'isolated vessel tortuosity', 'chorioretinitis']  # 38疾病修改后

pathology_class = ['white fundus mass', 'vessel attenuation', 'isolated chorioretinal atrophy', 'vascular occlusion',
                   'peripheral retinal degeneration', 'floaters', 'macular edema', 'isolated vessel tortuosity',
                   'cataract', 'erm', 'preretinal fibrosis',
                   'exudates', 'vitreous haemorrhage', 'laser spots', 'retinal white dots',
                   'congenital disc abnormality', 'retinal haemorrhage', 'bone-spicule-pigmentation',
                   'retinal opacities', 'surgery-band:buckle',
                   'hm', 'geographic macular atrophy', 'retinal breaks', 'gray-brown retinal mass',
                   'surgery-cryotherapy',
                   'nv', 'subretinal fibrosis', 'cotton-wool spots', 'macular haemorrhage', 'disc swelling',
                   'hemangioma', 'fundus mass',
                   'macular atrophy', 'vitreous opacity', 'isolated drusen', 'hyaloid remnant',
                   'isolated hyperpigmentary', 'mh', 'sun-set glow',
                   'optic atrophy', 'serous-exudative rd', 'microaneurysm', 'rd', 'disc swelling and haemorrhage',
                   'yellow subretinal lesions',
                   'vitreous lens', 'optic nv', 'silicone oil', 'choroid coloboma', 'vasculitis',
                   'subretinal haemorrhage',
                   'trd', 'chrp', 'ps', 'choroiditis', 'ah', 'myelinated nerve fiber', 'surgery-air', 'dragged disc',
                   'nevus', 'preretinal haemorrhage', 'aneurysms', 'retinal folds', 'surgery-medicine', 'macular star',
                   'osteoma', 'dalen fuchs nodules']  # 38疾病修改后


class OculoScope_TrainDataset(Dataset):
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


class OculoScope_ValDataset(Dataset):
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


class OculoScope_ValDataset_Age(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir

        self.transform = transform

        # full dataframe including train_val and test set
        self.df = self.get_df()
        print('self.df.shape: {}'.format(self.df.shape))

        self.all_classes = sorted(deepcopy(symptom_class))
        self.all_Pathology_classes = sorted(deepcopy(pathology_class))

    def __getitem__(self, index):
        row = self.df.iloc[index, :]

        img = cv2.imread(row['image_links'])
        labels = str.split(row['Symptom Labels'], ',')
        Pathology_labels = str.split(row['Pathology Labels'], ',')
        name = row['Image Index']
        group_label = row['Age Group']

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

        return img, Pathology_target, target, name, group_label

    def get_df(self):
        csv_path1 = os.path.join(self.data_dir, 'Data_1.csv')
        df1 = pd.read_csv(csv_path1)
        df1["Age Group"] = "group1"
        csv_path2 = os.path.join(self.data_dir, 'Data_2.csv')
        df2 = pd.read_csv(csv_path2)
        df2["Age Group"] = "group2"
        csv_path3 = os.path.join(self.data_dir, 'Data_3.csv')
        df3 = pd.read_csv(csv_path3)
        df3["Age Group"] = "group3"
        csv_path4 = os.path.join(self.data_dir, 'Data_4.csv')
        df4 = pd.read_csv(csv_path4)
        df4["Age Group"] = "group4"
        csv_path5 = os.path.join(self.data_dir, 'Data_5.csv')
        df5 = pd.read_csv(csv_path5)
        df5["Age Group"] = "group5"
        csv_path6 = os.path.join(self.data_dir, 'Data_6.csv')
        df6 = pd.read_csv(csv_path6)
        df6["Age Group"] = "group6"
        csv_path7 = os.path.join(self.data_dir, 'Data_7.csv')
        df7 = pd.read_csv(csv_path7)
        df7["Age Group"] = "group7"
        csv_path8 = os.path.join(self.data_dir, 'Data_8.csv')
        df8 = pd.read_csv(csv_path8)
        df8["Age Group"] = "group8"
        csv_path9 = os.path.join(self.data_dir, 'Data_9.csv')
        df9 = pd.read_csv(csv_path9)
        df9["Age Group"] = "group9"

        all_xray_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9])

        df = pd.DataFrame()
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, '*', '*'))]

        df['Image Index'] = df['image_links'].apply(lambda x: x.split('/')[-1])
        merged_df = df.merge(all_xray_df, how='inner', on=['Image Index'])
        merged_df = merged_df[['image_links', 'Symptom Labels', 'Pathology Labels', 'Image Index', "Age Group"]]
        return merged_df

    def __len__(self):
        return len(self.df)


class OculoScope_ValDataset_Gender(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir

        self.transform = transform

        # full dataframe including train_val and test set
        self.df = self.get_df()
        print('self.df.shape: {}'.format(self.df.shape))

        self.all_classes = sorted(deepcopy(symptom_class))
        self.all_Pathology_classes = sorted(deepcopy(pathology_class))

    def __getitem__(self, index):
        row = self.df.iloc[index, :]

        img = cv2.imread(row['image_links'])
        labels = str.split(row['Symptom Labels'], ',')
        Pathology_labels = str.split(row['Pathology Labels'], ',')
        name = row['Image Index']
        group_label = row['Gender Group']

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

        return img, Pathology_target, target, name, group_label

    def get_df(self):
        csv_path1 = os.path.join(self.data_dir, 'Data_female.csv')
        df1 = pd.read_csv(csv_path1)
        df1["Gender Group"] = "female"
        csv_path2 = os.path.join(self.data_dir, 'Data_male.csv')
        df2 = pd.read_csv(csv_path2)
        df2["Gender Group"] = "male"
        all_xray_df = pd.concat([df1, df2])

        df = pd.DataFrame()
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, '*', '*'))]

        df['Image Index'] = df['image_links'].apply(lambda x: x.split('/')[-1])
        merged_df = df.merge(all_xray_df, how='inner', on=['Image Index'])
        merged_df = merged_df[['image_links', 'Symptom Labels', 'Pathology Labels', 'Image Index', "Gender Group"]]
        return merged_df

    def __len__(self):
        return len(self.df)
