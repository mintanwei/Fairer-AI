import os
import argparse
from OculoScope import OculoScope_ValDataset_Age, OculoScope_ValDataset_Gender
from sklearn.metrics import roc_curve, auc
import torch
import pandas as pd
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch import nn
import config
import numpy as np
import sys
import re
import config
import torchvision
from src_files.models.resnet101 import ResNet_CSRA
from torch.cuda.amp import autocast
from metrics.multi_evalute import cal_metrics

parser = argparse.ArgumentParser('--OculoScope fairness analysis')
parser.add_argument('--gpuid', type=str, default='2')
parser.add_argument('--model_choice', type=str, default="FairerOPTH", choices=['FairerOPTH', 'baseline'])
parser.add_argument('--model_path',
                    default='checkpoints/OculoScope/model_best.ckpt', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
print("CUDA_VISIBLE_DEVICES: ", args.gpuid)
print("model choice: ", args.model_choice)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_val_age_csv():
    # load data
    val_dataset = OculoScope_ValDataset_Age(data_dir=config.OculoScope_dir, transform=config.transform_val)
    print('\n-----Initial Dataset Information-----')
    print('num images in val_dataset     : {}'.format(len(val_dataset)))
    print('dataset all symptom classes: %s' % str(val_dataset.all_classes))
    print('-------------------------------------')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # load model
    model = ResNet_CSRA(num_heads=4, lam=0.4,
                        num_classes=len(val_dataset.all_Pathology_classes),
                        num_Symptom_classes=len(val_dataset.all_classes),
                        cutmix=None)
    model = nn.DataParallel(model)
    state = torch.load(os.path.join(os.getcwd(), args.model_path), map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()
    model = model.cpu()
    model = model.cuda().half().eval()

    # inference
    Sig = torch.nn.Sigmoid()
    img_names = []
    preds_regular = []
    targets = []
    group_labels = []

    for i, (inputData, target, Symptom_target, name, group_label) in enumerate(val_loader):
        with torch.no_grad():
            with autocast():
                output_regular, output_regular1, _ = model(inputData.cuda(), False)
                output_regular = Sig(output_regular).cpu()

        preds_regular.append(output_regular.cpu().detach())
        targets.append(Symptom_target.cpu().detach())

        group_labels.extend(list(group_label))
        img_names.extend(list(name))

    val_age_df = pd.DataFrame()
    val_age_df['prediction'] = torch.cat(preds_regular).numpy().tolist()
    val_age_df['target'] = torch.cat(targets).numpy().tolist()
    val_age_df['group label'] = group_labels
    val_age_df['img name'] = img_names

    save_path = os.path.join(os.getcwd(), 'fairness_analysis', args.model_choice, 'val_age_res.csv')
    save_dir = os.path.dirname(save_path)
    mkdir(save_dir)
    val_age_df.to_csv(save_path, index=False)


def generate_val_gender_csv():
    # load data
    val_dataset = OculoScope_ValDataset_Gender(data_dir=config.OculoScope_dir, transform=config.transform_val)
    print('\n-----Initial Dataset Information-----')
    print('num images in val_dataset     : {}'.format(len(val_dataset)))
    print('dataset all symptom classes: %s' % str(val_dataset.all_classes))
    print('-------------------------------------')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # load model
    model = ResNet_CSRA(num_heads=4, lam=0.4,
                        num_classes=len(val_dataset.all_Pathology_classes),
                        num_Symptom_classes=len(val_dataset.all_classes),
                        cutmix=None)
    model = nn.DataParallel(model)
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()
    model = model.cpu()
    model = model.cuda().half().eval()

    # inference
    Sig = torch.nn.Sigmoid()
    img_names = []
    preds_regular = []
    targets = []
    group_labels = []

    for i, (inputData, target, Symptom_target, name, group_label) in enumerate(val_loader):
        with torch.no_grad():
            with autocast():
                output_regular, output_regular1, _ = model(inputData.cuda(), False)
                output_regular = Sig(output_regular).cpu()

        preds_regular.append(output_regular.cpu().detach())
        targets.append(Symptom_target.cpu().detach())

        group_labels.extend(list(group_label))
        img_names.extend(list(name))

    val_gender_df = pd.DataFrame()
    val_gender_df['prediction'] = torch.cat(preds_regular).numpy().tolist()
    val_gender_df['target'] = torch.cat(targets).numpy().tolist()
    val_gender_df['group label'] = group_labels
    val_gender_df['img name'] = img_names

    save_path = os.path.join(os.getcwd(), 'fairness_analysis', args.model_choice, 'val_gender_res.csv')
    save_dir = os.path.dirname(save_path)
    mkdir(save_dir)
    val_gender_df.to_csv(save_path, index=False)


if __name__ == '__main__':
    print('generate prediction on validation set with age annotation...')
    generate_val_age_csv()
    print("calculate metrics on age groups...")
    age_res_dict = cal_metrics(
        csv_path=os.path.join(os.getcwd(), 'fairness_analysis', args.model_choice, 'val_age_res.csv'), mode="age",
        save_dir=os.path.join(os.getcwd(), 'fairness_analysis', args.model_choice))

    print('generate prediction on validation set with gender annotation...')
    generate_val_gender_csv()
    print("calculate metrics on age groups...")
    gender_res_dict = cal_metrics(
        csv_path=os.path.join(os.getcwd(), 'fairness_analysis', args.model_choice, 'val_gender_res.csv'), mode="age",
        save_dir=os.path.join(os.getcwd(), 'fairness_analysis', args.model_choice))
