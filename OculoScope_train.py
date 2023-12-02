from src_files.models.resnet101 import ResNet_CSRA

import os
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch import nn
from torch.optim import lr_scheduler
import torch.nn.functional as F

import config
from OculoScope import OculoScope_TrainDataset, OculoScope_ValDataset
from src_files.helper_functions.helper_functions import ModelEma, \
    add_weight_decay, calc_average_precision, get_roc_auc_score, evaluation, setup_seed
from src_files.loss_functions.losses import AsymmetricLoss
from src_files.helper_functions.log import create_logger
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser(description='PyTorch FairerOPTH Training on OculoScope')
parser.add_argument('--gpuid', type=str, default='0,1')
parser.add_argument('--seed', type=int, default=21)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
setup_seed(args.seed)
print("CUDA_VISIBLE_DEVICES:", args.gpuid)
print("Random Seed: ", args.seed)
model_save_dir = os.path.join(os.getcwd(), 'saved_models', 'OculoScope')
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)


def main():
    log, logclose = create_logger(os.path.join(model_save_dir, 'training_record.log'))

    # initiate dataset
    Train_dataset = OculoScope_TrainDataset(data_dir=config.OculoScope_dir, transform=config.transform_train)
    Val_dataset = OculoScope_ValDataset(data_dir=config.OculoScope_dir, transform=config.transform_val)
    log('\n-----Initial Dataset Information-----')
    log('num images in train_dataset   : {}'.format(len(Train_dataset)))
    log('num images in val_dataset     : {}'.format(len(Val_dataset)))
    log('-------------------------------------')

    # Pytorch Dataloader
    train_loader = torch.utils.data.DataLoader(
        Train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        Val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    log('\n-----Initial Batchloaders Information -----')
    log('num batches in train_loader: {}'.format(len(train_loader)))
    log('num batches in val_loader  : {}'.format(len(val_loader)))
    log("num Symptom class: %d" % len(Train_dataset.all_classes))
    log("num Pathology class: %d" % len(Train_dataset.all_Pathology_classes))
    log("num val Symptom class: %d " % len(Val_dataset.all_classes))
    log("num val Pathology class: %d" % len(Val_dataset.all_Pathology_classes))
    log('-------------------------------------------')

    # define model
    model = ResNet_CSRA(num_heads=4, lam=0.4,
                        num_classes=len(Train_dataset.all_Pathology_classes),
                        num_Symptom_classes=len(Train_dataset.all_classes),
                        cutmix=None)
    model.cuda()
    if torch.cuda.device_count() > 1:
        print("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[i for i in torch.cuda.device_count()])

    # Actuall Training
    train_multi_label_coco(model, train_loader, val_loader, args.lr, log)
    logclose()


def train_multi_label_coco(model, train_loader, val_loader, lr, log):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 100
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    loss_func = F.mse_loss
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.3)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()

    model.train()
    for epoch in range(Epochs):
        for i, (inputData, target, Symptom_target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()
            Symptom_target = Symptom_target.cuda()
            with autocast():  # mixed precision
                output1, output2, output3, x1, x2 = model(inputData, True)  # sigmoid will be done in loss !
                output1, output2, output3 = output1.float(), output2.float(), output3.float()
                x1, x2 = x1.float(), x2.float()
            loss1 = criterion(output1, target)
            loss2 = criterion(output2, Symptom_target)
            all_target = torch.hstack((target, Symptom_target))
            loss3 = criterion(output3, all_target)
            loss4 = loss_func(x1, x2, reduction="sum")
            loss = 0.01 * loss1 + loss2 + 0.01 * loss3 + 0.00001 * loss4
            model.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                log('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'.format(epoch, Epochs, str(i).zfill(3),
                                                                                  str(steps_per_epoch).zfill(3),
                                                                                  scheduler.get_last_lr()[0],
                                                                                  loss.item()))
        try:
            torch.save(model.state_dict(), os.path.join(
                model_save_dir, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        except:
            pass

        model.eval()

        mAP_score = val_multi(val_loader, model, log)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            try:
                torch.save(model.state_dict(), os.path.join(
                    model_save_dir, 'model_best.ckpt'))
            except:
                pass
        log('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))
    log('highest_mAP = {:.2f}\n'.format(highest_mAP))


def val_multi(val_loader, model, log):
    log("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    targets = []
    for i, (input, target, Symptom_target, name) in enumerate(val_loader):
        with torch.no_grad():
            with autocast():
                output_regular, output_regular1, _ = model(input.cuda(), False)
                output_regular = Sig(output_regular).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        targets.append(Symptom_target.cpu().detach())

    mAP_score_regular = calc_average_precision(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy(), log)

    specificity = dict()
    sensitivity = dict()

    for i in range(37):
        specificity[i], sensitivity[i] = evaluation(torch.cat(targets).numpy()[:, i],
                                                    torch.cat(preds_regular).numpy()[:, i], 3)
    log("symptom_specificity: %s" % str(specificity))
    log("symptom_sensitivity:%s" % str(sensitivity))
    mean_specificity = sum(specificity.values())
    mean_specificity = mean_specificity / len(specificity)
    mean_sensitivity = sum(sensitivity.values())
    mean_sensitivity = mean_sensitivity / len(sensitivity)
    log("symptom_mean_specificity: %s" % str(mean_specificity))
    log("symptom_mean_sensitivity: %s" % str(mean_sensitivity))
    auc = get_roc_auc_score(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy(), log)
    log("symptom_auc: %s" % str(auc))
    return mAP_score_regular


if __name__ == '__main__':
    main()
