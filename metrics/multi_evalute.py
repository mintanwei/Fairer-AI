import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, auc
import os
import ast
import pickle
from copy import deepcopy

age_group_label_idx_tfs = {'group1': 0, 'group2': 1, 'group3': 2, 'group4': 3, 'group5': 4, 'group6': 5, 'group7': 6,
                           'group8': 7, 'group9': 8}
gender_group_idx_tfs = {'female': 0, 'male': 1}

# OculoScope
symptom_class = ['isolated drusen', 'normal', 'laser spots', 'ah', 'floaters', 'vitreous opacity', 'aneurysms',
                 'vasculitis', 'maculopathy', 'brvo', 'cataract', 'choroidal diseases', 'fundus neoplasm',
                 'coats', 'optic abnormalities', 'crvo', 'pdr', 'erm', 'fevr', 'hm', 'trd', 'fibrosis',
                 'lens dislocation', 'mh', 'myelinated nerve fiber', 'pm',
                 'peripheral retinal degeneration', 'rd', 'retinal breaks', 'retinal white dots', 'rp',
                 'silicone oil', 'surgery-air', 'surgery-band:buckle', 'surgery-medicine', 'vkh',
                 'isolated vessel tortuosity', 'chorioretinitis']  # 38疾病修改后
idx2symptom_class_tfs = {i: key for i, key in enumerate(sorted(symptom_class))}


def create_logger(log_filename, display=True):
    f = open(log_filename, 'a')
    counter = [0]

    # this function will still have access to f after create_logger terminates
    def logger(text):
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
        # Question: do we need to flush()

    return logger, f.close


def find_optimal_cutoff(tpr, fpr, threshold):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    return optimal_threshold


def best_confusion_matrix(y_test, y_test_predprob):
    """
        根据真实值和预测值（预测概率）的向量来计算混淆矩阵和最优的划分阈值

        Args:
            y_test:真实值
            y_test_predprob：预测值

        Returns:
            返回最佳划分阈值和混淆矩阵
        """

    fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)
    cutoff = find_optimal_cutoff(tpr, fpr, thresholds)
    y_pred = list(map(lambda x: 1 if x >= cutoff else 0, y_test_predprob))
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    return cutoff, TN, FN, FP, TP


def evaluation(x_test, y_test_predprob, digits):
    test_cutoff, TN2, FN2, FP2, TP2 = best_confusion_matrix(x_test, y_test_predprob)

    # Sen Spe
    best_recall, best_prec = round(TP2 / (TP2 + FN2), digits), round(TN2 / (FP2 + TN2), digits)

    return best_prec, best_recall


def get_roc_auc_score(y_true, y_probs):
    '''
    Uses roc_auc_score function from sklearn.metrics to calculate the micro ROC AUC score for a given y_true and y_probs.
    '''

    class_roc_auc_list = []
    useful_classes_roc_auc_list = []

    for i in range(y_true.shape[1]):
        try:
            class_roc_auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        except ValueError:
            class_roc_auc = 0
        class_roc_auc_list.append(class_roc_auc)
        useful_classes_roc_auc_list.append(class_roc_auc)

    return np.mean(np.array(useful_classes_roc_auc_list))


def cal_metrics(csv_path, mode, save_dir=None):
    """
    calculate average accuracy, accuracy per category, PQD,DPM,EOM
    ===================================================================
    input:
    csv_path: val_age_res.csv or val_gender_res.csv
            ['img name','prediction','target','group_label']
    mode: 'age' or 'gender'
    ===================================================================
    return: dict,KEYS=
                    ['avg_acc','avg_acc_per_category','avg_acc_per_group','avg_acc_array',
                     'PQD','PQD_per_category,'DPM','EOM',
                     'avg_specificity','specificity_per_category','avg_specificity_per_group','specificity_array',
                     'avg_sensitivity','sensitivity_per_category','avg_sensitivity_per_group','sensitivity_array',
                     'avg_tpr','tpr_per_category','avg_tpr_per_group','tpr_array',
                     'avg_fpr','fpr_per_category','avg_fpr_per_group','fpr_array',
                     'avg_auc','auc_per_category','avg_auc_per_group','auc_array',
                     'mAP', 'AP_per_category','mAP_per_group','AP_array',
                     'deltaD_per_category','deltaA','deltaM']
    """

    assert mode.lower() in ['age', 'gender']
    type_num = 9 if mode.lower() == 'age' else 2
    group_label_idx_tfs = age_group_label_idx_tfs if mode.lower() == 'age' else gender_group_idx_tfs

    df = pd.read_csv(csv_path, converters={'prediction': ast.literal_eval, 'target': ast.literal_eval})
    df['target'] = df['target'].apply(np.array)
    df['prediction'] = df['prediction'].apply(np.array)
    category_num = len(df['target'][0])
    count_array = np.zeros((type_num, category_num))

    y_pred = [[] for i in range(type_num)]
    y_true = [[] for i in range(type_num)]

    for i in range(df.shape[0]):
        prediction = df.iloc[i]['prediction']
        target = df.iloc[i]['target']
        group_type_idx = group_label_idx_tfs[df.iloc[i]['group label']]
        count_array[group_type_idx, target == 1.] += 1

        y_pred[group_type_idx].append(prediction)
        y_true[group_type_idx].append(target)

    AP_array = np.zeros((type_num, category_num))  # (num_group,num_category)
    for i in range(type_num):
        y_true[i] = np.array(y_true[i])
        y_pred[i] = np.array(y_pred[i])
    for group_type_idx in range(type_num):
        for category_idx in range(category_num):
            AP_array[group_type_idx, category_idx] = average_precision_score(
                y_true[group_type_idx][:, category_idx], y_pred[group_type_idx][:, category_idx])

    # 'mAP', 'AP_per_category','mAP_per_group'
    # precision=tp/(tp+fp)
    mAP_per_group = np.nanmean(AP_array, axis=1)  # (num_group,)
    AP_per_category = np.zeros(category_num)
    for category_idx in range(category_num):
        AP_per_category[category_idx] = average_precision_score(
            np.concatenate([y_true[i][:, category_idx] for i in range(type_num)], axis=0),
            np.concatenate([y_pred[i][:, category_idx] for i in range(type_num)], axis=0)
        )
    mAP = np.nanmean(AP_per_category)

    # 'avg_specificity','specificity_per_category', 'specificity_array'
    # 'avg_sensitivity','sensitivity_per_category', 'sensitivity_array'
    # 'avg_FPR', 'FRP_per_category', 'FPR_array'
    # 'avg_FNR', 'FNR_per_category', 'FNR_array'
    # 'avg_roc_auc', 'roc_auc_per_category', 'roc_auc_array'
    # specificity=tn/(tn+fp)=TNR, FPR=FP/(FP+TN)=1-TNR=1-specificity
    # sensitivity=tp/(tp+fn)=TPR, FNR=FN/(FN+TP)=1-TPR=1-sensitivity

    fpr, tpr = dict(), dict()
    FPR_array, FNR_array, roc_auc_array, specificity_array, sensitivity_array = \
        np.zeros((type_num, category_num)), np.zeros((type_num, category_num)), np.zeros(
            (type_num, category_num)), np.zeros((type_num, category_num)), np.zeros((type_num, category_num))

    for group_type_idx in range(type_num):
        for category_idx in range(category_num):
            try:
                specificity_array[group_type_idx, category_idx], sensitivity_array[
                    group_type_idx, category_idx] = evaluation(y_true[group_type_idx][:, category_idx],
                                                               y_pred[group_type_idx][:, category_idx], 3)
            except ValueError:
                specificity_array[group_type_idx, category_idx], sensitivity_array[
                    group_type_idx, category_idx] = np.nan, np.nan
            FPR_array[group_type_idx, category_idx] = 1 - specificity_array[group_type_idx, category_idx]
            FNR_array[group_type_idx, category_idx] = 1 - sensitivity_array[group_type_idx, category_idx]

            try:
                fpr[group_type_idx, category_idx], tpr[group_type_idx, category_idx], _ = roc_curve(
                    y_true[group_type_idx][:, category_idx], y_pred[group_type_idx][:, category_idx])
                roc_auc_array[group_type_idx, category_idx] = auc(fpr[group_type_idx, category_idx],
                                                                  tpr[group_type_idx, category_idx])
            except ValueError:
                roc_auc_array[group_type_idx, category_idx] = np.nan

    fpr, tpr = dict(), dict()
    FPR_per_category, FNR_per_category, roc_auc_per_category, specificity_per_category, sensitivity_per_category = np.zeros(
        category_num), np.zeros(category_num), np.zeros(category_num), np.zeros(category_num), np.zeros(category_num)
    for category_idx in range(category_num):
        try:
            specificity_per_category[category_idx], sensitivity_per_category[category_idx] = evaluation(
                np.concatenate([y_true[i][:, category_idx] for i in range(type_num)], axis=0),
                np.concatenate([y_pred[i][:, category_idx] for i in range(type_num)], axis=0), 3)
        except ValueError:
            specificity_per_category[category_idx], sensitivity_per_category[category_idx] = np.nan, np.nan
        FPR_per_category[category_idx] = 1 - specificity_per_category[category_idx]
        FNR_per_category[category_idx] = 1 - sensitivity_per_category[category_idx]
        try:
            fpr[category_idx], tpr[category_idx], _ = roc_curve(
                np.concatenate([y_true[i][:, category_idx] for i in range(type_num)], axis=0),
                np.concatenate([y_pred[i][:, category_idx] for i in range(type_num)], axis=0))
            roc_auc_per_category[category_idx] = auc(fpr[category_idx], tpr[category_idx])
        except ValueError:
            roc_auc_per_category[category_idx] = np.nan

    avg_specificity = np.nanmean(specificity_per_category)
    avg_sensitivity = np.nanmean(sensitivity_per_category)
    avg_FPR = np.nanmean(FPR_per_category)
    avg_FNR = np.nanmean(FNR_per_category)
    avg_roc_auc = np.nanmean(roc_auc_per_category)

    # avg_acc_array, avg_acc_per_group,avg_acc_per_category, avg_acc
    avg_acc_array = np.zeros((type_num, category_num))  # (num_group,num_category)
    for group_type_idx in range(type_num):
        for category_idx in range(category_num):
            avg_acc_array[group_type_idx, category_idx] = (specificity_array[group_type_idx, category_idx] +
                                                           sensitivity_array[group_type_idx, category_idx]) * 0.5
    avg_acc_per_group = np.nanmean(avg_acc_array, axis=1)  # (num_group,)
    avg_acc_per_category = np.zeros(category_num)  # (num_category,)
    for category_idx in range(category_num):
        avg_acc_per_category[category_idx] = 0.5 * (
                sensitivity_per_category[category_idx] + specificity_per_category[category_idx])
    avg_acc = np.nanmean(avg_acc_per_category)

    # PQD, PQD_per_category
    PQD = np.nanmin(avg_acc_per_group) / np.nanmax(avg_acc_per_group)
    PQD_per_category = np.nanmin(avg_acc_array, axis=0) / np.nanmax(avg_acc_array, axis=0)

    # DPM, DPM_per_category
    TP_array, TN_array, FP_array, FN_array = np.zeros((type_num, category_num)), np.zeros(
        (type_num, category_num)), np.zeros((type_num, category_num)), np.zeros((type_num, category_num))
    for group_type_idx in range(type_num):
        for category_idx in range(category_num):
            try:
                _, tn, fn, fp, tp = best_confusion_matrix(
                    y_true[group_type_idx][:, category_idx], y_pred[group_type_idx][:, category_idx])
                TP_array[group_type_idx, category_idx] = tp
                TN_array[group_type_idx, category_idx] = tn
                FP_array[group_type_idx, category_idx] = fp
                FN_array[group_type_idx, category_idx] = fn
            except ValueError:
                TP_array[group_type_idx, category_idx] = np.nan
                TN_array[group_type_idx, category_idx] = np.nan
                FP_array[group_type_idx, category_idx] = np.nan
                FN_array[group_type_idx, category_idx] = np.nan

    demo_array = (TP_array + FP_array) / count_array.clip(1e-15)
    DPM_per_category = np.nanmin(demo_array, axis=0) / np.nanmax(demo_array, axis=0)
    if mode == "gender":
        DPM_per_category = np.delete(DPM_per_category, [sorted(deepcopy(symptom_class)).index('retinal white dots')],
                                     axis=0)
    DPM = np.nanmean(DPM_per_category)

    # EOM, EOM_per_category
    eo_array = TP_array / count_array.clip(1e-15)
    EOM_per_category = np.nanmin(eo_array, axis=0) / np.nanmax(eo_array, axis=0).clip(1e-15)
    if mode == "gender":
        EOM_per_category = np.delete(EOM_per_category, [sorted(deepcopy(symptom_class)).index('retinal white dots')],
                                     axis=0)
    EOM = np.nanmean(EOM_per_category)

    # 'deltaD_per_category','deltaA','deltaM'
    deltaD_per_category = (np.nanmax(AP_array, axis=0) - np.nanmin(AP_array, axis=0)) / np.nanmean(
        AP_array, axis=0).clip(1e-15)
    if mode == "gender":
        deltaD_per_category = np.delete(deltaD_per_category,
                                        [sorted(deepcopy(symptom_class)).index('retinal white dots')], axis=0)
    deltaA = np.nanmean(deltaD_per_category)
    deltaM = np.nanmax(deltaD_per_category)

    res_dict = {'avg_acc': avg_acc, 'avg_acc_per_category': avg_acc_per_category,
                'avg_acc_per_group': avg_acc_per_group, 'avg_acc_array': avg_acc_array,

                'TP_array': TP_array, 'FP_array': FP_array, 'TN_array': TN_array, 'FN_array': FN_array,
                'count_array': count_array,
                'PQD': PQD, 'PQD_per_category': PQD_per_category,
                'DPM': DPM, 'DPM_per_category': DPM_per_category,
                'EOM': EOM, 'EOM_per_category': EOM_per_category,

                'avg_specificity': avg_specificity, 'specificity_per_category': specificity_per_category,
                'specificity_array': specificity_array,
                'avg_sensitivity': avg_sensitivity, 'sensitivity_per_category': sensitivity_per_category,
                'sensitivity_array': sensitivity_array,
                'avg_FPR': avg_FPR, 'FPR_per_category': FPR_per_category, 'FPR_array': FPR_array,
                'avg_FNR': avg_FNR, 'FNR_per_category': FNR_per_category, 'FNR_array': FNR_array,
                'avg_auc': avg_roc_auc, 'auc_per_category': roc_auc_per_category, 'roc_auc_array': roc_auc_array,

                'mAP': mAP, 'AP_per_category': AP_per_category, 'mAP_per_group': mAP_per_group, 'AP_array': AP_array,
                'deltaD_per_category': deltaD_per_category, 'deltaA': deltaA, 'deltaM': deltaM,
                'idx2symptom_class_tfs': idx2symptom_class_tfs}
    if save_dir is None:
        save_dir = os.getcwd()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'val_metrics_on_%s.pkl' % mode), 'wb') as f:
        pickle.dump(res_dict, f)
    log, logclose = create_logger(os.path.join(save_dir, 'val_metrics_on_%s.log' % mode))
    log('==' * 15)
    for key in res_dict.keys():
        log("%s:\n" % key)
        log("%s" % str(res_dict[key]))
        log('--' * 15)
    log('==' * 15)
    logclose()
    return res_dict
