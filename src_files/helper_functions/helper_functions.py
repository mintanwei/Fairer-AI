from copy import deepcopy
import random
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, confusion_matrix
import torch


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速


def parse_args(parser):
    # parsing args
    args = parser.parse_args()
    return args


def calc_average_precision(y_true, y_score, log=None):
    aps = np.zeros(y_score.shape[1])

    for i in range(y_score.shape[1]):
        true = y_true[:, i]
        score = y_score[:, i]

        true[true == -1.] = 0

        ap = average_precision_score(true, score)
        aps[i] = ap
    if log is None:
        print("per-category:", aps)
        print("mAP: {:4f}".format(np.mean(aps)))
    else:
        log("per-category:%s" % str(aps))
        log("mAP: {:4f}".format(np.mean(aps)))
    return 100 * aps.mean()


def get_roc_auc_score(y_true, y_probs, log=None):
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
    if True:
        if log is None:
            print('\nuseful_classes_roc_auc_list', useful_classes_roc_auc_list)
        else:
            log('\nuseful_classes_roc_auc_list: %s' % str(useful_classes_roc_auc_list))

    return np.mean(np.array(useful_classes_roc_auc_list))


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


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
