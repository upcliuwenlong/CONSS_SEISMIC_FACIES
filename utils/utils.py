import logging
import sys
import random
import torch
import numpy as np

loggers = {}
def get_logger(name,log_path='./', level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(log_path+'/log.txt', encoding='UTF-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        loggers[name] = logger
        return logger

def fix_seed(seed):
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)

class runningScore(object):

    def __init__(self, n_classes = 6):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _prf_divide(self,numerator, denominator, ):
        """Performs division and handles divide-by-zero.
        On zero-division, sets the corresponding result elements equal to
        0 or 1 (according to ``zero_division``).
        """
        mask = denominator == 0.0
        denominator = denominator.copy()
        denominator[mask] = 1  # avoid infs/nans
        result = numerator / denominator

        return result

    def fus_mat_convert(self,fus_mat, n_class):
        mcm = np.zeros((n_class, 2, 2))
        for n in range(n_class):
            # tp
            mcm[n, 1, 1] = fus_mat[n, n]
            # fn
            mcm[n, 1, 0] = np.sum(fus_mat[n, :]) - fus_mat[n, n]
            # fp
            mcm[n, 0, 1] = np.sum(fus_mat[:, n]) - fus_mat[n, n]

        return mcm

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)

        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            #self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
            self.confusion_matrix += self._fast_hist(lt[lt!=-1], lp[lt!=-1], self.n_classes)


    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        mean_acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum() # fraction of the pixels that come from each class
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        MCM = self.fus_mat_convert(self.confusion_matrix,self.n_classes)

        tp_sum = MCM[:, 1, 1]
        tn_sum = MCM[:, 0, 0]
        fn_sum = MCM[:, 1, 0]
        fp_sum = MCM[:, 0, 1]

        class_weights = [1, 1, 1, 1, 1, 1]

        # precision : tp / (tp + fp)
        precision = self._prf_divide(tp_sum,(tp_sum + fp_sum))
        # recall : tp / (tp + fn)
        recall = self._prf_divide(tp_sum,(tp_sum + fn_sum))
        # f1 : 2 * (recall * precision) / (recall + precision)
        f1_score = self._prf_divide(2 * precision * recall,precision + recall)
        f1_score_weighted = np.dot(class_weights, f1_score) / np.sum(class_weights)

        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Pixel Acc': acc,
                'Class Accuracy': acc_cls,
                'Mean Class Acc': mean_acc_cls,
                'Freq Weighted IoU': fwavacc,
                # 'IoU':iu,
                'Mean IoU': mean_iu,
                'F1 Score Weighted': f1_score_weighted,
                }, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def negative_index_sampler(samp_num, negative_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(negative_num_list[:j]),
                                                high=sum(negative_num_list[:j+1]),
                                                size=int(samp_num[i, j])).tolist()
    return negative_index

def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)


def generate_cutout_mask(img_size, ratio=2):
    cutout_area = img_size[0] * img_size[1] / ratio

    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.float()

def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)  # all unique labels
    labels_select = labels[torch.randperm(len(labels))][:len(labels) // 2]  # randomly select half of labels
    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()

def generate_aug_data(data, target, logits, mode='cutmix'):
    batch_size, _, im_h, im_w = data.shape
    device = data.device
    new_data = []
    new_target = []
    new_logits = []
    for i in range(batch_size):
        if mode == 'cutmix':
            mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
        if mode == 'classmix':
            mix_mask = generate_class_mask(target[i]).to(device)

        new_data.append((data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_target.append((target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

    new_data, new_target, new_logits = torch.cat(new_data), torch.cat(new_target), torch.cat(new_logits)
    return new_data, new_target.long(), new_logits
