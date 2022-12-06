import numpy as np

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)  # 分類task外設為False, bincount就不會掃到, MRI分類無task分別所以全部為True
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],  # 0+0=0, 0+1=1, 2+0=2, 2+1=3
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist  # hist[[0_0, 0_1], [1_0, 1_1]]

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy  # 分類正確比例
            - mean accuracy     # 每種標籤分類正確比例的平均(不參考權重)
            - mean dice         # 本次競賽評分標準
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix  # hist = [[NP, NF], [TF, TP]]
        acc = np.diag(hist).sum() / hist.sum()
        acc_mean = (2 * hist[1, 1]) / (2 * hist[1, 1] + hist[1,0] + hist[0,1])
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        # acc_cls = np.diag(hist) / hist.sum(axis=1)
        # acc_cls = np.nanmean(acc_cls)
        # freq = hist.sum(axis=1) / hist.sum()
        # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "OverallAcc": acc,
                "MeanDice": acc_mean,
                "mIoU": mean_iu,
                # "MeanAcc": acc_cls,
                # "FreqWAcc": fwavacc,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.__val = 0
        self.__avg = 0
        self.__sum = 0
        self.__count = 0

    def update(self, val, n=1):
        self.__val = val
        self.__sum += val * n
        self.__count += n
        self.__avg = self.__sum / self.__count

    @property
    def avg(self):
        return self.__avg
