from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import os
import torch
from os.path import join
import numpy as np
from collections import OrderedDict
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
params = {'legend.fontsize': 13,
         'axes.labelsize': 15,
         'axes.titlesize':15,
         'xtick.labelsize':15,
         'ytick.labelsize':15}
pylab.rcParams.update(params)

class Evaluate:
    def __init__(self, save_path=None):
        self.target = None
        self.output = None
        self.save_path = save_path
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        self.threshold_confusion = 0.1

    def add_batch(self, batch_tar, batch_out):
        batch_tar = batch_tar.flatten()
        batch_out = batch_out.flatten()

        self.target = batch_tar if self.target is None else np.concatenate((self.target,batch_tar))
        self.output = batch_out if self.output is None else np.concatenate((self.output,batch_out))

    def auc_roc(self, plot=False):
        valid_indices = ~np.isnan(self.target) & ~np.isnan(self.output)
        filtered_target = self.target[valid_indices]
        filtered_output = self.output[valid_indices]

        if len(filtered_target) == 0:
            return 0.0

        AUC_ROC = roc_auc_score(filtered_target, filtered_output)
        if plot and self.save_path is not None:
            fpr, tpr, _ = roc_curve(filtered_target, filtered_output)
            plt.figure()
            plt.plot(fpr, tpr, '-', label='AUC = %0.4f' % AUC_ROC)
            plt.title('ROC curve')
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend()
            plt.savefig(join(self.save_path, "ROC.png"))
            plt.close()
        return AUC_ROC

    def auc_pr(self, plot=False):
        valid_indices = ~np.isnan(self.target) & ~np.isnan(self.output)
        filtered_target = self.target[valid_indices]
        filtered_output = self.output[valid_indices]

        if len(filtered_target) == 0:
            return 0.0

        precision, recall, _ = precision_recall_curve(filtered_target, filtered_output)
        AUC_pr = np.trapz(precision, recall)
        if plot and self.save_path is not None:
            plt.figure()
            plt.plot(recall, precision, '-', label='AUC = %0.4f' % AUC_pr)
            plt.title('Precision-Recall Curve')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            plt.savefig(join(self.save_path, "Precision_recall.png"))
            plt.close()
        return AUC_pr


    def confusion_matrix(self):
        valid_indices = ~np.isnan(self.target) & ~np.isnan(self.output)
        y_true = self.target[valid_indices]
        y_pred = (self.output[valid_indices] >= self.threshold_confusion).astype(int)

        if len(y_true) == 0:
            return np.zeros((2, 2)), 0.0, 0.0, 0.0, 0.0

        batch_size = 1024
        num_classes = 2

        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(0, len(y_true), batch_size):
            batch_y_true = y_true[i:i + batch_size]
            batch_y_pred = y_pred[i:i + batch_size]

            batch_confusion = sk_confusion_matrix(batch_y_true, batch_y_pred, labels=[0, 1])
            conf_matrix += batch_confusion

        tn, fp, fn, tp = conf_matrix.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        return conf_matrix, accuracy, specificity, sensitivity, precision

    def f1_score(self):
        valid_indices = ~np.isnan(self.target) & ~np.isnan(self.output)
        y_true = self.target[valid_indices]
        y_pred = (self.output[valid_indices] >= self.threshold_confusion).astype(int)

        if len(y_true) == 0:
            return 0.0

        return f1_score(y_true, y_pred)

    def save_all_result(self, plot_curve=True, save_name=None):
        AUC_ROC = self.auc_roc(plot=plot_curve)
        AUC_pr  = self.auc_pr(plot=plot_curve)
        F1_score = self.f1_score()
        confusion,accuracy, specificity, sensitivity, precision = self.confusion_matrix()
        if save_name is not None:
            file_perf = open(join(self.save_path, save_name), 'w')
            file_perf.write("AUC ROC curve: "+str(AUC_ROC)
                            + "\nAUC PR curve: " +str(AUC_pr)
                            + "\nF1 score: " +str(F1_score)
                            +"\nAccuracy: " +str(accuracy)
                            +"\nSensitivity(SE): " +str(sensitivity)
                            +"\nSpecificity(SP): " +str(specificity)
                            +"\nPrecision: " +str(precision)
                            + "\n\nConfusion matrix:"
                            + str(confusion)
                            )
            file_perf.close()
        return OrderedDict([("AUC_ROC",AUC_ROC),("AUC_PR",AUC_pr),
                            ("f1-score",F1_score),("Acc",accuracy),
                            ("SE",sensitivity),("SP",specificity),
                            ("precision",precision)
                            ])