# Copyright (c) Microsoft. All rights reserved.
from enum import Enum

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import classification_report
from data_utils.squad_eval import evaluate_func
import numpy as np
import sys

def compute_acc(predicts, labels,scores):
   #Codice per la log loss
    #np.set_printoptions(threshold=sys.maxsize)
    #data = np.asarray(labels)
   
    #nb_classes = 5  
  
    #data = np.array(data).reshape(-1)
    #labels=np.eye(nb_classes)[data]

    #labels=labels.flatten()
   
   
    return 100.0 * accuracy_score(labels, predicts)

def compute_f1_ddi(predicts, labels,scores):
    return 100.0 * f1_score(labels, predicts,average='micro', labels=[0,1,2,3])

def compute_f1_binary(predicts, labels,scores):
    return 100.0 * f1_score(labels, predicts,average='binary')

def compute_precision_binary(predicts, labels,scores):
    return 100.0 * precision_score(labels, predicts,average='binary')

def compute_recall_binary(predicts, labels,scores):
    return 100.0 * recall_score(labels, predicts,average='binary')
    
def compute_f1_chem(predicts, labels,scores):
    return 100.0 * f1_score(labels, predicts,average='micro', labels=[0,1,2,3,4])

def compute_f1_i2b2(predicts, labels,scores):
    return 100.0 * f1_score(labels, predicts,average='micro', labels=[0,1,2,3,4,5,6,7])
    
def compute_precision_ddi(predicts, labels,scores):
    return 100.0 * precision_score(labels, predicts,average='micro', labels=[0,1,2,3])
    
def compute_precision_chem(predicts, labels,scores):
    return 100.0 * precision_score(labels, predicts,average='micro', labels=[0,1,2,3,4])

def compute_precision_i2b2(predicts, labels,scores):
    return 100.0 * precision_score(labels, predicts,average='micro', labels=[0,1,2,3,4,5,6,7])

def compute_recall_ddi(predicts, labels,scores):
    return 100.0 * recall_score(labels, predicts,average='micro', labels=[0,1,2,3])
    
def compute_recall_chem(predicts, labels,scores):
    return 100.0 * recall_score(labels, predicts,average='micro', labels=[0,1,2,3,4])

def compute_recall_i2b2(predicts, labels,scores):
    return 100.0 * recall_score(labels, predicts,average='micro', labels=[0,1,2,3,4,5,6,7])

def compute_f1mac(predicts, labels,scores):
    return 100.0 * f1_score(labels, predicts, average='macro')

def compute_f1mic(predicts, labels,scores):
    return 100.0 * f1_score(labels, predicts, average='micro')

def compute_mcc(predicts, labels,scores):
    return 100.0 * matthews_corrcoef(labels, predicts)

def compute_pearson(predicts, labels,scores):
    pcof = pearsonr(labels, predicts)[0]
    return 100.0 * pcof

def compute_spearman(predicts, labels,scores):
    scof = spearmanr(labels, predicts)[0]
    return 100.0 * scof

def compute_auc(predicts, labels,scores):
    auc = roc_auc_score(labels, predicts)
    return 100.0 * auc

def compute_cmat(predicts, labels,scores):
    #return str(confusion_matrix(labels, predicts))
    return confusion_matrix(labels, predicts)

def compute_seqacc(predicts, labels, label_mapper):
    y_true, y_pred = [], []
    def trim(predict, label):
        temp_1 =  []
        temp_2 = []
        for j, m in enumerate(predict):
            if j == 0:
                continue
            if label_mapper[label[j]] != 'X':
                temp_1.append(label_mapper[label[j]])
                temp_2.append(label_mapper[m])
        temp_1.pop()
        temp_2.pop()
        y_true.append(temp_1)
        y_pred.append(temp_2)
    for predict, label in zip(predicts, labels):
        trim(predict, label)
    report = classification_report(y_true, y_pred,digits=4)
    return report

def compute_emf1(predicts, labels):
    return evaluate_func(labels, predicts)


class Metric(Enum):
    ACC = 0
    #F1 = 1
    MCC = 2
    Pearson = 3
    Spearman = 4
    AUC = 5
    SeqEval = 7
    EmF1 = 8
    F1MAC = 9
    F1MIC = 10
    CMAT = 11 
    F1_chem=12
    F1_i2b2=13
    F1_ddi=14
    Precision_chem=15
    Precision_ddi=16
    Precision_i2b2=17
    Recall_chem=18
    Recall_ddi=19
    Recall_i2b2=20
    F1_binary=21
    binary_precision=22
    binary_recall=23




METRIC_FUNC = {
    Metric.ACC: compute_acc,
    #Metric.F1: compute_f1,
    Metric.MCC: compute_mcc,
    Metric.Pearson: compute_pearson,
    Metric.Spearman: compute_spearman,
    Metric.AUC: compute_auc,
    Metric.SeqEval: compute_seqacc,
    Metric.EmF1: compute_emf1,
    Metric.F1MAC: compute_f1mac,
    Metric.F1MIC: compute_f1mic,
    Metric.CMAT: compute_cmat,
    Metric.F1_chem :compute_f1_chem,
    Metric.F1_i2b2 :compute_f1_i2b2,
    Metric.F1_ddi :compute_f1_ddi,
    Metric.Precision_chem :compute_precision_chem,
    Metric.Precision_i2b2 :compute_precision_i2b2,
    Metric.Precision_ddi :compute_precision_ddi,
    Metric.Recall_chem :compute_recall_chem,
    Metric.Recall_i2b2 :compute_recall_i2b2,
    Metric.Recall_ddi :compute_recall_ddi,
    Metric.F1_binary:compute_f1_binary,
    Metric.binary_precision:compute_precision_binary,
    Metric.binary_recall:compute_recall_binary
    

    
}


def calc_metrics(metric_meta, golds, predictions, scores, label_mapper=None):
    """Label Mapper is used for NER/POS etc. 
    TODO: a better refactor, by xiaodl
    """
    metrics = {}
    for mm in metric_meta:
        metric_name = mm.name
        metric_func = METRIC_FUNC[mm]
        if mm in (Metric.ACC, Metric.MCC, Metric.F1MAC, Metric.F1MIC, Metric.CMAT, Metric.F1_chem,Metric.F1_ddi,Metric.F1_i2b2,Metric.Precision_chem,Metric.Precision_i2b2,Metric.Precision_ddi,Metric.Recall_chem,Metric.Recall_i2b2,Metric.Recall_ddi,Metric.F1_binary,Metric.binary_precision,Metric.binary_recall):
            metric = metric_func(predictions, golds,scores)
        elif mm == Metric.SeqEval:
            metric = metric_func(predictions, golds, label_mapper)
        elif mm == Metric.EmF1:
            metric = metric_func(predictions, golds)
        else:
            if mm == Metric.AUC:
                assert len(scores) == 2 * len(golds), "AUC is only valid for binary classification problem"
                scores = scores[1::2]
            metric = metric_func(scores, golds)
        metrics[metric_name] = metric
    return metrics
