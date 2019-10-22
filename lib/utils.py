import torch.nn as nn
import torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
import matplotlib.pyplot as plt
class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FullModel(nn.Module):
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss =  loss
    def forward(self, inputs, targets):
        outputs = self.model(inputs[0],inputs[1])
        loss = self.loss(outputs, targets)
        return torch.unsqueeze(loss,0),outputs

def DataParallel_withLoss(model,loss, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>1:
        print("lets use multiple gpu!",torch.cuda.device_count())
    model=FullModel(model, loss)
    if 'device_ids' in kwargs.keys():
        device_ids=kwargs['device_ids']
    else:
        device_ids=None
    if 'output_device' in kwargs.keys():
        output_device=kwargs['output_device']
    else:
        output_device=None
    if 'cuda' in kwargs.keys():
        cudaID=kwargs['cuda'] 
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).to(device)
    else:
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).to(device)
    return model    

def clip_gradients(myModel, i_iter, max_grad_l2_norm):
    #max_grad_l2_norm = cfg['training_parameters']['max_grad_l2_norm']
    if max_grad_l2_norm is not None:
        norm = nn.utils.clip_grad_norm_(myModel.parameters(), max_grad_l2_norm)

def balanced_accuracy_score(y_true, y_pred, sample_weight=None,
                            adjusted=False):
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score

def get_auc_data(logit_all, target_all,n_label):
    y = label_binarize(target_all, classes=list(range(n_label)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for k in range(n_label):
        fpr[k], tpr[k], _ = roc_curve(y[:, k], logit_all[:, k])
        roc_auc[k] = auc(fpr[k], tpr[k])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), logit_all.ravel())
    roc_auc[k+1] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[k] for k in range(n_label)]))

    mean_tpr = np.zeros_like(all_fpr)
    for k in range(n_label):
        mean_tpr += interp(all_fpr, fpr[k], tpr[k])

    # Finally average it and compute AUC
    mean_tpr /= n_label

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc[k+2] = auc(fpr["macro"], tpr["macro"])
    plotting_fpr = []
    plotting_tpr = []
    for k in range(n_label):
        plotting_fpr.append(fpr[k])
        plotting_tpr.append(tpr[k])
    plotting_fpr += [fpr["micro"], fpr["macro"]]
    plotting_tpr += [tpr["micro"], tpr["macro"]]
    return plotting_fpr, plotting_tpr, roc_auc



class visualize_visdom:
    def __init__(self, cfg):
        import visdom
        self.cfg = cfg
        exp_name = cfg['exp_name']
        self.viz = visdom.Visdom(port=cfg['visdom']['port'], server='http://'+cfg['visdom']['server'])
        self.viz.env = exp_name
        self.loss_plot = self.viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 2)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Losses',
                title='Train & Val Losses',
                legend=['Train-Loss', 'Val-Loss']
            )
        )



        self.eval_plot = self.viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 2)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Accuracy',
                title='Train & Val Accuracies',
                legend=['trainTop1','valTop1']
            )
        )

        self.conf_mat_plot = self.viz.heatmap(
            X=np.outer(np.arange(1, 4), np.arange(1, 4)),
            opts=dict(
                columnnames=['CN','MCI','AD'],
                rownames=['CN','MCI','AD'],
                title='Confusion Matrix',
                colormap='Electric',
                )
            )
        import matplotlib.pyplot as plt
        plt.plot([1, 23, 2, 4])
        plt.ylabel('some numbers')
   
        self.auc_plots = self.viz.matplot(plt)

    def plot(self, epoch, train_loss, val_loss, train_acc, val_acc, confusion_matrix, auc_outs):

        self.viz.line(
                    X=torch.ones((1, 2)).cpu() * epoch,
                    Y=torch.Tensor([train_loss, val_loss]).unsqueeze(0).cpu(),
                    win=self.loss_plot,
                    update='append'
                )
                
        self.viz.line(
            X=torch.ones((1, 2)).cpu() * epoch,
            Y=torch.Tensor([train_acc, val_acc]).unsqueeze(0).cpu(),
            win=self.eval_plot,
            update='append'
        )

        self.viz.heatmap(
        X=confusion_matrix,
        win=self.conf_mat_plot
        )

        # AUC curve:
        try:
            name = ['Class ' + str(i) + ' ' for i in range(self.cfg['model']['n_label'])]+['Micro ', 'Macro ']
            
            from itertools import cycle
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue','navy','deeppink'])
            plt.figure()
            for i, color in zip(range(len(auc_outs[0])), colors):
                plt.plot(auc_outs[0][i], auc_outs[1][i], color=color, lw=2, label=name[i] + 'ROC curve (area = %0.2f)' % auc_outs[2][i])
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('AUC curves')
            plt.legend(loc="lower right")
            self.viz.matplot(plt, win=self.auc_plots)
        except BaseException as err:
            print('Skipped matplotlib example')
            print('Error message: ', err)



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    correct_all = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_all.append(correct_k.clone())
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, correct_all