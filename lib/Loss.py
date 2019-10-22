import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def get_loss_criterion(loss_config,type):
    if type == 'NCE':
        loss_criterion = NCECriterion(loss_config['NCE']['nLem'])
    elif type == 'CrossEntropyLoss':
        loss_criterion = nn.CrossEntropyLoss(ignore_index=-100)#, weight = torch.FloatTensor([1.0/2000,1.0/1000,1.0/1000]))
    elif type == 'combined':
        loss_criterion = CombinedLoss()
    elif type == 'mse':
        loss_criterion = nn.MSELoss()
    elif type == 'LMCL':
        loss_criterion = LMCL_loss(num_classes=loss_config['model']['n_label'], feat_dim = loss_config['model']['nhid'])
    elif type == 'comp':
        loss_criterion = complementary_CE(ignore_index=-100)
    else:
        raise NotImplementedError
    return loss_criterion


class complementary_CE(nn.Module):
    def __init__(self,ignore_index=-100,weight=None):
        super(complementary_CE, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self,pred_score,target_score):
        #log_softmax = ((-F.softmax(pred_score,dim=1)).exp_()+1).log_() 
        #return F.nll_loss(log_softmax,target_score,weight=self.weight,ignore_index=self.ignore_index) - F.softmax(pred_score, dim=1) * F.log_softmax(pred_score, dim=1).sum()
        return - F.softmax(pred_score, dim=1) * F.log_softmax(pred_score, dim=1).sum()

class LogitBinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(LogitBinaryCrossEntropy, self).__init__()

    def forward(self, pred_score, target_score, weights=None):
        loss = F.binary_cross_entropy_with_logits(pred_score,
                                                  target_score,
                                                  size_average=True)
        loss = loss * target_score.size(1)
        return loss


def kl_div(log_x, y):
    y_is_0 = torch.eq(y.data, 0)
    y.data.masked_fill_(y_is_0, 1)
    log_y = torch.log(y)
    y.data.masked_fill_(y_is_0, 0)
    res = y * (log_y - log_x)

    return torch.sum(res, dim=1, keepdim=True)


class weighted_softmax_loss(nn.Module):
    def __init__(self):
        super(weighted_softmax_loss, self).__init__()

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = loss * tar_sum
        loss = torch.sum(loss) / loss.size(0)
        return loss


class SoftmaxKlDivLoss(nn.Module):
    def __init__(self):
        super(SoftmaxKlDivLoss, self).__init__()

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = torch.sum(loss) / loss.size(0)
        return loss


class wrong_loss(nn.Module):
    def __init__(self):
        super(wrong_loss, self).__init__()

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = F.kl_div(res, tar, size_average=True)
        loss *= target_score.size(1)
        return loss

