import argparse
import os
import sys
import shutil
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from pathlib import Path
from torch import optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR,MultiStepLR
from models.build_model import build_model
from torchvision import utils
import os
import datasets
import models
import math
import yaml
import numpy as np

from datasets.adni_3d import ADNI_3D

from lib.Loss import get_loss_criterion
from lib.utils import DataParallel_withLoss, get_auc_data, accuracy, AverageMeter, balanced_accuracy_score, clip_gradients, visualize_visdom
#import warnings


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    type=str,
                    default="config",
                    required=False,
                    help="config")
parser.add_argument("--expansion",
                    type=int,
                    default=0,
                    required=False,
                    help="expansions to decide the width of the model")

parser.add_argument("--percentage_usage",type=float,
                    default=1.0,
                    required=False,
                    help="percentage of data to use for training")
arguments = parser.parse_args()

best_prec1 = 0
best_loss = 1000
best_micro_auc = 0
best_macro_auc = 0
with open(os.path.join('./'+arguments.config+'.yaml'), 'r') as f:
    cfg = yaml.load(f)

if arguments.expansion > 0:
    cfg['model']['expansion'] = arguments.expansion
cfg['data']['percentage_usage'] = arguments.percentage_usage
cfg['file_name'] = cfg['file_name']+'_train_perc_'+str(arguments.percentage_usage*100)+'_expansion_'+str(arguments.expansion)+'.pth.tar' 
cfg['exp_name'] = cfg['exp_name']+'_train_perc_'+str(arguments.percentage_usage*100)+'_expansion_'+str(arguments.expansion)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def main():
    global cfg, best_prec1, best_loss, device, best_micro_auc, best_macro_auc

    # Set seeds
    seed = 168
    torch.manual_seed(seed)
    if torch.cuda.device_count()>0:
        torch.cuda.manual_seed(seed)

    main_model = build_model(cfg)
    main_model = main_model.to(device)


    criterion = get_loss_criterion(cfg,type='CrossEntropyLoss').to(device)

    #Loss parallel
    model = DataParallel_withLoss(main_model,criterion)

    if hasattr(model, 'module'):
        print('has module!')
        model = model.module

    # Optimization set up
    params = [{'params': filter(lambda p: p.requires_grad, model.parameters())}]


    main_optim = getattr(optim, cfg['optimizer']['method'])(
        params, **cfg['optimizer']['par'])

    scheduler = MultiStepLR(main_optim, milestones=[20,50], gamma=0.1)#get_optim_scheduler(main_optim)

    # Plot with visdom
    if cfg['visdom']['server'] is not None:
        viz_plot = visualize_visdom(cfg)

    #Load data
    dir_to_scans = cfg['data']['dir_to_scans']
    dir_to_tsv = cfg['data']['dir_to_tsv']
    train_dataset = ADNI_3D(dir_to_scans, dir_to_tsv, mode = 'Train', 
        n_label = cfg['model']['n_label'], percentage_usage=cfg['data']['percentage_usage'])
    val_dataset = ADNI_3D(dir_to_scans, dir_to_tsv, mode = 'Val', n_label = cfg['model']['n_label'])


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True,
        num_workers=cfg['data']['workers'], pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg['data']['val_batch_size'], shuffle=False,
        num_workers=cfg['data']['workers'], pin_memory=True)

    ndata = len(train_dataset.subject_id)
    print('In total ', str(ndata), ' patients in training set')

    # Training !!!
    for epoch in range(cfg['training_parameters']['start_epoch'], cfg['training_parameters']['epochs']):
        
        # train for one epoch
        train_loss, train_acc = train(cfg,train_loader, model, scheduler, criterion, main_optim, epoch)

        

        # evaluate on validation set
        val_loss, val_acc, confusion_matrix, auc_outs = validate(cfg,val_loader,model,criterion,epoch)
        #prec1 /= len(val_dataset)
        #scheduler.step()
        print('Epoch [{0}]: Validation Accuracy {prec1:.3f}\t'.format(
               epoch, prec1=val_acc))

        if cfg['visdom']['server'] is not None:
            viz_plot.plot(epoch, train_loss, val_loss, train_acc, val_acc, confusion_matrix, auc_outs)
            


        # Save model
        is_best = (val_acc > best_prec1) 
        lowest_loss = (val_loss < best_loss)
        is_best_micro_auc = (auc_outs[2][len(auc_outs[2])-2]>= best_micro_auc)
        is_best_macro_auc = (auc_outs[2][len(auc_outs[2])-1]> best_macro_auc)

        best_prec1 = max(val_acc, best_prec1)
        best_loss = min(val_loss,best_loss)

        best_micro_auc = max(auc_outs[2][len(auc_outs[2])-2], best_micro_auc)
        best_macro_auc = max(auc_outs[2][len(auc_outs[2])-1], best_macro_auc)

        is_best_auc = (auc_outs[2][len(auc_outs[2])-2]>0.8) & (auc_outs[2][len(auc_outs[2])-1]>0.8)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_loss' : best_loss,
            'optimizer' : main_optim.state_dict(),
        }, is_best, lowest_loss, is_best_micro_auc, is_best_macro_auc, is_best_auc, filename=cfg['file_name'])


def train(cfg, train_loader, main_model, scheduler, 
    criterion,  main_optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_losses = AverageMeter()

    # switch to train mode
    main_model.train()
    
    end = time.time()

    logit_all = []
    target_all = []
    for i, (input, target, index, mmse, segment,age) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)


        index = index.to(device)
        if cfg['training_parameters']['use_age']:
            age = age.to(device)
        else: 
            age = None

        # compute output
        input = input.to(device)
        target = target.to(device)

        main_loss, logit = main_model([input, age], target)
        main_loss = main_loss.mean()

        logit_all.append(logit.data.cpu())
        target_all.append(target.data.cpu())
        acc,_ = accuracy(logit.data.cpu(),target.data.cpu())
        
        main_optimizer.zero_grad()
        main_loss.backward()
        clip_gradients(main_model, i, cfg['training_parameters']['max_grad_l2_norm'])
        main_optimizer.step()

        # measure accuracy and record loss
        main_losses.update(main_loss.cpu().item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg['training_parameters']['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy:.3f}\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=main_losses, accuracy=acc[0].item()))

    logit_all = torch.cat(logit_all).numpy()
    target_all = torch.cat(target_all).numpy()
    acc_all = balanced_accuracy_score(target_all, np.argmax(logit_all,1))


    return main_losses.avg, acc_all*100

def validate(cfg,val_loader,main_model,criterion,epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_losses = AverageMeter()

    end = time.time()
    correct_all = 0.0

    # switch to validation mode
    main_model.eval()

    confusion_matrix = torch.zeros(cfg['model']['n_label'], cfg['model']['n_label'])
    logit_all = []
    target_all = []

    for i, (input, target, patient_idx, mmse, segment,age) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.to(device)
        target = target.to(device)
        if cfg['training_parameters']['use_age']:
            age = age.to(device)
        else: 
            age = None
        # compute output

        main_loss, logit = main_model([input, age], target)
        main_loss = main_loss.mean()

        logit_all.append(torch.tensor(logit.data.cpu()))
        target_all.append(torch.tensor(target.data.cpu()))
        acc,_ = accuracy(logit.data.cpu(),target.data.cpu())
        _, preds = torch.max(logit.cpu(), 1)
        for t, p in zip(target.cpu().view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        acc,correct = accuracy(logit.cpu(),target.cpu())
        correct_all += correct[0].item()

        # measure accuracy and record loss
        main_losses.update(main_loss.cpu().item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

       
        if i % cfg['training_parameters']['print_freq'] == 0:
            print('Validation [{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy:.3f}\t'.format(
                   epoch, i, len(val_loader), batch_time=batch_time,
                   data_time=data_time, loss=main_losses, accuracy=acc[0].item()))

    #plot AUC curves
    logit_all = torch.cat(logit_all).numpy()
    target_all = torch.cat(target_all).numpy()
    acc_all = balanced_accuracy_score(target_all, np.argmax(logit_all,1))
    plotting_fpr, plotting_tpr, roc_auc = get_auc_data(logit_all, target_all,cfg['model']['n_label'])
    

    return main_losses.avg, acc_all*100, confusion_matrix, [plotting_fpr, plotting_tpr, roc_auc]


def save_checkpoint(state, is_best, lowest_loss, is_best_micro_auc, is_best_macro_auc, is_best_auc, filename='checkpoint.pth.tar'):
    saving_dir = Path(filename).parent
    print(saving_dir)
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth.tar','')+'_model_best.pth.tar')
    if lowest_loss:
        shutil.copyfile(filename, filename.replace('.pth.tar','')+'_model_low_loss.pth.tar')
    if is_best_micro_auc:
        shutil.copyfile(filename, filename.replace('.pth.tar','')+'_model_best_micro.pth.tar')
    if is_best_macro_auc:
        shutil.copyfile(filename, filename.replace('.pth.tar','')+'_model_best_macro.pth.tar')
    if is_best_auc:
        shutil.copyfile(filename, filename.replace('.pth.tar','')+'_model_best_auc.pth.tar')




if __name__ == '__main__':
    main()