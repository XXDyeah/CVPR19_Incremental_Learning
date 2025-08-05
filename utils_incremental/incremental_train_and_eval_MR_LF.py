#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils_pytorch import *
from .tiaw import TIAWWeighting

cur_features = []
ref_features = []
old_scores = []
new_scores = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs

def incremental_train_and_eval_MR_LF(epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, \
            iteration, start_iteration, \
            lamda, \
            dist, K, lw_mr, \
            fix_bn=False, weight_per_class=None, device=None, \
            cba_lambda: float = 1.0, tiaw_module: TIAWWeighting = None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model = tg_model.to(device)
    if iteration > start_iteration and ref_model is not None:
        ref_model = ref_model.to(device)
    #trainset.train_data = X_train.astype('uint8')
    #trainset.train_labels = Y_train
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
    #    shuffle=True, num_workers=2)
    #testset.test_data = X_valid.astype('uint8')
    #testset.test_labels = Y_valid
    #testloader = torch.utils.data.DataLoader(testset, batch_size=100,
    #    shuffle=False, num_workers=2)
    #print('Max and Min of train labels: {}, {}'.format(min(Y_train), max(Y_train)))
    #print('Max and Min of valid labels: {}, {}'.format(min(Y_valid), max(Y_valid)))

    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
        handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
        handle_cur_features = tg_model.fc.register_forward_hook(get_cur_features)
        handle_old_scores_bs = tg_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
        handle_new_scores_bs = tg_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)
    for epoch in range(epochs):
        #train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    #m.weight.requires_grad = False
                    #m.bias.requires_grad = False
        train_loss = 0
        train_loss1 = 0
        train_loss3 = 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr())
        for batch_idx, (inputs, targets, indices, flags) in enumerate(trainloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            indices = indices.to(device)
            flags = flags.view(-1).to(device)
            tg_optimizer.zero_grad()

            outputs = tg_model(inputs)
            probs = F.softmax(outputs.detach(), dim=1)
            if tiaw_module is not None:
                weights = tiaw_module.update_and_get_weights(indices, probs)
            else:
                weights = torch.ones(inputs.size(0), device=device)

            real_mask = flags == 0
            cf_mask = flags == 1
            weights_cat = torch.cat([weights[real_mask], weights[cf_mask]])

            real_logits = outputs[real_mask]
            real_targets = targets[real_mask].argmax(dim=1).long()
            loss_vec = []
            if real_mask.any():
                loss_real = nn.CrossEntropyLoss(weight_per_class, reduction='none')(real_logits, real_targets)
                loss_vec.append(loss_real)
            else:
                loss_real = torch.tensor([], device=device)

            if cf_mask.any():
                cf_logits = outputs[cf_mask]
                cf_soft = targets[cf_mask]
                cf_soft = cf_soft[:, :cf_logits.size(1)]
                adv_loss = (-cf_soft * F.log_softmax(cf_logits, dim=1)).sum(dim=1)
                loss_vec.append(cba_lambda * adv_loss)

            loss_vec = torch.cat(loss_vec)
            loss_cls = (loss_vec * weights_cat).mean()

            if iteration == start_iteration:
                loss = loss_cls
            else:
                ref_outputs = ref_model(inputs[real_mask])
                loss1 = nn.CosineEmbeddingLoss()(cur_features[real_mask], ref_features.detach(), \
                    torch.ones(real_mask.sum(), device=device)) * lamda
                #################################################
                outputs_bs = torch.cat((old_scores, new_scores), dim=1)[real_mask]
                assert(outputs_bs.size(0) == real_mask.sum())
                gt_index = torch.zeros(outputs_bs.size(), device=device)
                real_targets = targets[real_mask].argmax(dim=1).long()
                gt_index = gt_index.scatter(1, real_targets.view(-1,1), 1).ge(0.5)
                gt_scores = outputs_bs.masked_select(gt_index)
                max_novel_scores = outputs_bs[:, num_old_classes:].topk(K, dim=1)[0]
                hard_index = real_targets.lt(num_old_classes)
                hard_num = torch.nonzero(hard_index).size(0)
                if  hard_num > 0:
                    gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, K)
                    max_novel_scores = max_novel_scores[hard_index]
                    assert(gt_scores.size() == max_novel_scores.size())
                    assert(gt_scores.size(0) == hard_num)
                    loss3 = nn.MarginRankingLoss(margin=dist)(gt_scores.view(-1, 1), \
                        max_novel_scores.view(-1, 1), torch.ones(hard_num*K).to(device)) * lw_mr
                else:
                    loss3 = torch.zeros(1).to(device)
                #################################################
                loss = loss1 + loss_cls + loss3
            loss.backward()
            tg_optimizer.step()

            train_loss += loss.item()
            if iteration > start_iteration:
                train_loss1 += loss1.item()
                train_loss3 += loss3.item()
            _, predicted = real_logits.max(1)
            total += real_targets.size(0)
            correct += predicted.eq(real_targets).sum().item()

            #if iteration == 0:
            #    msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
            #    (train_loss/(batch_idx+1), 100.*correct/total, correct, total)
            #else:
            #    msg = 'Loss1: %.3f Loss3: %.3f Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
            #    (loss1.item(), loss3.item(), train_loss/(batch_idx+1), 100.*correct/total, correct, total)
            #progress_bar(batch_idx, len(trainloader), msg)
        if iteration == start_iteration:
            print('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format(\
                len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))
        else:
            print('Train set: {}, Train Loss1: {:.4f}, Train Loss3: {:.4f},\
                Train Loss: {:.4f} Acc: {:.4f}'.format(len(trainloader), \
                train_loss1/(batch_idx+1), train_loss3/(batch_idx+1),
                train_loss/(batch_idx+1), 100.*correct/total))

        #eval
        tg_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
            len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    if iteration > start_iteration:
        print("Removing register_forward_hook")
        handle_ref_features.remove()
        handle_cur_features.remove()
        handle_old_scores_bs.remove()
        handle_new_scores_bs.remove()
    return tg_model
