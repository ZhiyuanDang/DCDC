#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import os
import sys
sys.path.append('..')
import time
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
import scipy.io as scio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from lib import Config as cfg, networks, datasets, Session
from lib.utils import (lr_policy, optimizers, transforms, save_checkpoint, 
                            AverageMeter, TimeProgressMeter, traverse)
from lib.utils.loggers import STDLogger as logger, TFBLogger as SummaryWriter

from dc.utils import ConcatDataset, RepeatSampler, RandomSampler, get_reduced_transform
from dc.losses import DCLoss

def require_args():

    # args for training
    cfg.add_argument('--max-epochs', default=200, type=int,
                        help='maximal training epoch')
    cfg.add_argument('--display-freq', default=80, type=int,
                        help='log display frequency')
    cfg.add_argument('--batch-size', default=256, type=int,
                        help='size of mini-batch')
    cfg.add_argument('--num-workers', default=4, type=int,
                        help='number of workers used for loading data')
    cfg.add_argument('--data-nrepeat', default=1, type=int,
                        help='how many times each image in a ' +
                             'mini-batch should be repeated')
    cfg.add_argument('--dc-lamda', default=0.5, type=float,
                        help='temperature of contrastive learning')
    cfg.add_argument('--pica', default=False, type=bool,
                        help='pica or not')

def main():

    logger.info('Start to declare training variable')
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Session will be ran in device: [%s]' % cfg.device)
    start_epoch = 0
    best_acc = 0.

    if cfg.pica:
        logger.info('Work at PICA !!!!')

    logger.info('Start to prepare data')
    # get transformers
    # train_transform is for data perturbation
    #train_transform = transforms.get(train=True)
    # test_transform is for evaluation
    test_transform = transforms.get(train=False)
    # reduced_transform is for original training data
    #reduced_transform = get_reduced_transform(cfg.tfm_resize, cfg.tfm_size, 
    #                                            cfg.tfm_means, cfg.tfm_stds)
    # get datasets
    # each head should have its own trainset
    #train_splits = dict(cifar100=[['train', 'test']], cifar10=[['train', 'test']],
    #    stl10=[['train+unlabeled', 'test'], ['train', 'test']])
    test_splits = dict(cifar100=['train', 'test'], cifar10=['train', 'test'],
        stl10=['train', 'test'])
    # instance dataset for each head
    if cfg.dataset.startswith('stl') or cfg.dataset.startswith('cifar'):
        # otrainset: original trainset
        # otrainset = [ConcatDataset([datasets.get(split=split, transform=reduced_transform) 
        #                 for split in train_splits[cfg.dataset][hidx]]) 
        #                 for hidx in xrange(len(train_splits[cfg.dataset]))]
        # # ptrainset: perturbed trainset
        # ptrainset = [ConcatDataset([datasets.get(split=split, transform=train_transform) 
        #                 for split in train_splits[cfg.dataset][hidx]]) 
        #                 for hidx in xrange(len(train_splits[cfg.dataset]))]
        # testset
        testset = ConcatDataset([datasets.get(split=split, transform=test_transform) 
                        for split in test_splits[cfg.dataset]])
    else:
    #    otrainset = [ImageFolder(root = cfg.data_root, transform = reduced_transform) for hidx in xrange(2)]
    #    ptrainset = [ImageFolder(root = cfg.data_root, transform = train_transform) for hidx in xrange(2)]
        testset = ImageFolder(root = cfg.data_root, transform = test_transform)
        logger.debug('Dataset [%s] from directory [%s] is declared and %d samples '
        'are loaded' % (cfg.dataset, cfg.data_root, len(testset)))
    # declare data loaders for testset only
    test_loader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, 
                                num_workers=cfg.num_workers)

    logger.info('Start to build model')
    net = networks.get()
    criterion = DCLoss(cfg.dc_lamda)
    optimizer = optimizers.get(params=[val for _, val in net.trainable_parameters().iteritems()])
    lr_handler = lr_policy.get()

    # load session if checkpoint is provided
    if cfg.resume:
        assert os.path.exists(cfg.resume), "Resume file not found"
        ckpt = torch.load(cfg.resume)
        logger.info('Start to resume session for file: [%s]' % cfg.resume)
        if not cfg.pica:
            net.load_state_dict(ckpt['net'])
            best_acc = ckpt['acc']
            start_epoch = ckpt['epoch']
        else:
            net.load_state_dict(ckpt)
            best_acc = 0
            start_epoch = 0    

    # data parallel
    if cfg.device == 'cuda' and len(cfg.gpus.split(',')) > 1:
        logger.info('Data parallel will be used for acceleration purpose')
        device_ids = range(len(cfg.gpus.split(',')))
        if not (hasattr(net, 'data_parallel') and net.data_parallel(device_ids)):
            net = nn.DataParallel(net, device_ids=device_ids)
        cudnn.benchmark = True
    else:
        logger.info('Data parallel will not be used for acceleration')

    # move modules to target device
    net, criterion = net.to(cfg.device), criterion.to(cfg.device)

    # tensorboard wrtier
    writer = SummaryWriter(cfg.debug, log_dir=cfg.tfb_dir)
    # start training
    lr = cfg.base_lr
    epoch = start_epoch

    logger.info('Start to evaluate after %d epoch of training' % epoch)
    acc, nmi, ari = evaluate(net, test_loader)
    logger.info('Evaluation results at epoch %d are: '
        'ACC: %.3f, NMI: %.3f, ARI: %.3f' % (epoch, acc, nmi, ari))
    writer.add_scalar('Evaluate/ACC', acc, epoch)
    writer.add_scalar('Evaluate/NMI', nmi, epoch)
    writer.add_scalar('Evaluate/ARI', ari, epoch)

    logger.info('Done')


def evaluate(net, loader):
    """evaluates on provided data
    """

    net.eval()
    predicts = np.zeros(len(loader.dataset), dtype=np.int32)
    labels = np.zeros(len(loader.dataset), dtype=np.int32)
    features = np.zeros((len(loader.dataset),512), dtype=np.float32)
    pre_logits = np.zeros((len(loader.dataset),cfg.net_heads[-1]), dtype=np.float32)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            logger.progress('processing %d/%d batch' % (batch_idx, len(loader)))
            inputs = inputs.to(cfg.device, non_blocking=True)
            # assuming the last head is the main one
            # output dimension of the last head 
            # should be consistent with the ground-truth
            x, logits = net(inputs)
            logits = logits[-1]
            start = batch_idx * loader.batch_size
            end = start + loader.batch_size
            end = min(end, len(loader.dataset))
            labels[start:end] = targets.cpu().numpy()
            predicts[start:end] = logits.max(1)[1].cpu().numpy()
            features[start:end] = x.cpu().numpy()
            pre_logits[start:end] = logits.cpu().numpy()

    # compute accuracy
    num_classes = labels.max().item() + 1
    count_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for i in xrange(predicts.shape[0]):
        count_matrix[predicts[i], labels[i]] += 1
    reassignment = np.dstack(linear_sum_assignment(count_matrix.max() - count_matrix))[0]
    acc = count_matrix[reassignment[:,0], reassignment[:,1]].sum().astype(np.float32) / predicts.shape[0]
    if not cfg.pica:
        scio.savemat('features_dc.mat', {'features': features, 'predicts': predicts, 'labels':labels, 'pre_logits': pre_logits, 'reassignment':reassignment})
    else:
        scio.savemat('features_pica.mat', {'features': features, 'predicts': predicts, 'labels':labels, 'pre_logits': pre_logits, 'reassignment':reassignment})     
    return acc, NMI(labels, predicts), ARI(labels, predicts)


if __name__ == '__main__':
    Session(__name__).run()
