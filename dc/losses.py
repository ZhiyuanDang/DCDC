#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import Config as cfg

class DCLoss(nn.Module):

    def __init__(self, lamda=0.5):
        super(DCLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()
        self.t = lamda

    def forward(self, net, x, x_tf, head):
        """Partition Uncertainty Index
        
        Arguments:
            x {Tensor} -- [assignment probabilities of original inputs (N x K)]
            y {Tensor} -- [assignment probabilities of perturbed inputs (N x k)]
        
        Returns:
            [Tensor] -- [Loss value]
        """
        assert x.shape == x_tf.shape, ('Inputs are required to have same shape')

        # partition uncertainty index
        #pui = torch.mm(F.normalize(x.t(), p=2, dim=1), F.normalize(y, p=2, dim=0))
        #loss_ce = self.xentropy(pui, torch.arange(pui.size(0)).to(cfg.device))

        # balance regularisation
        p = x.sum(0).view(-1)
        p /= p.sum()
        loss_ne = math.log(p.size(0)) + (p * p.log()).sum()
        
        # t = 0.5
        
        x_norm = F.normalize(x)
        x_tf_norm = F.normalize(x_tf)
        
        logits = torch.mm(x_norm, x_tf_norm.t()) / self.t
        
        labels = torch.tensor(range(logits.shape[0])).cuda()
        
        
        #for c
        x_norm = F.normalize(x, dim=0)
        x_tf_norm = F.normalize(x_tf, dim=0)
        logits_c = torch.mm(x_norm.t(), x_tf_norm) / self.t
        
        labels_c = torch.tensor(range(logits_c.shape[0])).cuda()
        

        loss = torch.nn.CrossEntropyLoss()(logits, labels) + torch.nn.CrossEntropyLoss()(logits_c, labels_c) + loss_ne

        # loss1 = torch.nn.CrossEntropyLoss()(logits, labels)

        # loss2 = torch.nn.CrossEntropyLoss()(logits_c, labels_c)

        return loss