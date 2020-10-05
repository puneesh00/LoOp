'''
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mxnet as mx
import numpy as np

import time
import datetime
import csv
import os

from .embedding_aug_mx import get_embedding_aug, get_opt_emb_dis

def euclidean_dist(F, x, y, clip_min=1e-12, clip_max=1e12):
    m, n = x.shape[0], y.shape[0]

    squared_x = F.power(x, 2).sum(axis=1, keepdims=True).broadcast_to((m, n))
    squared_y = F.power(y, 2).sum(axis=1, keepdims=True).broadcast_to((n, m)).T

    dist = squared_x + squared_y
    dist = dist - 2 * F.dot(x, y.T)
    dist = dist.clip(a_min=clip_min, a_max=clip_max).sqrt()

    return dist


class HPHNTripletLoss(mx.gluon.loss.Loss):
    def __init__(self, margin=0.2, soft_margin=False, weight=None, batch_axis=0, num_instances=2, n_inner_pts=0, l2_norm=True):
        super(HPHNTripletLoss, self).__init__(weight, batch_axis)
        self.margin = margin
        self.soft_margin = soft_margin
        self.num_instance = num_instances
        self.n_inner_pts = n_inner_pts
        self.batch_size = None
        self.l2_norm = l2_norm

    def hybrid_forward(self, F, embeddings, labels):
        total_start_time = time.time()
        gen_time = 0
        self.batch_size = embeddings.shape[0]
        '''
        if self.n_inner_pts != 0:
            gen_start_time = time.time()
            embeddings, labels = get_embedding_aug(F, embeddings, labels, self.num_instance, self.n_inner_pts, self.l2_norm)
            #dis_ap, dis_an = get_opt_emb_dis(F, embeddings, labels, self.num_instance, self.l2_norm)
            gen_time = time.time() - gen_start_time
        '''
        #dist_mat = euclidean_dist(F, embeddings, embeddings)
        #dist_apo, dist_ano = self.hard_example_mining(F, dist_mat, labels)
        
        gen_start_time = time.time()
        dist_ap, dist_an = get_opt_emb_dis(F, embeddings, labels, self.num_instance, self.l2_norm)
        gen_time = time.time() - gen_start_time
        
        #dist_ap=0.5*dist_apo+0.5*dist_ape
        #dist_an=0.5*dist_ano+0.5*dist_ane
        
        if self.soft_margin:
            loss = F.log(1 + F.exp(dist_ap - dist_an))
        else:
            loss = F.relu(dist_ap - dist_an + self.margin)
        total_time = time.time() - total_start_time

        return loss

    def hard_example_mining(self, F, dist_mat, labels, return_inds=False):
        assert len(dist_mat.shape) == 2
        assert dist_mat.shape[0] == dist_mat.shape[1]

        N = dist_mat.shape[0]

        is_pos = F.equal(labels.broadcast_to((N, N)), labels.broadcast_to((N, N)).T).astype('float32')
        is_neg = F.not_equal(labels.broadcast_to((N, N)), labels.broadcast_to((N, N)).T).astype('float32')

        dist_pos = dist_mat * is_pos
        print(dist_pos[0,:])
        #if self.n_inner_pts != 0:
        #    dist_ap = F.max(dist_pos[:self.batch_size, :self.batch_size], axis=1)
        #else:
        dist_ap = F.max(dist_pos, axis=1)
        t2 = dist_ap.reshape((N//2),2)
        print(t2)
        #print('t before', t)
        k2 = F.equal(t2[:,0], t2[:,1]).astype('float32')
        k2 = F.expand_dims(k2, axis = 1)
        k2 = k2.repeat(repeats = 2, axis = 1)
        t2 = t2 + k2*F.array([0.0,-1.0])
        #print('t after', t)
        dist_ap = F.max(t2, axis=1)

        dist_neg = dist_mat * is_neg + 1000*is_pos #F.max(dist_mat, axis=1, keepdims=True) * is_pos
        print(dist_neg[0,:])
        dist_an = F.min(dist_neg, axis=1)
        t1 = dist_an.reshape((N//2),2)
        print(t1)
        #print('t before', t)
        k1 = F.equal(t1[:,0], t1[:,1]).astype('float32')
        k1 = F.expand_dims(k1, axis = 1)
        k1 = k1.repeat(repeats = 2, axis = 1)
        t1 = t1 + k1*F.array([0.0,1.0])
        #print('t after', t)
        dist_an = F.min(t1, axis=1)

        #if self.n_inner_pts != 0:
        #    num_group = N // self.batch_size
        #    dist_an = F.min(F.reshape(dist_an, (num_group, self.batch_size)), axis=0) # include synthetic positives

        return dist_ap, dist_an
