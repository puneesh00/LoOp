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

from .embedding_aug_mx import  get_opt_emb_dis, get_min_dis, get_sum_exp_dis, euclidean_dist, pair_mining
from .optimum_pts_rot import *
from .optimum_pts_lin import *


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
        
        #dist_mat = euclidean_dist(F, embeddings, embeddings)
        #dist_apo, dist_ano = self.hard_example_mining(F, dist_mat, labels)
        
        gen_start_time = time.time()
        #dist_ap, dist_an = get_opt_emb_dis(F, embeddings, labels, self.num_instance, self.l2_norm)
        dist_ap, dist_an, ids, a1l, a2l = get_opt_emb_dis(F, embeddings, labels, self.num_instance, self.l2_norm)
        dist_an = get_min_dis(F, dist_an, ids, a1l, a2l)
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
        
        dist_ap = F.max(dist_pos, axis=1)
        t2 = dist_ap.reshape((N//2),2)
        print(t2)
        k2 = F.equal(t2[:,0], t2[:,1]).astype('float32')
        k2 = F.expand_dims(k2, axis = 1)
        k2 = k2.repeat(repeats = 2, axis = 1)
        t2 = t2 + k2*F.array([0.0,-1.0])
        dist_ap = F.max(t2, axis=1)

        dist_neg = dist_mat * is_neg + 1000*is_pos #F.max(dist_mat, axis=1, keepdims=True) * is_pos
        print(dist_neg[0,:])
        dist_an = F.min(dist_neg, axis=1)
        t1 = dist_an.reshape((N//2),2)
        print(t1)
        k1 = F.equal(t1[:,0], t1[:,1]).astype('float32')
        k1 = F.expand_dims(k1, axis = 1)
        k1 = k1.repeat(repeats = 2, axis = 1)
        t1 = t1 + k1*F.array([0.0,1.0])
        dist_an = F.min(t1, axis=1)

        return dist_ap, dist_an


class LiftedStructureLoss(mx.gluon.loss.Loss):
    def __init__(self, margin=0.2, soft_margin=False, weight=None, batch_axis=0, num_instances=2, n_inner_pts=0, l2_norm=True):
        super(LiftedStructureLoss, self).__init__(weight, batch_axis)
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
        
        gen_start_time = time.time()
        dist_ap, dist_an0, ids, a1l, a2l = get_opt_emb_dis(F, embeddings, labels, self.num_instance, self.l2_norm)
        dist_an = get_sum_exp_dis(F, dist_an0, ids, a1l, a2l)
        print(dist_ap)
        gen_time = time.time() - gen_start_time
        
        loss = F.relu(F.log(dist_an) + dist_ap + self.margin)
        
        total_time = time.time() - total_start_time

        return loss
    
class Npairloss(mx.gluon.loss.Loss):
    def __init__(self,soft_margin=False, weight=None, batch_axis=0, num_instances=2, n_inner_pts=0, l2_norm=False):
        super(Npairloss, self).__init__(weight, batch_axis)
        self.soft_margin = soft_margin
        self.num_instance = num_instances
        self.n_inner_pts = n_inner_pts
        self.batch_size = None
        self.l2_norm = l2_norm

    def hybrid_forward(self, F, embeddings, labels):
        total_start_time = time.time()
        gen_time = 0
        self.batch_size = embeddings.shape[0]

        gen_start_time = time.time()
        dist_ap, dist_an0, ids, a1l, a2l = get_opt_emb_dis(F, embeddings, labels, self.num_instance, l2_norm=False,npair=True )
        dist_an, numf = get_sum_exp_dis(F, -dist_an0, ids, a1l, a2l)
        print(dist_an)
        print(F.exp(-dist_ap))
        gen_time = time.time() - gen_start_time
        
        X1=embeddings[0:self.batch_size:2]
        X2=embeddings[1:self.batch_size:2]
        l2_reg=F.sum(X1*X1, axis = 1)+F.sum(X2*X2, axis = 1)
        print(F.sqrt(F.sum((X1-X2)**2, axis = 1)))

        loss = F.log(1.0 + F.exp(-dist_ap)*dist_an) #+ 0.000075*l2_reg

        total_time = time.time() - total_start_time

        return loss
    
class MSloss(mx.gluon.loss.Loss):
    def __init__(self, th = -0.02, mrg = 0.4, alpha = 2.0, beta = 50.0, soft_margin=False, weight=None, batch_axis=0, num_instances=2, n_inner_pts=0, l2_norm=True):
        super(MSloss, self).__init__(weight, batch_axis)
        self.soft_margin = soft_margin
        self.num_instance = num_instances
        self.n_inner_pts = n_inner_pts
        self.batch_size = None
        self.l2_norm = l2_norm
        self.th = th 
        self.mrg = mrg
        self.alpha = alpha
        self.beta= beta

    def hybrid_forward(self, F, embeddings, labels):
        total_start_time = time.time()
        gen_time = 0
        self.batch_size = embeddings.shape[0]
        count = 0
        
        gen_start_time = time.time()
        dist_ap, dist_an0, ids, a1l, a2l, ind = get_opt_emb_dis(F, embeddings, labels, self.num_instance, self.l2_norm, multisim = True)
        #print('DISTANCES...')
        #print(dist_ap)
        #print(dist_an0)
        dist_ap = (2-dist_ap**2)/2.0
        dist_an0 = (2 -dist_an0**2)/2.0
        dist_neg, dist_pos = pair_mining(F, dist_ap, dist_an0, ids, a1l, a2l, ind, labels, self.num_instance, self.th, self.alpha, self.beta, self.mrg)
        gen_time = time.time() - gen_start_time
        
        ''' 
        sim_mat = F.linalg.gemm2(embeddings, F.transpose(embeddings))
        #print(labels, labels.shape) 

        epsilon = 1e-5
        #loss = list()
        k=0

        for i in range(self.batch_size):
            pos_pair_ = F.contrib.boolean_mask(sim_mat[i],labels == labels[i])
            pos_pair_ = F.contrib.boolean_mask(pos_pair_,pos_pair_ < 1 - epsilon)
            neg_pair_ = F.contrib.boolean_mask(sim_mat[i],labels != labels[i])
            
            ind_neg=neg_pair_ + self.th > F.min(pos_pair_)
            ind_pos=pos_pair_ - self.th < F.max(neg_pair_)
            if F.sum(ind_neg) < 1 or F.sum(ind_pos) < 1:
                count = count + 1
                continue

            neg_pair = F.contrib.boolean_mask(neg_pair_,ind_neg)
            pos_pair = F.contrib.boolean_mask(pos_pair_,ind_pos)

            #print('NEG...', neg_pair)
            #print('POS...', pos_pair)  
            # weighting step
            dis_pos= F.sum(F.exp(-self.alpha * (pos_pair - self.mrg)))
            dis_neg = F.sum(F.exp(self.beta * (neg_pair - self.mrg)))
            #loss.append(pos_loss + neg_loss)
            if k==0:
              k=k+1
              dist_pos=dis_pos
              dist_neg=dis_neg
            else:
              dist_pos=F.concat(dist_pos,dis_pos,dim=0)
              dist_neg=F.concat(dist_neg,dis_neg,dim=0)

        if k==0:
          dist_pos=pos_pair_-pos_pair_
          dist_neg=dist_pos
       
        #print('MEASURE OF emptiness...', float(count)/float(self.batch_size))
        '''
        loss = 1/(self.alpha)*F.log(1.0 + dist_pos) + 1/(self.beta)*F.log(1.0 + dist_neg)

        #total_time = time.time() - total_start_time

        return loss    

class Tripletloss(mx.gluon.loss.Loss):
    def __init__(self, margin = 0.1, soft_margin=False, weight=None, batch_axis=0, num_instances=2, n_inner_pts=0, l2_norm=True):
        super(Tripletloss, self).__init__(weight, batch_axis)
        self.soft_margin = soft_margin
        self.num_instance = num_instances
        self.n_inner_pts = n_inner_pts
        self.batch_size = None
        self.l2_norm = l2_norm
        self.margin = margin

    def hybrid_forward(self, F, embeddings, labels):
        total_start_time = time.time()
        gen_time = 0
        self.batch_size = embeddings.shape[0]
        dim = embeddings.shape[1]
        
        pos_embed = embeddings[:self.batch_size // 2]
        neg_embed = embeddings[self.batch_size // 2:]
        
        X1 = pos_embed[0:self.batch_size // 2:2]
        X2 = pos_embed[1:self.batch_size // 2:2]
        X3 = neg_embed[0:self.batch_size // 2:2]
        X4 = neg_embed[1:self.batch_size // 2:2]
        
        sim = F.arccos(F.sum(X1*X2, axis = 1))
        sim1 = F.arccos(F.sum(X3*X4, axis = 1))
        
        ind = sim>1e-3
        ind1 = sim1>1e-3
        ind1 = ind*ind1
            
        if F.sum(ind1)>0:
            X1 = F.contrib.boolean_mask(X1,ind1)
            X2 = F.contrib.boolean_mask(X2,ind1)
            X3 = F.contrib.boolean_mask(X3,ind1)
            X4 = F.contrib.boolean_mask(X4,ind1)

            gen_start_time = time.time()
            if self.l2_norm:
                dis_an = opt_pts_rot(F.transpose(X1), F.transpose(X2), F.transpose(X3), F.transpose(X4), self.batch_size, dim)
            else:
                dis_an = opt_pts_lin(F.transpose(X1), F.transpose(X2), F.transpose(X3), F.transpose(X4))
            gen_time = time.time() - gen_start_time
        else:
            dis_an = F.sqrt(F.sum((X1-X3)*(X1-X3), axis = 1) + 1e-20)
        
        dis_ap = F.sqrt(F.sum((X1-X2)*(X1-X2), axis = 1) + 1e-20)
        loss = F.relu(dis_ap - dis_an + self.margin)
        total_time = time.time() - total_start_time

        return loss    
