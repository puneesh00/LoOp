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
#import mxnet.ndarray as mxn
from .optimum_pts_mx import *

def get_embedding_aug(F, embeddings, labels, num_instance, n_inner_pts, l2_norm=True):
    batch_size = embeddings.shape[0]

    assert num_instance % 2 == 0, 'num_instance should be even number for simple implementation'
    swap_axes_list = [i + 1 if i % 2 == 0 else i - 1 for i in range(batch_size)]
    swap_embeddings = embeddings[swap_axes_list]
    pos = embeddings
    anchor = swap_embeddings
    concat_embeddings = embeddings.copy()
    concat_labels = labels.copy()
    n_pts = n_inner_pts
    l2_normalize = l2_norm
    total_length = float(n_pts + 1)
    for n_idx in range(n_pts):
        left_length = float(n_idx + 1)
        right_length = total_length - left_length
        inner_pts = (anchor * left_length + pos * right_length) / total_length
        if l2_normalize:
            inner_pts = F.L2Normalization(inner_pts)
        concat_embeddings = F.concat(concat_embeddings, inner_pts, dim=0)
        concat_labels = F.concat(concat_labels, labels, dim=0)

    return concat_embeddings, concat_labels

def concat(X1,X2,X1l,X2l):
    n=X1.shape[0]

    ind1=[i for i in range(n-1) for i in range(n-1-i)]
    ind2=[i+n for i in range(n-1) for i in range(-(n-1-i),0)]
    #print(ind1)
    #print(ind2)
    #print(X1l)
    #print(X1l[ind1])
    #print(X1l[ind2])
    
    X3=X1[ind2]
    X3l=X1l[ind2]
    X1=X1[ind1]
    X1l=X1l[ind1]
    X4=X2[ind2]
    X4l=X2l[ind2]
    X2=X2[ind1]
    X2l=X2l[ind1]
    '''

    X1=X1[ind1]
    X1l=X1l[ind1]
    X3=X1[ind2]
    X3l=X1l[ind2]
    X2=X2[ind1]
    X2l=X2l[ind1]
    X4=X2[ind2]
    X4l=X2l[ind2]
    '''
    return X1,X2,X3,X4,X1l,X2l,X3l,X4l

def get_min_dis(F, dis,label,a1l,a2l):
    #min_dis=mxn.zeros(label.shape)
    for l in range(label.shape[0]):
      if l==0:
        id1=[i for i in range(a1l.shape[0]) if a1l[i]==label[l] ]
        #print('id1',id1)
        min_dis=F.min(dis[id1])

      elif l==label.shape[0]-1:
        id2=[i for i in range(a2l.shape[0]) if a2l[i]==label[l] ]
        #print('id2',id2)
        min_dis=F.concat(min_dis,F.min(dis[id2]),dim=0)

      else:
        id1=[i for i in range(a1l.shape[0]) if a1l[i]==label[l] ]
        #print('id1',id1)
        id2=[i for i in range(a2l.shape[0]) if a2l[i]==label[l] ]
        #print('id2',id2)
        min_dis=F.concat(min_dis, F.min(F.concat(F.min(dis[id1]), F.min(dis[id2]), dim=0)), dim=0)
    return min_dis

def get_opt_emb_dis(F, embeddings, labels, num_instance, l2_norm=True):
    batch_size = embeddings.shape[0]
    dim=embeddings.shape[1]

    assert num_instance % 2 == 0, 'num_instance should be even number for simple implementation'

    X1=embeddings[0:batch_size:2]
    X2=embeddings[1:batch_size:2]
    X1l=labels[0:batch_size:2]
    X2l=labels[1:batch_size:2]
    
    #print(labels)
    sim=F.arccos(F.sum(X1*X2, axis = 1))
    ind=[i for i in range(sim.shape[0]) if sim[i]>1e-5]
    
    if len(ind)>1:
      X1=X1[ind]
      X2=X2[ind]
      X1l=X1l[ind]
      X2l=X2l[ind]
      
    X1n=X1 
    X2n=X2 
    labels = X1l
    X1, X2, X3, X4, X1l, X2l, X3l, X4l = concat(X1,X2,X1l,X2l)
    
    if len(ind)>1:
      batch_size = X1.shape[0]
      dis = opt_pts_rot(F.transpose(X1), F.transpose(X2), F.transpose(X3), F.transpose(X4), batch_size, dim)   
      
    else:
      dis = F.sqrt(F.sum((X1-X3)*(X1-X3), axis=1)+1e-20)
      
    dis_ap = F.sqrt(F.sum((X1n-X2n)*(X1n-X2n), axis=1)+1e-20)
    dis_an = get_min_dis(F, dis,labels,X1l,X3l)
      
    return dis_ap, dis_an
