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

def euclidean_dist(F, x, y, clip_min=1e-12, clip_max=1e12):
    m, n = x.shape[0], y.shape[0]

    squared_x = F.power(x, 2).sum(axis=1, keepdims=True).broadcast_to((m, n))
    squared_y = F.power(y, 2).sum(axis=1, keepdims=True).broadcast_to((n, m)).T

    dist = squared_x + squared_y
    dist = dist - 2 * F.dot(x, y.T)
    dist = dist.clip(a_min=clip_min, a_max=clip_max).sqrt()

    return dist

def euclidean_dist_alt(F,x,y):
    n = x.shape[0]
    z = F.expand_dims(x, axis = 1)
    z = z.repeat(repeats = n, axis = 1)
    z1 = F.transpose(z, axes=(1,0,2))
    dist = F.sqrt(F.sum((z-z1)*(z-z1), axis = 2)+1e-20)

    return dist
'''
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
'''
def concat(F,X1,X2,X1l,X2l):
    n=X1.shape[0]
    ids=F.array([i for i in range(n)])

    ind1=[i for i in range(n-1) for i in range(n-1-i)]
    ind2=[i+n for i in range(n-1) for i in range(-(n-1-i),0)]
    
    a2l=ids[ind2]
    a1l=ids[ind1]
    
    X3=X1[ind2]
    X3l=X1l[ind2]
    X1=X1[ind1]
    X1l=X1l[ind1]
    X4=X2[ind2]
    X2=X2[ind1]

    ind=[i for i in range(len(X1l)) if X1l[i]!=X3l[i]]
    X1=X1[ind]
    X2=X2[ind]
    X3=X3[ind]
    X4=X4[ind]
    
    a1l=a1l[ind]
    a2l=a2l[ind]
    
    return X1,X2,X3,X4,a1l,a2l,ids 

def get_min_dis(F, dis,label,a1l,a2l):
    k=0
    for l in range(label.shape[0]):
      id1=[i for i in range(a1l.shape[0]) if a1l[i]==label[l] ]
      id2=[i for i in range(a2l.shape[0]) if a2l[i]==label[l] ]
      
      if len(id1)>0 or len(id2)>0:
        if len(id1)<1:
          dist=F.min(dis[id2])
        elif len(id2)<1:
          dist=F.min(dis[id1])
        else:
          dist=F.min(F.concat(F.min(dis[id1]), F.min(dis[id2]), dim=0))
        
        if k==0:
          k=k+1
          dis=dist
        else:
          dis=F.concat(dis,dist,dim=0)
    print('dis_an',dis)
                            
    return dis
    
def get_sum_dis(F, dis,label,a1l,a2l):
    k=0
    for l in range(label.shape[0]):
      id1=[i for i in range(a1l.shape[0]) if a1l[i]==label[l] ]
      id2=[i for i in range(a2l.shape[0]) if a2l[i]==label[l] ]
      
      if len(id1)>0 or len(id2)>0:
        if len(id1)<1:
          dist=F.sum(dis[id2])
        elif len(id2)<1:
          dist=F.sum(dis[id1])
        else:
          dist=F.sum(F.concat(F.sum(dis[id1]), F.sum(dis[id2]), dim=0))
        
        if k==0:
          k=k+1
          dis=dist
        else:
          dis=F.concat(dis,dist,dim=0)
        
    return dis

def get_opt_emb_dis(F, embeddings, labels, num_instance, l2_norm=True):
    batch_size = embeddings.shape[0]
    dim=embeddings.shape[1]

    assert num_instance % 2 == 0, 'num_instance should be even number for simple implementation'

    X1=embeddings[0:batch_size:2]
    X2=embeddings[1:batch_size:2]
    X1l=labels[0:batch_size:2]
    X2l=labels[1:batch_size:2]
    labelsorg = labels
    labels=labels[0:batch_size:num_instance]
   
    print('labelsorg shape', labelsorg.shape)
    sim=F.arccos(F.sum(X1*X2, axis = 1))
    ind=[i for i in range(sim.shape[0]) if sim[i]>1e-3]
    indx=[]
    
    if len(ind)>1:
      indx=[i for i in range(len(labels)) if labels[i] in X1l[ind]]
      if len(indx)>1:
        X1=X1[ind]
        X2=X2[ind]
        X1l=X1l[ind]
        X2l=X2l[ind]
      
    X1, X2, X3, X4, a1l, a2l, ids = concat(F,X1,X2,X1l,X2l)
    
    if len(indx)>1:
      batch_size = X1.shape[0]
      dis = opt_pts_rot(F.transpose(X1), F.transpose(X2), F.transpose(X3), F.transpose(X4), batch_size, dim)  
    else:
      dis = F.sqrt(F.sum((X1-X3)*(X1-X3), axis=1)+1e-20)
    
    dis_ap = euclidean_dist_alt(F, embeddings, embeddings)
    N = dis_ap.shape[0]
    is_pos = F.equal(labelsorg.broadcast_to((N, N)), labelsorg.broadcast_to((N, N)).T).astype('float32')
    #print(is_pos)
    dis_pos = dis_ap * is_pos
    dis_ap1 = F.max(dis_pos, axis=1)
    
    num_pairs = N // 2
    t = dis_ap1.reshape(num_pairs,2)
    print('t before', t)
    k = F.equal(t[:,0], t[:,1]).astype('float32')
    k1 = F.expand_dims(k, axis = 1)
    k1 = k1.repeat(repeats = 2, axis = 1)
    t = t + k1*F.array([0.0,-1.0])
    print('t after', t)
    dis_ap1 = F.max(t, axis=1)
    
    if len(indx)>1:
        dis_ap1 = dis_ap1[ind]
    print('dis_ap', dis_ap1)
    
    #dis_an = get_min_dis(F, dis, ids, a1l, a2l) #for hphn-triplet
    #dis_an = get_sum_dis(F, dis, ids, a1l, a2l) #for lifted-struct
      
    return dis_ap1, dis, ids, a1l, a2l #dis_an
