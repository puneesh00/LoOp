'''
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tqdm import tqdm

import mxnet as mx
import numpy as np


class Evaluator(object):
    def __init__(self, model, test_loader, ctx):
        self.model = model
        self.test_loader = test_loader
        self.ctx = ctx

    def _eval_step(self, inputs):
        images, instance_ids, category_ids, view_ids = inputs
        data = mx.gluon.utils.split_and_load(images, self.ctx, even_split=False)
        instance_ids = instance_ids.asnumpy()
        view_ids = view_ids.asnumpy()
        feats = []
        for d in data:
            feats.append(self.model(d))
        feats = mx.nd.concatenate(feats, axis=0)
        return feats, instance_ids, view_ids
        
    def get_feats(self):
        print('Extracting eval features...')
        features, labels = [], []
        for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            feature, instance_ids, view_ids = self._eval_step(inputs)
            features.append(feature.asnumpy())
            labels.extend(instance_ids)
        features = np.concatenate(features)
        labels = np.asarray(labels)
        return features, labels

    
    def get_distmat(self):
        print('Extracting eval features...')
        features, labels = [], []
        for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            feature, instance_ids, view_ids = self._eval_step(inputs)
            features.append(feature.asnumpy())
            labels.extend(instance_ids)
        features = np.concatenate(features)
        labels = np.asarray(labels)
        
        m = features.shape[0]
        squared_sum_features = np.sum(features ** 2.0, axis=1, keepdims=True)
        distmat = squared_sum_features + squared_sum_features.transpose() - (2.0 * np.dot(features, features.transpose()))

        return distmat, labels


    def get_metric_at_ranks(self, distmat, labels, ranks):
        np.fill_diagonal(distmat, 100000.0)

        recall_at_ranks = []

        recall_dict = {k: 0 for k in ranks}

        max_k = np.max(ranks)

        # do partition
        arange_idx = np.arange(len(distmat))[:,None]
        part_idx = np.argpartition(distmat, max_k, axis=1)[:,:max_k]
        part_mat = distmat[arange_idx, part_idx]

        # do sort
        sorted_idx = np.argsort(part_mat, axis=1)#[::-1]
        top_k_idx = part_idx[arange_idx, sorted_idx]

        for top_k, gt in zip(top_k_idx, labels):
            top_k_labels = labels[top_k]
            for r in ranks:
                if gt in top_k_labels[:r]:
                    recall_dict[r] += 1

        for r in ranks:
            recall_at_ranks.append(recall_dict[r] / len(distmat))

        return recall_at_ranks
    
    def evaluate_recall(features, labels, neighbours):
        """
        A function that calculate the recall score of a embedding
        :param features: The 2-d array of the embedding
        :param labels: The 1-d array of the label
        :param neighbours: A 1-d array contains X in Recall@X
        :return: A 1-d array of the Recall@X
        """
        dims = features.shape
 
        #D2 = distance_matrix(features)

        #D2 = dist_mat(features)

        # set diagonal to very high number
        num = dims[0]
        parts = 100
        parts_x = num // parts
        for i in range(parts):
          recalls = []
          feat1 = features[i*parts_x:(i+1)*parts_x]
          D = dist_mat(feat1, features)
          D = np.sqrt(np.abs(D))
          #diagn = np.diag([float('inf') for i in range(0, D.shape[0])])
          diagn = np.zeros((parts_x, num))
          for k in range(parts_x):
            diagn[k, (i*parts_x + k)] = float('inf')
          D = D + diagn
          lab = labels[i*parts_x:(i+1)*parts_x]
          for j in range(0, np.shape(neighbours)[0]):
              recall_i = compute_recall_at_K(D, neighbours[j], lab, labels, parts_x)
              recalls.append(recall_i)
          recalls = np.array(recalls)
          if i==0:
            RECALL = recalls/float(num)*float(parts_x)
          else:
            RECALL+=recalls/float(num)*float(parts_x)

        feat = features[(i+1)*parts_x:num]
        D = dist_mat(feat, features)
        diagn = np.zeros((D.shape[0], num))
        for k in range(D.shape[0]):
           diagn[k, ((i+1)*parts_x+k)] = float('inf')
        D = D + diagn
        lab = labels[(i+1)*parts_x:num]
        recalls = []  
        for j in range(0,np.shape(neighbours)[0]):
            recall_i = compute_recall_at_K(D, neighbours[j], lab, labels, D.shape[0])
            recalls.append(recall_i)
        recalls = np.array(recalls)
        RECALL+=recalls/float(num)*float(D.shape[0])

        print('done')
        print(RECALL)
        return RECALL

    def compute_recall_at_K(D, K, lab, class_ids, num):
        num_correct = 0
        for i in range(0, num):
            this_gt_class_idx = lab[i]
            this_row = D[i, :]
            inds = np.array(np.argsort(this_row))
            knn_inds = inds[0:K]
            knn_class_inds = [class_ids[i] for i in knn_inds]
            if sum(np.in1d(knn_class_inds, this_gt_class_idx)) > 0:
                num_correct = num_correct + 1
        recall = float(num_correct)/float(num)

        print('num_correct:', num_correct)
        print('num:', num)
        print("K: %d, Recall: %.4f\n" % (K, recall))
        return recall
    
    def dist_mat(X, features):
      squared_X = np.sum(X**2.0, axis=1, keepdims=True) 
      squared_f = np.sum(features**2.0, axis=1, keepdims=True)
      distmat = squared_X + squared_f.transpose() - (2.0 * np.dot(X, features.transpose()))                                                                    
      return distmat
