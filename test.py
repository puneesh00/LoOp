# encoding: utf-8

import argparse
import os

from runner import Evaluator
import transforms as T
import dataset as D
from model import Model

import mxnet as mx
import numpy as np

import math
from scipy.special import comb
from sklearn import cluster
from sklearn import neighbors
from sklearn.decomposition import PCA
import copy
from tqdm import tqdm


parser = argparse.ArgumentParser(description = 'Inference code')

parser.add_argument('--weight_file', default = None, type = str, help = 'path for weight file to be used for inference')
parser.add_argument('--gpu_idx', default=None, type=str, help='gpu index')
parser.add_argument('--image_size', default=227, type=int, help='width and height of input image')
parser.add_argument('--data_name', default='CARS196', type=str, help='CARS196 or CUB200 or SOP')
parser.add_argument('--ee_l2norm', default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'], help='whether do l2 normalizing augmented embeddings')
parser.add_argument('--backbone', default='googlenet', type=str, help='googlenet')
parser.add_argument('--embed_dim', default=512, type=int, help='dimension of embeddings')
parser.add_argument('--recallk', default='1,2,4,8', type=str, help='k values for recall')
parser.add_argument('--data_dir', default='./data/CARS_196', type=str, help='image_path')
parser.add_argument('--num_instances', default=8, type=int, help='how many instances per class')
parser.add_argument('--batch_size', default=32, type=int,help='batch size')
parser.add_argument('--num_workers', default=0, type=int,help='for data preprocessing')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx)
args.ctx = [mx.gpu(0)]


if args.data_name=='CARS196':
  args.train_meta = './meta/CARS196/train.txt'
  args.test_meta = './meta/CARS196/test.txt'
  n_clusters=98
elif args.data_name=='CUB200':
  args.train_meta = './meta/CUB_200_2011/train.txt'
  args.test_meta = './meta/CUB_200_2011/test.txt'
  n_clusters=100
elif args.data_name=='SOP':
  args.train_meta = './meta/SOP/train.txt'
  args.test_meta = './meta/SOP/test.txt'
  n_clusters=11316
args.recallk = [int(k) for k in args.recallk.split(',')]

def evaluate_cluster(feats, labels, n_clusters):
    """
    A function that calculate the NMI as well as F1 of a given embedding
    :param feats: The feature (embedding)
    :param labels: The labels
    :param n_clusters: How many classes
    :return: The NMI and F1 score of the given embedding
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=1).fit(feats)
    centers = kmeans.cluster_centers_

    # k-nearest neighbors
    neigh = neighbors.KNeighborsClassifier(n_neighbors=1)
    neigh.fit(centers, range(len(centers)))

    idx_in_centers = neigh.predict(feats)
    num = len(feats)
    d = np.zeros(num)
    for i in range(num):
        d[i] = np.linalg.norm(feats[i, :] - centers[idx_in_centers[i], :])

    labels_pred = np.zeros(num)
    for i in np.unique(idx_in_centers):
        index = np.where(idx_in_centers == i)[0]
        ind = np.argmin(d[index])
        cid = index[ind]
        labels_pred[index] = cid
    nmi, f1 = compute_clutering_metric(labels, labels_pred)
    return nmi, f1


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

def distance_matrix(X):
    X = np.matrix(X)
    m = X.shape[0]
    t = np.matrix(np.ones([m, 1]))
    x = np.matrix(np.empty([m, 1]))
    for i in range(0, m):
        n = np.linalg.norm(X[i, :])
        x[i] = n * n
    D = x * np.transpose(t) + t * np.transpose(x) - 2 * X * np.transpose(X)
    return D

                                                                       
def dist_mat(X, features):
  squared_X = np.sum(X**2.0, axis=1, keepdims=True) 
  squared_f = np.sum(features**2.0, axis=1, keepdims=True)
  distmat = squared_X + squared_f.transpose() - (2.0 * np.dot(X, features.transpose()))                                                                    
  return distmat

                                                                       
def compute_clutering_metric(idx, item_ids):

    N = len(idx)

    # cluster centers
    centers = np.unique(idx)
    num_cluster = len(centers)
    # print('Number of clusters: #d\n' % num_cluster);

    # count the number of objects in each cluster
    count_cluster = np.zeros(num_cluster)
    for i in range(num_cluster):
        count_cluster[i] = len(np.where(idx == centers[i])[0])

    # build a mapping from item_id to item index
    keys = np.unique(item_ids)
    num_item = len(keys)
    values = range(num_item)
    item_map = dict()
    for i in range(len(keys)):
        item_map.update([(keys[i], values[i])])

    # count the number of objects of each item
    count_item = np.zeros(num_item)
    for i in range(N):
        index = item_map[item_ids[i]]
        count_item[index] = count_item[index] + 1

    # compute purity
    purity = 0
    for i in range(num_cluster):
        member = np.where(idx == centers[i])[0]
        member_ids = item_ids[member]

        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1
        purity = purity + max(count)

    # compute Normalized Mutual Information (NMI)
    count_cross = np.zeros((num_cluster, num_item))
    for i in range(N):
        index_cluster = np.where(idx[i] == centers)[0]
        index_item = item_map[item_ids[i]]
        count_cross[index_cluster, index_item] = count_cross[index_cluster, index_item] + 1

    # mutual information
    I = 0
    for k in range(num_cluster):
        for j in range(num_item):
            if count_cross[k, j] > 0:
                s = count_cross[k, j] / N * math.log(N * count_cross[k, j] / (count_cluster[k] * count_item[j]))
                I = I + s

    # entropy
    H_cluster = 0
    for k in range(num_cluster):
        s = -count_cluster[k] / N * math.log(count_cluster[k] / float(N))
        H_cluster = H_cluster + s

    H_item = 0
    for j in range(num_item):
        s = -count_item[j] / N * math.log(count_item[j] / float(N))
        H_item = H_item + s

    NMI = 2 * I / (H_cluster + H_item)

    # compute True Positive (TP) plus False Positive (FP)
    tp_fp = 0
    for k in range(num_cluster):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2)

    # compute True Positive (TP)
    tp = 0
    for k in range(num_cluster):
        member = np.where(idx == centers[k])[0]
        member_ids = item_ids[member]

        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1

        for i in range(num_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2)

    # False Positive (FP)
    fp = tp_fp - tp

    # compute False Negative (FN)
    count = 0
    for j in range(num_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2)

    fn = count - tp

    # compute F measure
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    beta = 1
    F = (beta*beta + 1) * P * R / (beta*beta * P + R)

    return NMI, F


model = Model(args.embed_dim, args.ctx)
model.load_parameters(args.weight_file, ctx = args.ctx)
train_transform, test_transform = T.get_transform(image_size=args.image_size)

_, test_loader = D.get_data_loader(args.data_dir, args.train_meta, args.test_meta, train_transform, test_transform,
                                                  args.batch_size, args.num_instances, args.num_workers)

evaluator = Evaluator(model, test_loader, args.ctx)

feats, labels = evaluator.get_feats()

#nmi,f1=evaluate_cluster(feats,labels,n_clusters)
#print(nmi,f1)
pca1 = PCA(n_components=3)
pca2 = PCA(n_components=3)
pca3 = PCA(n_components=3)
pca4 = PCA(n_components=3)

lab = np.asarray(labels)
ind1 = np.squeeze(np.argwhere(lab==1))
ind2 = np.squeeze(np.argwhere(lab==2))
ind3 = np.squeeze(np.argwhere(lab==3))
ind4 = np.squeeze(np.argwhere(lab==4))

feat1 = feats[ind1]
feat2 = feats[ind2]
feat3 = feats[ind3]
feat4 = feats[ind4]

pca1.fit(feat1)
pca2.fit(feat2)
pca3.fit(feat3)
pca4.fit(feat4)

print(pca1.singular_values_)
print(pca2.singular_values_)
print(pca3.singular_values_)
print(pca4.singular_values_)

recall=evaluate_recall(feats,labels,args.recallk)

#distmat, labels = evaluator.get_distmat()
#recall_at_ranks = evaluator.get_metric_at_ranks(distmat, labels, args.recallk)

#for recallk, recall in zip(args.recallk, recall_at_ranks):
#    print("R@{:3d}: {:.4f}".format(recallk, recall))

