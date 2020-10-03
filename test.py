# encoding: utf-8

import argparse
import os

from runner import Evaluator
import transforms as T
import dataset as D
from model import Model

import mxnet as mx
import numpy as np


parser = argparse.ArgumentParser(description = 'Inference code')

parser.add_argument('--weight_file', default = None, type = str, help = 'path for weight file to be used for inference')
parser.add_argument('--gpu_idx', default=None, type=str, help='gpu index')
parser.add_argument('--image_size', default=227, type=int, help='width and height of input image')
parser.add_argument('--data_name', default='car196', type=str, help='car196 | sop')
parser.add_argument('--ee_l2norm', default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'], help='whether do l2 normalizing augmented embeddings')
parser.add_argument('--backbone', default='googlenet', type=str, help='googlenet')
parser.add_argument('--embed_dim', default=512, type=int, help='dimension of embeddings')
parser.add_argument('--recallk', default='1,2,4,8', type=str, help='k values for recall')
parser.add_argument('--data_dir', default='./data/CARS_196', type=str, help='image_path')
parser.add_argument('--num_instances', default=32, type=int, help='how many instances per class')
parser.add_argument('--batch_size', default=128, type=int,help='batch size')
parser.add_argument('--num_workers', default=0, type=int,help='for data preprocessing')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx)
args.ctx = [mx.gpu(0)]

args.train_meta = './meta/CARS196/train.txt'
args.test_meta = './meta/CARS196/test.txt'

args.recallk = [int(k) for k in args.recallk.split(',')]

model = Model(args.embed_dim, args.ctx)
model.load_parameters(args.weight_file, ctx = args.ctx)
train_transform, test_transform = T.get_transform(image_size=args.image_size)

_, test_loader = D.get_data_loader(args.data_dir, args.train_meta, args.test_meta, train_transform, test_transform,
                                                  args.batch_size, args.num_instances, args.num_workers)

evaluator = Evaluator(model, test_loader, args.ctx)

distmat, labels = evaluator.get_distmat()
recall_at_ranks = evaluator.get_metric_at_ranks(distmat, labels, args.recallk)

for recallk, recall in zip(args.recallk, recall_at_ranks):
    print("R@{:3d}: {:.4f}".format(recallk, recall))
