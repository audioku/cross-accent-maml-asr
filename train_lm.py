import json
import time
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from trainer.lm.charlm_trainer import CharLMTrainer

from utils import constant
from utils.functions import init_lm_transformer_model, init_optimizer
from utils.data_loaders.lm_data_loader import LMDataset, LMDataLoader
from utils.parallel import DataParallel

if __name__ == '__main__':
    args = constant.args

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    # add PAD_CHAR, SOS_CHAR, EOS_CHAR
    labels = constant.PAD_CHAR + constant.SOS_CHAR + constant.EOS_CHAR + labels
    label2id = dict([(labels[i], i) for i in range(len(labels))])
    id2label = dict([(i, labels[i]) for i in range(len(labels))])

    train_dataset = LMDataset(constant.args.train_manifest, label2id, id2label)
    valid_dataset = LMDataset(constant.args.val_manifest, label2id, id2label)

    train_loader = LMDataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size)
    valid_loader = LMDataLoader(valid_dataset, num_workers=args.num_workers, batch_size=args.batch_size)

    start_epoch = 0
    metrics = None

    model = init_lm_transformer_model(args, label2id, id2label)
    opt = init_optimizer(constant.args, model)

    if constant.USE_CUDA:
        model = model.cuda()

    print(model)
    num_epochs = constant.args.epochs

    trainer = CharLMTrainer()
    trainer.train(model, train_loader, valid_loader, opt, start_epoch, num_epochs, label2id, id2label, metrics)