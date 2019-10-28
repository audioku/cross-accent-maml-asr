import json
import time
import math
import numpy as np

import torch
import torch.nn as nn
from torchsummary import summary
from torch.autograd import Variable

from trainer.asr.trainer import Trainer

from utils import constant
from utils.data_loader import SpectrogramDataset, LogFBankDataset, AudioDataLoader, BucketingSampler
from utils.functions import save_model, load_model, init_transformer_model, init_deepspeech_model, init_las_model, init_optimizer
from utils.parallel import DataParallel
import logging

import sys
import os

def compute_num_params(model):
    """
    Computes number of trainable and non-trainable parameters
    """
    sizes = [(np.array(p.data.size()).prod(), int(p.requires_grad)) for p in model.parameters()]
    return sum(map(lambda t: t[0]*t[1], sizes)), sum(map(lambda t: t[0]*(1 - t[1]), sizes))

def generate_labels(labels, train_lang_list, valid_lang_list, prefix_list, lang_list, special_token_list):
    # add PAD_CHAR, SOS_CHAR, EOS_CHAR, UNK_CHAR
    label2id, id2label = {}, {}
    count = 0

    if len(train_lang_list) == 0:
        train_lang_list = [constant.SOS_CHAR] * len(args.train_manifest_list)
    if len(valid_lang_list) == 0:
        valid_lang_list = [constant.SOS_CHAR] * len(args.valid_manifest_list)

    for i in range(len(prefix_list)):
        label2id[prefix_list[i]] = count
        id2label[count] = prefix_list[i]
        count += 1
    
    # print("lang_list:", lang_list)
    for i in range(len(lang_list)):
        label2id[lang_list[i]] = count
        id2label[count] = lang_list[i]
        count += 1

    for i in range(len(labels)):
        if labels[i] not in label2id:
            labels[i] = labels[i]
            label2id[labels[i]] = count
            id2label[count] = labels[i]
            count += 1
        else:
            print("multiple label: ", labels[i])
    return label2id, id2label


if __name__ == '__main__':
    args = constant.args
    print("="*50)
    print("THE EXPERIMENT LOG IS SAVED IN: " + "log/" + args.name)
    print("TRAINING MANIFEST: ", args.train_manifest_list)
    print("VALID MANIFEST: ", args.valid_manifest_list)
    print("TEST MANIFEST: ", args.test_manifest_list)
    print("INPUT TYPE: ", args.input_type)
    print("="*50)

    if not os.path.exists("./log"):
        os.mkdir("./log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    if args.continue_from == '':
        logging.basicConfig(filename="log/" + args.name + ".log", filemode='w+', format='%(asctime)s - %(message)s', level=logging.INFO)
        print("TRAINING FROM SCRATCH")
        logging.info("TRAINING FROM SCRATCH")
    else:
        logging.basicConfig(filename="log/" + args.name + ".log", filemode='a+', format='%(asctime)s - %(message)s', level=logging.INFO)
        print("RESUME TRAINING")
        logging.info("RESUME TRAINING")

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    logging.info(audio_conf)

    early_stop = args.early_stop
    is_accu_loss = args.is_accu_loss
    is_factorized = args.is_factorized
    r = args.r

    train_lang_list = constant.args.train_lang_list
    valid_lang_list = constant.args.valid_lang_list

    train_lang_list = [] if train_lang_list is None else train_lang_list
    valid_lang_list = [] if valid_lang_list is None else valid_lang_list

    if len(train_lang_list) != 0:
        train_lang_list = ["<" + lang.upper() + ">" for lang in train_lang_list]
    if len(valid_lang_list) != 0:
        valid_lang_list = ["<" + lang.upper() + ">" for lang in valid_lang_list]

    prefix_list = [constant.PAD_CHAR, constant.SOS_CHAR, constant.EOS_CHAR, constant.UNK_CHAR]
    lang_list = list(set(train_lang_list + valid_lang_list))
    special_token_list = prefix_list + lang_list

    # Source encoders
    with open(args.labels_path, encoding="utf-8") as label_file:
        labels = json.load(label_file)
    src_label2id, src_id2label = generate_labels(labels, train_lang_list, valid_lang_list, prefix_list, lang_list, special_token_list)
    trg_label2ids, trg_id2labels = [], []

    # Target decoders
    if len(args.trg_label_list) == len(args.train_lang_list) and not args.combine_decoder:
        # LANGUAGE-SPECIFIC TARGET DECODERS
        print("LANGUAGE-SPECIFIC TARGET DECODERS")
        for i in range(len(args.trg_label_list)):
            with open(args.trg_label_list[i], encoding="utf-8") as label_file:
                labels = json.load(label_file)
            label2id, id2label = generate_labels(labels, train_lang_list, valid_lang_list, prefix_list, lang_list, special_token_list)
            
            trg_label2ids.append(label2id)
            trg_id2labels.append(id2label)
            print("Target decoder ", i)
            print(len(label2id), len(id2label))
            print("train lang list:", train_lang_list)
            print("valid lang list:", valid_lang_list)
            print(len(label2id))
    else:
        with open(args.labels_path, encoding="utf-8") as label_file:
            labels = json.load(label_file)
        label2id, id2label = generate_labels(labels, train_lang_list, valid_lang_list, prefix_list, lang_list, special_token_list)

        trg_label2ids.append(label2id)
        trg_id2labels.append(id2label)

    print("src_label2id:", len(src_label2id), " src_id2label", len(src_id2label))
    print("trg_id2labels:", trg_id2labels, "trg_label2ids", trg_label2ids)

    if args.feat == "spectrogram":
        train_data = SpectrogramDataset(audio_conf, lang_list=train_lang_list, all_lang_list=train_lang_list, manifest_filepath_list=args.train_manifest_list, src_label2id=src_label2id, trg_label2ids=trg_label2ids, normalize=True, augment=args.augment, input_type=args.input_type, is_train=True)
    elif args.feat == "logfbank":
        train_data = LogFBankDataset(audio_conf, lang_list=train_lang_list, all_lang_list=train_lang_list, manifest_filepath_list=args.train_manifest_list, src_label2id=src_label2id, trg_label2ids=trg_label2ids, normalize=True, augment=args.augment, input_type=args.input_type, is_train=True)
    train_sampler = BucketingSampler(train_data, batch_size=args.batch_size)
    train_loader = AudioDataLoader(train_data, num_workers=args.num_workers, batch_sampler=train_sampler)

    valid_loader_list, test_loader_list = [], []
    for i in range(len(args.valid_manifest_list)):
        # lang_id = 10000
        lang_id = 0
        for j in range(len(train_lang_list)):
            lang = train_lang_list[j]
            if valid_lang_list[i] == lang:
                lang_id = j
                break
        print("lang id:", lang_id)
        if len(trg_label2ids) == 1:
            lang_id = 0
        print(len(trg_label2ids[lang_id]))
        if args.feat == "spectrogram":
            valid_data = SpectrogramDataset(audio_conf, lang_list=[valid_lang_list[i]], all_lang_list=train_lang_list, manifest_filepath_list=[args.valid_manifest_list[i]], src_label2id=src_label2id, trg_label2ids=[trg_label2ids[lang_id]], normalize=True, augment=args.augment, input_type=args.input_type)
        elif args.feat == "logfbank":
            valid_data = LogFBankDataset(audio_conf, lang_list=[valid_lang_list[i]], all_lang_list=train_lang_list, manifest_filepath_list=[args.valid_manifest_list[i]], src_label2id=src_label2id, trg_label2ids=[trg_label2ids[lang_id]], normalize=True, augment=False, input_type=args.input_type)
        valid_sampler = BucketingSampler(valid_data, batch_size=args.batch_size)
        valid_loader = AudioDataLoader(valid_data, num_workers=args.num_workers)
        valid_loader_list.append(valid_loader)

    start_epoch = 0
    metrics = None
    loaded_args = None
    if constant.args.continue_from != "":
        logging.info("Continue from checkpoint:" + constant.args.continue_from)
        model, opt, epoch, metrics, loaded_args, src_label2id, src_id2label, trg_label2ids, trg_id2labels = load_model(
            constant.args.continue_from)
        start_epoch = (epoch)  # index starts from zero
        verbose = constant.args.verbose
    else:
        if constant.args.model == "TRFS":
            model = init_transformer_model(constant.args, src_label2id, src_id2label, trg_label2ids, trg_id2labels, is_factorized=is_factorized, r=r)
            opt = init_optimizer(constant.args, model, "noam")
        elif constant.args.model == "DEEPSPEECH":
            model = init_deepspeech_model(constant.args, label2id, id2label)
            opt = init_optimizer(constant.args, model, "sgd")
        elif constant.args.model == "LAS":
            model = init_las_model(constant.args, label2id, id2label)
            opt = init_optimizer(constant.args, model, "noam")
        else:
            logging.info("The model is not supported, check args --h")
    
    loss_type = args.loss

    if constant.USE_CUDA:
        model = model.cuda()

    # Parallelize the batch
    if args.parallel:
        device_ids = args.device_ids
        model = DataParallel(model, device_ids=device_ids)
    else:
        if loaded_args != None:
            if loaded_args.parallel:
                logging.info("unwrap from DataParallel")
                model = model.module

    logging.info(model)
    num_epochs = constant.args.epochs

    print("Parameters: {}(trainable), {}(non-trainable)".format(
            compute_num_params(model)[0], compute_num_params(model)[1]))

    trainer = Trainer()
    trainer.train(model, train_loader, train_sampler, valid_loader_list, special_token_list, opt, loss_type, start_epoch, num_epochs, src_label2id, src_id2label, trg_label2ids, trg_id2labels, metrics, early_stop=early_stop, is_accu_loss=is_accu_loss)
