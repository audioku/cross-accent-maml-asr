import torch
import os
import math
import torch.nn as nn
import logging

from models.asr.las import LAS, LASEncoder, LASDecoder
from models.asr.deepspeech import DeepSpeech
from models.asr.transformer import Transformer, Encoder, Decoder
from models.lm.transformer_lm import TransformerLM
from utils.optimizer import NoamOpt, AnnealingOpt
from utils import constant
from utils.parallel import DataParallel


def save_model(model, epoch, opt, metrics, src_label2id, src_id2label, trg_label2ids, trg_id2labels, best_model=False):
    """
    Saving model, TODO adding history
    """
    if best_model:
        save_path = "{}/{}/best_model.th".format(
            constant.args.save_folder, constant.args.name)
    else:
        save_path = "{}/{}/epoch_{}.th".format(constant.args.save_folder,
                                               constant.args.name, epoch)

    if not os.path.exists(constant.args.save_folder + "/" + constant.args.name):
        os.makedirs(constant.args.save_folder + "/" + constant.args.name)

    print("SAVE MODEL to", save_path)
    logging.info("SAVE MODEL to " + save_path)
    if constant.args.loss == "ce":
        args = {
            'src_label2id': src_label2id,
            'src_id2label': src_id2label,
            'trg_label2ids': trg_label2ids,
            'trg_id2labels': trg_id2labels,
            'args': constant.args,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.optimizer.state_dict(),
            'optimizer_params': {
                '_step': opt._step,
                '_rate': opt._rate,
                'warmup': opt.warmup,
                'factor': opt.factor,
                'model_size': opt.model_size
            },
            'metrics': metrics
        }
    elif constant.args.loss == "ctc":
        args = {
            'label2id': label2id,
            'id2label': id2label,
            'args': constant.args,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.optimizer.state_dict(),
            'optimizer_params': {
                'lr': opt.lr,
                'lr_anneal': opt.lr_anneal
            },
            'metrics': metrics
        }
    else:
        print("Loss is not defined")
        logging.info("Loss is not defined")
    torch.save(args, save_path)


def load_model(load_path, train=True):
    """
    Loading model
    args:
        load_path: string
    """
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))

    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    if 'args' in checkpoint:
        args = checkpoint['args']

    src_label2id = checkpoint['src_label2id']
    src_id2label = checkpoint['src_id2label']
    trg_label2ids = checkpoint['trg_label2ids']
    trg_id2labels = checkpoint['trg_id2labels']
    is_factorized = args.is_factorized
    r = args.r

    # args.feat_extractor = "vgg_cnn"
    args.k_lr = 1
    args.min_lr = 1e-6

    model = init_transformer_model(args, src_label2id, src_id2label, trg_label2ids, trg_id2labels, train=train, is_factorized=is_factorized, r=r)
    model.load_state_dict(checkpoint['model_state_dict'])
    if args.cuda:
        print("CUDA")
        model = model.cuda()
    else:
        model = model.cpu()

    opt = init_optimizer(args, model)
    if opt is not None:
        opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if constant.args.loss == "ce":
            opt._step = checkpoint['optimizer_params']['_step']
            opt._rate = checkpoint['optimizer_params']['_rate']
            opt.warmup = checkpoint['optimizer_params']['warmup']
            opt.factor = checkpoint['optimizer_params']['factor']
            opt.model_size = checkpoint['optimizer_params']['model_size']
        elif constant.args.loss == "ctc":
            opt.lr = checkpoint['optimizer_params']['lr']
            opt.lr_anneal = checkpoint['optimizer_params']['lr_anneal']
        else:
            print("Need to define loss type")
            logging.info("Need to define loss type")

    return model, opt, epoch, metrics, args, src_label2id, src_id2label, trg_label2ids, trg_id2labels


def init_optimizer(args, model, opt_type="noam"):
    dim_input = args.dim_input
    warmup = args.warmup
    lr = args.lr

    if opt_type == "noam":
        opt = NoamOpt(dim_input, args.k_lr, warmup, torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9), min_lr=args.min_lr)
    elif opt_type == "sgd":
        opt = AnnealingOpt(lr, args.lr_anneal, torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, nesterov=True))
    else:
        opt = None
        print("Optimizer is not defined")

    return opt


def init_las_model(args, label2id, id2label):
    """
    Initialize a new Listen-Attend-Spell object
    """
    dim_input = args.dim_input
    dim_model = args.dim_model
    dim_emb = args.dim_emb
    num_layers = args.num_layers
    dropout = args.dropout

    if hasattr(args, 'bidirectional'):
        bidirectional = args.bidirectional
    else:
        bidirectional = False
    
    tgt_max_len = args.tgt_max_len
    emb_trg_sharing = args.emb_trg_sharing

    encoder = LASEncoder(dim_input, dim_model=dim_model,
                         num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
    decoder = LASDecoder(id2label, num_src_vocab=len(label2id), num_trg_vocab=len(label2id), num_layers=num_layers,
                         dim_emb=dim_emb, dim_model=dim_model, dropout=dropout, trg_max_length=tgt_max_len, emb_trg_sharing=emb_trg_sharing)
    model = LAS(encoder, decoder)

    return model


def init_transformer_model(args, src_label2id, src_id2label, trg_label2ids, trg_id2labels, train=True, is_factorized=False, r=100):
    """
    Initiate a new transformer object
    """
    if args.feat_extractor == 'emb_cnn':
        hidden_size = int(math.floor(
            (args.sample_rate * args.window_size) / 2) + 1)
        hidden_size = int(math.floor(hidden_size - 41) / 2 + 1)
        hidden_size = int(math.floor(hidden_size - 21) / 2 + 1)
        hidden_size *= 32
        args.dim_input = hidden_size
    elif args.feat_extractor == 'vgg_cnn':
        hidden_size = int(math.floor((args.sample_rate * args.window_size) / 2) + 1) # 161
        hidden_size = int(math.floor(int(math.floor(hidden_size)/2)/2)) * 128 # divide by 2 for maxpooling
        args.dim_input = hidden_size
        if args.feat == "logfbank":
            args.dim_input = 2560
    elif args.feat_extractor == 'large_cnn':
        hidden_size = int(math.floor((args.sample_rate * args.window_size) / 2) + 1) # 161
        hidden_size = int(math.floor(int(math.floor(hidden_size)/2)/2)) * 64 # divide by 2 for maxpooling
        args.dim_input = hidden_size
    else:
        print("the model is initialized without feature extractor")

    num_enc_layers = args.num_enc_layers
    num_dec_layers = args.num_dec_layers
    num_heads = args.num_heads
    dim_model = args.dim_model
    dim_key = args.dim_key
    dim_value = args.dim_value
    dim_input = args.dim_input
    dim_inner = args.dim_inner
    dim_emb = args.dim_emb
    src_max_len = args.src_max_len
    tgt_max_len = args.tgt_max_len
    dropout = args.dropout
    emb_trg_sharing = args.emb_trg_sharing
    feat_extractor = args.feat_extractor
    combine_decoder = args.combine_decoder
    
    if args.train_lang_list is not None:
        num_lang = len(args.train_lang_list)
    else:
        num_lang = 0

    encoder = Encoder(num_enc_layers, num_heads=num_heads, dim_model=dim_model, dim_key=dim_key,
                      dim_value=dim_value, dim_input=dim_input, dim_inner=dim_inner, src_max_length=src_max_len, dropout=dropout, num_lang=num_lang, is_factorized=is_factorized, r=r)
    decoder = Decoder(src_label2id, src_id2label, trg_label2ids, trg_id2labels, num_layers=num_dec_layers, num_heads=num_heads,
                      dim_emb=dim_emb, dim_model=dim_model, dim_inner=dim_inner, dim_key=dim_key, dim_value=dim_value, trg_max_length=tgt_max_len, dropout=dropout, emb_trg_sharing=emb_trg_sharing, num_lang=num_lang,
                      combine_decoder=combine_decoder, is_factorized=is_factorized, r=r)
    decoder = decoder if train else decoder
    model = Transformer(encoder, decoder, feat_extractor=feat_extractor, train=train)

    if args.parallel:
        device_ids = args.device_ids
        if constant.args.device_ids:
            print("load with device_ids", constant.args.device_ids)
            model = DataParallel(model, device_ids=constant.args.device_ids)
        else:
            model = DataParallel(model)

    return model

def init_deepspeech_model(args, label2id, id2label):
    """
    Initiate a new DeepSpeech object
    """
    hidden_size = int(math.floor(
        (args.sample_rate * args.window_size) / 2) + 1)
    hidden_size = int(math.floor(hidden_size - 41) / 2 + 1)
    hidden_size = int(math.floor(hidden_size - 21) / 2 + 1)
    hidden_size *= 32
    args.dim_input = hidden_size
    dim_input = hidden_size
    
    num_layers = args.num_layers
    dim_model = args.dim_model
    dim_input = args.dim_input

    model = DeepSpeech(dim_input, dim_model=dim_model, num_layers=num_layers, bidirectional=True, context=20, label2id=label2id, id2label=id2label)

    if args.parallel:
        device_ids = args.device_ids
        if constant.args.device_ids:
            print("load with device_ids", constant.args.device_ids)
            model = DataParallel(model, device_ids=constant.args.device_ids)
        else:
            model = DataParallel(model)

    return model

def init_lm_transformer_model(args, label2id, id2label):
    """
    Initiate a new transformer object
    """
    num_layers = args.num_layers
    num_heads = args.num_heads
    dim_model = args.dim_model
    dim_key = args.dim_key
    dim_value = args.dim_value
    dim_inner = args.dim_inner
    dim_emb = args.dim_emb
    tgt_max_len = args.tgt_max_len
    dropout = args.dropout

    model = TransformerLM(id2label, num_src_vocab=len(label2id), num_trg_vocab=len(label2id), num_layers=num_layers, dim_emb=dim_emb, dim_model=dim_model, dim_inner=dim_inner, num_heads=num_heads, dim_key=dim_key, dim_value=dim_value, trg_max_length=tgt_max_len, dropout=dropout)

    if args.parallel:
        if constant.args.device_ids:
            print("load with device_ids", constant.args.device_ids)
            model = DataParallel(model, device_ids=constant.args.device_ids)
        else:
            model = DataParallel(model)

    return model