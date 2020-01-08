import argparse
import json
import time
import math
import logging
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import random

from torchsummary import summary
from trainer.asr.trainer import Trainer
from utils.data import Vocab
from utils.functions import load_meta_model, load_joint_model, init_transformer_model, post_process, init_optimizer, compute_num_params, generate_labels

from torch.autograd import Variable
from tqdm import tqdm
from models.asr.transformer import Transformer
from modules.encoder import Encoder
from modules.decoder import Decoder
from utils.data_loader import SpectrogramDataset, LogFBankDataset, AudioDataLoader, BucketingSampler
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer, calculate_cer_en_zh
from utils.lm import LM

parser = argparse.ArgumentParser(description='Transformer ASR meta training')
parser.add_argument('--model', default='TRFS', type=str, help="")
parser.add_argument('--name', default='model', help="Name of the model for saving")

parser.add_argument('--train-manifest-list', nargs='+', type=str)
parser.add_argument('--valid-manifest-list', nargs='+', type=str)
parser.add_argument('--test-manifest-list', nargs='+', type=str)

parser.add_argument('--train-partition-list', nargs='+', type=float, default=None)
parser.add_argument('--training-mode', default='meta')

parser.add_argument('--sample-rate', default=22050, type=int, help='Sample rate')
parser.add_argument('--k-train', default=20, type=int, help='Batch size for training')
parser.add_argument('--k-test', default=1, type=int, help='Batch size for training')

parser.add_argument('--num-workers', default=8, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--label-smoothing', default=0.0, type=float, help='Label smoothing')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')

parser.add_argument('--early-stop', default="loss,10", type=str, help='Early stop (loss,10) or (cer,10)')
parser.add_argument('--save-every', default=5, type=int, help='Save model every certain number of epochs')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--emb-trg-sharing', action='store_true', help='Share embedding weight source and target')
parser.add_argument('--feat_extractor', default='vgg_cnn', type=str, help='emb_cnn or vgg_cnn or none')
parser.add_argument('--feat', type=str, default='spectrogram', help='spectrogram or logfbank')
parser.add_argument('--verbose', action='store_true', help='Verbose')

parser.add_argument('--continue-from', default='', type=str, help='Continue from checkpoint model')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)

# Transformer
parser.add_argument('--num-enc-layers', default=3, type=int, help='Number of layers')
parser.add_argument('--num-dec-layers', default=3, type=int, help='Number of layers')
parser.add_argument('--num-heads', default=5, type=int, help='Number of heads')
parser.add_argument('--dim-model', default=512, type=int, help='Model dimension')
parser.add_argument('--dim-key', default=64, type=int, help='Key dimension')
parser.add_argument('--dim-value', default=64, type=int, help='Value dimension')
parser.add_argument('--dim-input', default=161, type=int, help='Input dimension')
parser.add_argument('--dim-inner', default=1024, type=int, help='Inner dimension')
parser.add_argument('--dim-emb', default=512, type=int, help='Embedding dimension')

parser.add_argument('--src-max-len', default=2500, type=int, help='Source max length')
parser.add_argument('--tgt-max-len', default=1000, type=int, help='Target max length')

# optimizer
parser.add_argument('--lr', default=1e-4, type=float, help='lr')
parser.add_argument('--evaluate-every', default=10, type=int, help='evaluate every')

# Noam optimizer
parser.add_argument('--warmup', default=4000, type=int, help='Warmup')
parser.add_argument('--min-lr', default=1e-5, type=float, help='min lr')
parser.add_argument('--k-lr', default=1, type=float, help='factor lr')

# Decoder search
parser.add_argument('--beam-search', action='store_true', help='Beam search')
parser.add_argument('--beam-width', default=3, type=int, help='Beam size')
parser.add_argument('--beam-nbest', default=5, type=int, help='Number of best sequences')
parser.add_argument('--lm-rescoring', action='store_true', help='Rescore using LM')
parser.add_argument('--lm-path', type=str, default="lm_model.pt", help="Path to LM model")
parser.add_argument('--lm-weight', default=0.1, type=float, help='LM weight')
parser.add_argument('--c-weight', default=0.1, type=float, help='Word count weight')
parser.add_argument('--prob-weight', default=1.0, type=float, help='Probability E2E weight')

# loss
parser.add_argument('--loss', type=str, default='ce', help='ce or ctc')
parser.add_argument('--clip', action='store_true', help="clip")
parser.add_argument('--max-norm', default=400, type=float, help="max norm for clipping")
parser.add_argument('--is-factorized', action='store_true', help="is factorized. experimental")
parser.add_argument('--r', default=100, type=int, help='rank')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')

# input
parser.add_argument('--input_type', type=str, default='char', help='char or bpe or ipa')

# Post-training factorization
parser.add_argument('--rank', default=10, type=float, help="rank")
parser.add_argument('--factorize', action='store_true', help='factorize')

# Training config
parser.add_argument('--copy-grad', action='store_true', help="copy grad for MAML") # Useless
parser.add_argument('--cpu-state-dict', action='store_true', help='store state dict in cpu')
parser.add_argument('--opt_name', type=str, default='adam', help='adam or sgd')

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)
np.random.seed(123456)
random.seed(123456)
args = parser.parse_args()
USE_CUDA = args.cuda

def evaluate(model, vocab, test_loader, args, lm=None, start_token=-1):
    """
    Evaluation
    args:
        model: Model object
        test_loader: DataLoader object
    """
    model.eval()

    total_word, total_char, total_cer, total_wer = 0, 0, 0, 0
    total_en_cer, total_zh_cer, total_en_char, total_zh_char = 0, 0, 0, 0
    total_hyp_char = 0
    total_time = 0

    with torch.no_grad():
        test_pbar = tqdm(iter(test_loader), leave=False, total=len(test_loader))
        for i, (data) in enumerate(test_pbar):
            src, trg, src_percentages, src_lengths, trg_lengths = data

            if USE_CUDA:
                src = src.cuda()
                trg = trg.cuda()

            start_time = time.time()
            batch_ids_hyps, batch_strs_hyps, batch_strs_gold = model.evaluate(
                src, src_lengths, trg, args, lm_rescoring=args.lm_rescoring, lm=lm, lm_weight=args.lm_weight, beam_search=args.beam_search, beam_width=args.beam_width, beam_nbest=args.beam_nbest, c_weight=args.c_weight, start_token=start_token, verbose=args.verbose)

            for x in range(len(batch_strs_gold)):
                hyp = post_process(batch_strs_hyps[x], vocab.special_token_list)
                gold = post_process(batch_strs_gold[x], vocab.special_token_list)

                wer = calculate_wer(hyp, gold)
                cer = calculate_cer(hyp.strip(), gold.strip())

                if args.verbose:
                    print("HYP",hyp)
                    print("GOLD:",gold)
                    print("CER:",cer)

                en_cer, zh_cer, num_en_char, num_zh_char = calculate_cer_en_zh(hyp, gold)
                total_en_cer += en_cer
                total_zh_cer += zh_cer
                total_en_char += num_en_char
                total_zh_char += num_zh_char
                total_hyp_char += len(hyp)

                total_wer += wer
                total_cer += cer
                total_word += len(gold.split(" "))
                total_char += len(gold)

            end_time = time.time()
            diff_time = end_time - start_time
            total_time += diff_time
            diff_time_per_word = total_time / total_word

            test_pbar.set_description("TEST CER:{:.2f}% WER:{:.2f}% CER_EN:{:.2f}% CER_ZH:{:.2f}% TOTAL_TIME:{:.7f} TOTAL HYP CHAR:{:.2f}".format(
                total_cer*100/total_char, total_wer*100/total_word, total_en_cer*100/max(1, total_en_char), total_zh_cer*100/max(1, total_zh_char), total_time, total_hyp_char))
            
            print("TEST CER:{:.2f}% WER:{:.2f}% CER_EN:{:.2f}% CER_ZH:{:.2f}% TOTAL_TIME:{:.7f} TOTAL HYP CHAR:{:.2f}".format(
                total_cer*100/total_char, total_wer*100/total_word, total_en_cer*100/max(1, total_en_char), total_zh_cer*100/max(1, total_zh_char), total_time, total_hyp_char), flush=True)
            logging.info("{}/{} TEST CER:{:.2f}% WER:{:.2f}% CER_EN:{:.2f}% CER_ZH:{:.2f}% TOTAL_TIME:{:.7f} TOTAL HYP CHAR:{:.2f}".format(
                i, len(test_loader), total_cer*100/total_char, total_wer*100/total_word, total_en_cer*100/max(1, total_en_char), total_zh_cer*100/max(1, total_zh_char), total_time, total_hyp_char))
    return total_cer*100/total_char, total_wer*100/total_word

if __name__ == '__main__':
    print("="*50)
    print("THE EXPERIMENT LOG IS SAVED IN: " + "log/" + args.name)
    print("TRAINING MANIFEST: ", args.train_manifest_list)
    print("VALID MANIFEST: ", args.valid_manifest_list)
    print("TEST MANIFEST: ", args.test_manifest_list)
    print("INPUT TYPE: ", args.input_type)
    print("OPT NAME: ", args.opt_name)
    print("="*50)

    if not os.path.exists("./log"): os.mkdir("./log")
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        
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
    
    with open(args.labels_path, encoding="utf-8") as label_file:
        labels = json.load(label_file)

    vocab = Vocab()
    for label in labels:
        vocab.add_token(label)
        vocab.add_label(label)

    if args.feat == "spectrogram":
        train_data = SpectrogramDataset(vocab, args, audio_conf, manifest_filepath_list=[args.train_manifest_list[0]], normalize=True, augment=args.augment, input_type=args.input_type, is_train=True, partitions=args.train_partition_list)
    elif args.feat == "logfbank":
        train_data = LogFBankDataset(vocab, args, audio_conf, manifest_filepath_list=[args.train_manifest_list[0]], normalize=True, augment=args.augment, input_type=args.input_type, is_train=True)
    train_loader = AudioDataLoader(pad_token_id=vocab.PAD_ID, dataset=train_data, num_workers=args.num_workers)

    valid_loader_list = []
    for i in range(len(args.valid_manifest_list)):
        if args.feat == "spectrogram":
            valid_data = SpectrogramDataset(vocab, args, audio_conf, manifest_filepath_list=[args.valid_manifest_list[i]], normalize=True, augment=args.augment, input_type=args.input_type)
        elif args.feat == "logfbank":
            valid_data = LogFBankDataset(vocab, args, audio_conf, manifest_filepath_list=[args.valid_manifest_list[i]], normalize=True, augment=False, input_type=args.input_type)
        valid_loader = AudioDataLoader(pad_token_id=vocab.PAD_ID, dataset=valid_data, num_workers=args.num_workers)
        valid_loader_list.append(valid_loader)

    start_epoch = 0
    metrics = None
    loaded_args = None
    logging.info("Continue from checkpoint:" + args.continue_from)
    if args.training_mode == "meta":
        model, vocab, _, _, epoch, metrics, loaded_args = load_meta_model(args.continue_from)
    else:
        model, vocab, _, epoch, metrics, loaded_args = load_joint_model(args.continue_from)
    verbose = args.verbose
    
    loss_type = args.loss

    if USE_CUDA:
        model = model.cuda()

    logging.info(model)
    num_epochs = args.epochs

    print("Parameters: {}(trainable), {}(non-trainable)".format(compute_num_params(model)[0], compute_num_params(model)[1]))

    trainer = Trainer()
    trainer.train(model, vocab, train_loader, valid_loader_list, loss_type, start_epoch, num_epochs, args, last_metrics=metrics, evaluate_every=args.evaluate_every, early_stop=args.early_stop, opt_name=args.opt_name)

    # test
    logging.info("Test")
    print("Test")
    test_manifest_list = args.test_manifest_list
    args.tgt_max_len = 150

    cer_list, wer_list = [], []
    for i in range(len(test_manifest_list)):
        if loaded_args.feat == "spectrogram":
            test_data = SpectrogramDataset(vocab, args, audio_conf=audio_conf, manifest_filepath_list=[test_manifest_list[i]], normalize=True, augment=False, input_type=args.input_type)
        elif loaded_args.feat == "logfbank":
            test_data = LogFBankDataset(vocab, args, audio_conf=audio_conf, manifest_filepath_list=[test_manifest_list[i]], normalize=True, augment=False, input_type=args.input_type)
        test_sampler = BucketingSampler(test_data, batch_size=args.k_test)
        test_loader = AudioDataLoader(vocab.PAD_ID, dataset=test_data, num_workers=args.num_workers, batch_sampler=test_sampler)

        lm = None
        if args.lm_rescoring:
            lm = LM(args.lm_path, args)

        cer, wer = evaluate(model, vocab, test_loader, args, lm=lm, start_token=vocab.SOS_ID)
        cer_list.append(cer)
        wer_list.append(wer)

    print("="*50)
    logging.info("="*50)
    for index, (cer, wer) in enumerate(zip(cer_list, wer_list)):
        print("TEST DATASET [{}] CER:{:.4f} WER:{:.4f}".format(index, cer, wer))
        logging.info("TEST DATASET [{}] CER:{:.4f} WER:{:.4f}".format(index, cer, wer))
    print("="*50)
    logging.info("="*50)