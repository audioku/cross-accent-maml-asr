import json
import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from tqdm import tqdm
from models.asr.transformer import Transformer, Encoder, Decoder
from utils import constant
from utils.data_loader import SpectrogramDataset, LogFBankDataset, AudioDataLoader, BucketingSampler
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer, calculate_cer_en_zh
from utils.functions import save_model, load_model, post_process

parser = argparse.ArgumentParser(description='Transformer ASR training')
parser.add_argument('--model', default='TRFS', type=str, help="")
parser.add_argument('--name', default='model', help="Name of the model for saving")

parser.add_argument('--train-manifest-list', nargs='+', type=str)
parser.add_argument('--valid-manifest-list', nargs='+', type=str)
parser.add_argument('--test-manifest-list', nargs='+', type=str)

parser.add_argument('--sample-rate', default=22050, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')

parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--label-smoothing', default=0.0, type=float, help='Label smoothing')

parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')

parser.add_argument('--epochs', default=1000, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')

parser.add_argument('--device-ids', default=None, nargs='+', type=int,
                    help='If using cuda, sets the GPU devices for the process')
# parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
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

# Noam optimizer
parser.add_argument('--warmup', default=4000, type=int, help='Warmup')
parser.add_argument('--min-lr', default=1e-5, type=float, help='min lr')
parser.add_argument('--k-lr', default=1, type=float, help='factor lr')

# SGD optimizer
parser.add_argument('--lr', default=1e-4, type=float, help='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--lr-anneal', default=1.1, type=float, help='lr anneal')

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
parser.add_argument('--is-accu-loss', action='store_true', help="is accu loss. experimental")
parser.add_argument('--is-factorized', action='store_true', help="is factorized. experimental")
parser.add_argument('--r', default=100, type=int, help='rank')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')

# shuffle
parser.add_argument('--shuffle', action='store_true', help='Shuffle')

# input
parser.add_argument('--input_type', type=str, default='char', help='char or bpe or ipa')

# Post-training factorization
parser.add_argument('--rank', default=10, type=float, help="rank")
parser.add_argument('--factorize', action='store_true', help='factorize')

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

args = parser.parse_args()
USE_CUDA = args.cuda

def evaluate(model, test_loader, lm=None, special_token_list=[], start_token=SOS_TOKEN, lang_id=0):
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
        test_pbar = tqdm(iter(test_loader), leave=True, total=len(test_loader))
        for i, (data) in enumerate(test_pbar):
            src, trg, trg_transcript, src_percentages, src_lengths, trg_lengths, langs, lang_names = data

            if USE_CUDA:
                src = src.cuda()
                trg = trg.cuda()
                trg_transcript = trg_transcript.cuda()
                langs = langs.cuda()

            start_time = time.time()
            batch_ids_hyps, batch_strs_hyps, batch_strs_gold = model.evaluate(
                src, src_lengths, trg, trg_transcript, beam_search=args.beam_search, beam_width=args.beam_width, beam_nbest=args.beam_nbest, lm=lm, lm_rescoring=args.lm_rescoring, lm_weight=args.lm_weight, c_weight=args.c_weight, start_token=start_token, langs=langs, lang_names=lang_names, verbose=args.verbose, lang_id=lang_id)

            for x in range(len(batch_strs_gold)):
                hyp = post_process(batch_strs_hyps[x],special_token_list)
                gold = post_process(batch_strs_gold[x],special_token_list)

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


if __name__ == '__main__':
    start_iter = 0

    # Load the model
    load_path = args.continue_from
    model, vocab, opt, epoch, metrics, loaded_args = load_model(args.continue_from, train=False)
    
    print("EPOCH:", epoch)
    if loaded_args.parallel:
        print("unwrap data parallel")
        model = model.module

    audio_conf = dict(sample_rate=loaded_args.sample_rate,
                      window_size=loaded_args.window_size,
                      window_stride=loaded_args.window_stride,
                      window=loaded_args.window,
                      noise_dir=loaded_args.noise_dir,
                      noise_prob=loaded_args.noise_prob,
                      noise_levels=(loaded_args.noise_min, loaded_args.noise_max))

    train_lang_list = loaded_args.train_lang_list
    test_manifest_list = args.test_manifest_list
    test_lang_list = args.test_lang_list

    lang_id = 10000
    for i in range(len(train_lang_list)):
        if test_lang_list[0] == train_lang_list[i]:
            lang_id = i
            print("language id detected:", lang_id)
            break
    
    test_lang_list = [] if test_lang_list is None else test_lang_list
    if len(test_lang_list) == 0:
        test_lang_list = [SOS_CHAR] * len(args.test_manifest_list)
    else:
        test_lang_list = ["<" + lang.upper() + ">" for lang in args.test_lang_list]

    train_lang_list = ["<" + lang.upper() + ">" for lang in train_lang_list]

    print("train_lang_list:", train_lang_list)
    print(test_lang_list)
    print(len(src_label2id))
    print("trg label2ids", len(trg_label2ids))

    special_token_list = [PAD_CHAR, SOS_CHAR, EOS_CHAR] + test_lang_list
    print("INPUT TYPE: ", args.input_type)
    if loaded_args.feat == "spectrogram":
        if len(trg_label2ids) == 1:
            label2id = trg_label2ids[0]
        else:
            label2id = trg_label2ids[lang_id]
        test_data = SpectrogramDataset(audio_conf=audio_conf, lang_list=[test_lang_list[0]], all_lang_list=train_lang_list, manifest_filepath_list=[test_manifest_list[0]], src_label2id=src_label2id, trg_label2ids=[label2id], normalize=True, augment=False, input_type=args.input_type)
    elif loaded_args.feat == "logfbank":
        test_data = LogFBankDataset(audio_conf=audio_conf, lang_list=[test_lang_list[0]], all_lang_list=train_lang_list, manifest_filepath_list=[test_manifest_list[0]], label2id=label2id, normalize=True, augment=False, input_type=args.input_type)
    test_sampler = BucketingSampler(test_data, batch_size=args.batch_size)
    test_loader = AudioDataLoader(test_data, num_workers=args.num_workers, batch_sampler=test_sampler)

    lm = None
    if args.lm_rescoring:
        lm = LM(args.lm_path)

    # print(model)

    print("Parameters: {}(trainable), {}(non-trainable)".format(
            compute_num_params(model)[0], compute_num_params(model)[1]))

    if not args.cuda:
        model = model.cpu()

    evaluate(model, test_loader, lm=lm ,special_token_list=special_token_list, start_token=src_label2id.get(test_lang_list[0]), lang_id=lang_id)
