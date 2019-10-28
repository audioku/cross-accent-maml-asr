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
from utils.functions import save_model, load_model
from utils.lstm_utils import LM

def compute_num_params(model):
    """
    Computes number of trainable and non-trainable parameters
    """
    sizes = [(np.array(p.data.size()).prod(), int(p.requires_grad)) for p in model.parameters()]
    return sum(map(lambda t: t[0]*t[1], sizes)), sum(map(lambda t: t[0]*(1 - t[1]), sizes))

def post_process(string, special_token_list):
    for i in range(len(special_token_list)):
        if special_token_list[i] != constant.PAD_TOKEN:
            string = string.replace(special_token_list[i],"")
    string = string.replace("▁"," ")
    return string

def evaluate(model, test_loader, lm=None, special_token_list=[], start_token=constant.SOS_TOKEN, lang_id=0):
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

            if constant.USE_CUDA:
                src = src.cuda()
                trg = trg.cuda()
                trg_transcript = trg_transcript.cuda()
                langs = langs.cuda()

            start_time = time.time()
            batch_ids_hyps, batch_strs_hyps, batch_strs_gold = model.evaluate(
                src, src_lengths, trg, trg_transcript, beam_search=constant.args.beam_search, beam_width=constant.args.beam_width, beam_nbest=constant.args.beam_nbest, lm=lm, lm_rescoring=constant.args.lm_rescoring, lm_weight=constant.args.lm_weight, c_weight=constant.args.c_weight, start_token=start_token, langs=langs, lang_names=lang_names, verbose=constant.args.verbose, lang_id=lang_id)

            for x in range(len(batch_strs_gold)):
                hyp = post_process(batch_strs_hyps[x],special_token_list)
                gold = post_process(batch_strs_gold[x],special_token_list)

                wer = calculate_wer(hyp, gold)
                cer = calculate_cer(hyp.strip(), gold.strip())

                if constant.args.verbose:
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
    args = constant.args

    start_iter = 0

    # Load the model
    load_path = constant.args.continue_from
    model, opt, epoch, metrics, loaded_args, src_label2id, src_id2label, trg_label2ids, trg_id2labels = load_model(constant.args.continue_from, train=False)
    
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
        test_lang_list = [constant.SOS_CHAR] * len(args.test_manifest_list)
    else:
        test_lang_list = ["<" + lang.upper() + ">" for lang in args.test_lang_list]

    train_lang_list = ["<" + lang.upper() + ">" for lang in train_lang_list]

    print("train_lang_list:", train_lang_list)
    print(test_lang_list)
    print(len(src_label2id))
    print("trg label2ids", len(trg_label2ids))

    special_token_list = [constant.PAD_CHAR, constant.SOS_CHAR, constant.EOS_CHAR] + test_lang_list
    print("INPUT TYPE: ", args.input_type)
    if loaded_args.feat == "spectrogram":
        if len(trg_label2ids) == 1:
            label2id = trg_label2ids[0]
        else:
            label2id = trg_label2ids[lang_id]
        test_data = SpectrogramDataset(audio_conf=audio_conf, lang_list=[test_lang_list[0]], all_lang_list=train_lang_list, manifest_filepath_list=[test_manifest_list[0]], src_label2id=src_label2id, trg_label2ids=[label2id], normalize=True, augment=False, input_type=args.input_type)
    elif loaded_args.feat == "logfbank":
        test_data = LogFBankDataset(audio_conf=audio_conf, lang_list=[test_lang_list[0]], all_lang_list=train_lang_list, manifest_filepath_list=[test_manifest_list[0]], label2id=label2id, normalize=True, augment=False, input_type=args.input_type)
    test_sampler = BucketingSampler(test_data, batch_size=constant.args.batch_size)
    test_loader = AudioDataLoader(test_data, num_workers=args.num_workers, batch_sampler=test_sampler)

    lm = None
    if constant.args.lm_rescoring:
        lm = LM(constant.args.lm_path)

    # print(model)

    print("Parameters: {}(trainable), {}(non-trainable)".format(
            compute_num_params(model)[0], compute_num_params(model)[1]))

    if not args.cuda:
        model = model.cpu()

    evaluate(model, test_loader, lm=lm ,special_token_list=special_token_list, start_token=src_label2id.get(test_lang_list[0]), lang_id=lang_id)
