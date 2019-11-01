import time
import numpy as np
from tqdm import tqdm
from utils import constant
from utils.functions import save_model
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer
from torch.autograd import Variable
import torch
import logging

import sys

class Trainer():
    """
    Trainer class
    """
    def __init__(self):
        logging.info("Trainer is initialized")

    def post_process(self, string, special_token_list):
        for i in range(len(special_token_list)):
            if special_token_list[i] != constant.PAD_TOKEN:
                string = string.replace(special_token_list[i],"")
        string = string.replace("▁"," ")
        return string

    def train_one_batch(self, model, src, trg, trg_transcript, src_percentages, src_lengths, trg_lengths, langs, lang_names, special_token_list, trg_id2labels, smoothing, loss_type):
        pred, gold, hyp = model(src, src_lengths, trg, trg_transcript, langs, lang_names, verbose=False)
        strs_golds, strs_hyps = [], []

        for lang_id in range(len(gold)):
            gold_seq = gold[lang_id]
            for j in range(len(gold_seq)):
                ut_gold = gold_seq[j]
                if len(trg_id2labels) == 1: lang_id = 0
                strs_golds.append("".join([trg_id2labels[lang_id][int(x)] for x in ut_gold]))
        
        for lang_id in range(len(hyp)):
            hyp_seq = hyp[lang_id]
            for j in range(len(hyp_seq)):
                ut_hyp = hyp_seq[j]
                if len(trg_id2labels) == 1: lang_id = 0
                strs_hyps.append("".join([trg_id2labels[lang_id][int(x)] for x in ut_hyp]))

        # handling the last batch
        for j in range(len(pred)):
            if len(pred[j]) > 0:
                seq_length = pred[j].size(1)
                break
        sizes = Variable(src_percentages.mul_(int(seq_length)).int(), requires_grad=False)

        loss = None
        for j in range(len(pred)):
            if len(pred[j]) == 0: continue
            t_loss, num_correct = calculate_metrics(
                pred[j], gold[j], input_lengths=sizes, target_lengths=trg_lengths, smoothing=smoothing, loss_type=loss_type)
            if loss is None:
                loss = t_loss
            else:
                loss = loss + t_loss
        if loss is None:
            print("loss is None")

        if loss.item() == float('Inf'):
            logging.info("Found infinity loss, masking")
            print("Found infinity loss, masking")
            loss = torch.where(loss != loss, torch.zeros_like(loss), loss) # NaN masking

        total_cer, total_wer, total_char, total_word = 0, 0, 0, 0
        for j in range(len(strs_hyps)):
            strs_hyps[j] = self.post_process(strs_hyps[j], special_token_list)
            strs_golds[j] = self.post_process(strs_golds[j], special_token_list)
            cer = calculate_cer(strs_hyps[j].replace(' ', ''), strs_golds[j].replace(' ', ''))
            wer = calculate_wer(strs_hyps[j], strs_golds[j])
            total_cer += cer
            total_wer += wer
            total_char += len(strs_golds[j].replace(' ', ''))
            total_word += len(strs_golds[j].split(" "))

        return loss, total_cer, total_char

    def train(self, model, vocab, train_loader, train_sampler, valid_loader_list, opt, loss_type, start_epoch, num_epochs, last_metrics=None, early_stop=10, is_accu_loss=False):
        """
        Training
        args:
            model: Model object
            train_loader: DataLoader object of the training set
            valid_loader_list: a list of Validation DataLoader objects
            opt: Optimizer object
            start_epoch: start epoch (> 0 if you resume the process)
            num_epochs: last epoch
            last_metrics: (if resume)
        """
        history = []
        best_valid_val = 1000000000
        smoothing = constant.args.label_smoothing
        early_stop_criteria, early_stop_val = early_stop.split(",")[0], int(early_stop.split(",")[1])
        count_stop = 0

        logging.info("name " +  constant.args.name)

        for epoch in range(start_epoch, num_epochs):
            total_loss, total_cer, total_wer, total_char, total_word = 0, 0, 0, 0, 0
            total_time = 0

            start_iter = 0
            final_train_losses = []
            final_train_cers = []

            logging.info("TRAIN")
            print("TRAIN")
            model.train()
            pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
            max_len = 0
            for i, (data) in enumerate(pbar, start=start_iter):
                torch.cuda.empty_cache()
                src, trg, trg_transcript, src_percentages, src_lengths, trg_lengths, langs, lang_names = data
                max_len = max(max_len, src.size(-1))

                opt.zero_grad()

                try:
                    if constant.USE_CUDA:
                        src = src.cuda()
                        trg = trg.cuda()
                        trg_transcript = trg_transcript.cuda()
                        langs = langs.cuda()

                    start_time = time.time()
                    loss, cer, num_char = self.train_one_batch(model, src, trg, trg_transcript, src_percentages, src_lengths, trg_lengths, langs, lang_names, special_token_list, trg_id2labels, smoothing, loss_type)
                    total_cer += cer
                    total_char += num_char
                    loss.backward()

                    if constant.args.clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), constant.args.max_norm)
                    
                    opt.step()
                    total_loss += loss.item()

                    end_time = time.time()
                    diff_time = end_time - start_time
                    total_time += diff_time

                    pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} CER:{:.2f}% LR:{:.7f} TOTAL TIME:{:.7f}".format(
                        (epoch+1), total_loss/(i+1), total_cer*100/total_char, opt._rate, total_time))
                except:
                    # del loss
                    try:
                        torch.cuda.empty_cache()
                        src = src.cpu()
                        trg = trg.cpu()
                        src_splits, src_lengths_splits, trg_lengths_splits, trg_splits, trg_transcript_splits, src_percentages_splits, langs_splits = iter(src.split(2, dim=0)), iter(src_lengths.split(2, dim=0)), iter(trg_lengths.split(2, dim=0)), iter(trg.split(2, dim=0)), iter(trg_transcript.split(2, dim=0)), iter(src_percentages.split(2, dim=0)), iter(langs.split(2, dim=0))
                        j = 0

                        start_time = time.time()
                        for src, trg, src_lengths, trg_lengths, trg_transcript, src_percentages, langs in zip(src_splits, trg_splits, src_lengths_splits, trg_lengths_splits, trg_transcript_splits, src_percentages_splits, langs_splits):
                            opt.zero_grad()
                            torch.cuda.empty_cache()
                            if constant.USE_CUDA:
                                src = src.cuda()
                                trg = trg.cuda()

                            start_time = time.time()
                            loss, cer, num_char = self.train_one_batch(model, src, trg, trg_transcript, src_percentages, src_lengths, trg_lengths, langs, lang_names[j*2:j*2+2], special_token_list, trg_id2labels, smoothing, loss_type)
                            total_cer += cer
                            total_char += num_char
                            loss.backward()

                            if constant.args.clip:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), constant.args.max_norm)
                            
                            opt.step()
                            total_loss += loss.item()
                            j += 1

                        end_time = time.time()
                        diff_time = end_time - start_time
                        total_time += diff_time
                        logging.info("probably OOM, autosplit batch. succeeded")
                        print("probably OOM, autosplit batch. succeeded")
                    except:
                        logging.info("probably OOM, autosplit batch. skip batch")
                        print("probably OOM, autosplit batch. skip batch")
                        continue

            pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} CER:{:.2f}% LR:{:.7f} TOTAL TIME:{:.7f}".format((epoch+1), total_loss/(i+1), total_cer*100/total_char, opt._rate, total_time))

            final_train_loss = total_loss/(len(train_loader))
            final_train_cer = total_cer*100/total_char

            final_train_losses.append(final_train_loss)
            final_train_cers.append(final_train_cer)

            logging.info("(Epoch {}) TRAIN LOSS:{:.4f} CER:{:.2f}% LR:{:.7f}".format(
                (epoch+1), final_train_loss, final_train_cer, opt._rate))

            # evaluate
            print("")
            logging.info("VALID")
            model.eval()

            final_valid_losses = []
            final_valid_cers = []
            for ind in range(len(valid_loader_list)):
                valid_loader = valid_loader_list[ind]

                total_valid_loss, total_valid_cer, total_valid_wer, total_valid_char, total_valid_word = 0, 0, 0, 0, 0
                valid_pbar = tqdm(iter(valid_loader), leave=True, total=len(valid_loader))
                for i, (data) in enumerate(valid_pbar):
                    torch.cuda.empty_cache()

                    src, trg, trg_transcript, src_percentages, src_lengths, trg_lengths, langs, lang_names = data
                    try:
                        if constant.USE_CUDA:
                            src = src.cuda()
                            trg = trg.cuda()
                            trg_transcript = trg_transcript.cuda()
                            langs = langs.cuda()
                        loss, cer, num_char = self.train_one_batch(model, src, trg, trg_transcript, src_percentages, src_lengths, trg_lengths, langs, lang_names, special_token_list, trg_id2labels, smoothing, loss_type)
                        total_valid_cer += cer
                        total_valid_char += num_char

                        total_valid_loss += loss.item()
                        valid_pbar.set_description("VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format(ind,
                            total_valid_loss/(i+1), total_valid_cer*100/total_valid_char))
                        # valid_pbar.set_description("(Epoch {}) VALID LOSS:{:.4f} CER:{:.2f}% WER:{:.2f}%".format(
                            # (epoch+1), total_valid_loss/(i+1), total_valid_cer*100/total_valid_char, total_valid_wer*100/total_valid_word))
                    except:
                        try:
                            torch.cuda.empty_cache()
                            src = src.cpu()
                            trg = trg.cpu()
                            src_splits, src_lengths_splits, trg_lengths_splits, trg_splits, trg_transcript_splits, src_percentages_splits, langs_splits = iter(src.split(2, dim=0)), iter(src_lengths.split(2, dim=0)), iter(trg_lengths.split(2, dim=0)), iter(trg.split(2, dim=0)), iter(trg_transcript.split(2, dim=0)), iter(src_percentages.split(2, dim=0)), iter(langs.split(2, dim=0))
                            j = 0
                            for src, trg, src_lengths, trg_lengths, trg_transcript, src_percentages, langs in zip(src_splits, trg_splits, src_lengths_splits, trg_lengths_splits, trg_transcript_splits, src_percentages_splits, langs_splits):
                                opt.zero_grad()
                                torch.cuda.empty_cache()
                                if constant.USE_CUDA:
                                    src = src.cuda()
                                    trg = trg.cuda()

                                loss, cer, num_char = self.train_one_batch(model, src, trg, trg_transcript, src_percentages, src_lengths, trg_lengths, langs, lang_names[j*2:j*2+2], special_token_list, trg_id2labels, smoothing, loss_type)
                                total_valid_cer += cer
                                total_valid_char += num_char

                                if constant.args.clip:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), constant.args.max_norm)
                                
                                total_valid_loss += loss.item()
                                j += 1
                            valid_pbar.set_description("VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format(ind, total_valid_loss/(i+1), total_valid_cer*100/total_valid_char))

                            logging.info("probably OOM, autosplit batch. succeeded")
                            print("probably OOM, autosplit batch. succeeded")
                        except:
                            logging.info("probably OOM, autosplit batch. skip batch")
                            print("probably OOM, autosplit batch. skip batch")
                            continue

                final_valid_loss = total_valid_loss/(len(valid_loader))
                final_valid_cer = total_valid_cer*100/total_valid_char

                final_valid_losses.append(final_valid_loss)
                final_valid_cers.append(final_valid_cer)
                print("VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format(ind, final_valid_loss, final_valid_cer))
                logging.info("VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format(ind, final_valid_loss, final_valid_cer))

            metrics = {}
            avg_valid_loss = sum(final_valid_losses) / len(final_valid_losses)
            avg_valid_cer = sum(final_valid_cers) / len(final_valid_cers)
            metrics["avg_train_loss"] = sum(final_train_losses) / len(final_train_losses)
            metrics["avg_valid_loss"] = sum(final_valid_losses) / len(final_valid_losses)
            metrics["avg_train_cer"] = sum(final_train_cers) / len(final_train_cers)
            metrics["avg_valid_cer"] = sum(final_valid_cers) / len(final_valid_cers)
            metrics["train_loss"] = final_train_losses
            metrics["valid_loss"] = final_valid_losses
            metrics["train_cer"] = final_train_cers
            metrics["valid_cer"] = final_valid_cers
            metrics["history"] = history
            history.append(metrics)

            print("AVG VALID LOSS:{:.4f} AVG CER:{:.2f}%".format(sum(final_valid_losses) / len(final_valid_losses), sum(final_valid_cers) / len(final_valid_cers)))
            logging.info("AVG VALID LOSS:{:.4f} AVG CER:{:.2f}%".format(sum(final_valid_losses) / len(final_valid_losses), sum(final_valid_cers) / len(final_valid_cers)))

            if epoch % constant.args.save_every == 0:
                save_model(model, vocab, (epoch+1), opt, metrics, best_model=False)

            # save the best model
            early_stop_criteria, early_stop_val
            if early_stop_criteria == "cer":
                print("CRITERIA: CER")
                if best_valid_val > avg_valid_cer:
                    count_stop = 0
                    best_valid_val = avg_valid_cer
                    save_model(model, vocab, (epoch+1), opt, metrics, best_model=True)
                else:
                    print("count_stop:", count_stop)
                    count_stop += 1
            else:
                print("CRITERIA: LOSS")
                if best_valid_val > avg_valid_loss:
                    count_stop = 0
                    best_valid_val = avg_valid_loss
                    save_model(model, vocab, (epoch+1), opt, metrics, best_model=True)
                else:
                    count_stop += 1
                    print("count_stop:", count_stop)

            if count_stop >= early_stop_val:
                logging.info("EARLY STOP")
                print("EARLY STOP\n")
                break

            if constant.args.shuffle:
                logging.info("SHUFFLE")
                print("SHUFFLE\n")
                train_sampler.shuffle(epoch)