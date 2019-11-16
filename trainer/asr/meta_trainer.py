import time
import numpy as np
import torch
import logging
import sys

from copy import deepcopy
from tqdm import tqdm
# from utils import constant
from collections import deque
from utils.functions import save_meta_model, post_process
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer
from torch.autograd import Variable

class MetaTrainer():
    """
    Trainer class
    """
    def __init__(self):
        logging.info("Trainer is initialized")

    def forward_one_batch(self, model, vocab, src, trg, src_percentages, src_lengths, trg_lengths, smoothing, loss_type, verbose=False):
        pred, gold, hyp = model(src, src_lengths, trg, verbose=False)
        strs_golds, strs_hyps = [], []

        for j in range(len(gold)):
            ut_gold = gold[j]
            strs_golds.append("".join([vocab.id2label[int(x)] for x in ut_gold]))
        
        for j in range(len(hyp)):
            ut_hyp = hyp[j]
            strs_hyps.append("".join([vocab.id2label[int(x)] for x in ut_hyp]))

        # handling the last batch
        seq_length = pred.size(1)
        sizes = src_percentages.mul_(int(seq_length)).int()

        loss, num_correct = calculate_metrics(pred, gold, vocab.PAD_ID, input_lengths=sizes, target_lengths=trg_lengths, smoothing=smoothing, loss_type=loss_type)

        if loss is None:
            print("loss is None")

        if loss.item() == float('Inf'):
            logging.info("Found infinity loss, masking")
            print("Found infinity loss, masking")
            loss = torch.where(loss != loss, torch.zeros_like(loss), loss) # NaN masking

        # if verbose:
        #     print(">PRED:", strs_hyps)
        #     print(">GOLD:", strs_golds)

        total_cer, total_wer, total_char, total_word = 0, 0, 0, 0
        for j in range(len(strs_hyps)):
            strs_hyps[j] = post_process(strs_hyps[j], vocab.special_token_list)
            strs_golds[j] = post_process(strs_golds[j], vocab.special_token_list)
            cer = calculate_cer(strs_hyps[j].replace(' ', ''), strs_golds[j].replace(' ', ''))
            wer = calculate_wer(strs_hyps[j], strs_golds[j])
            total_cer += cer
            total_wer += wer
            total_char += len(strs_golds[j].replace(' ', ''))
            total_word += len(strs_golds[j].split(" "))
        
        if verbose:
            print('Total CER', total_cer)
            print('Total char', total_char)

            print("PRED:", strs_hyps)
            print("GOLD:", strs_golds, flush=True)

        return loss, total_cer, total_char

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, model, vocab, train_data_list, valid_loader_list, loss_type, start_it, num_it, args, evaluate_every=1000, window_size=100, last_summary_every=10, last_metrics=None, early_stop=10):
        """
        Training
        args:
            model: Model object
            train_data_list: DataLoader object of the training set
            valid_loader_list: a list of Validation DataLoader objects
            start_it: start it (> 0 if you resume the process)
            num_it: last epoch
            last_metrics: (if resume)
        """
        history = []
        best_valid_val = 1000000000
        smoothing = args.label_smoothing
        early_stop_criteria, early_stop_val = early_stop.split(",")[0], int(early_stop.split(",")[1])
        count_stop = 0

        logging.info("name " +  args.name)

        total_time = 0
        start_iter = 0

        logging.info("TRAIN")
        print("TRAIN")
        model.train()

        # define the optimizer
        inner_opt = torch.optim.SGD(model.parameters(), lr=args.lr)
        outer_opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr)

        last_sum_loss = deque(maxlen=window_size)
        last_sum_cer = deque(maxlen=window_size)
        last_sum_char = deque(maxlen=window_size)
        
        for it in range(start_it, num_it):
            try:
                weights_original = deepcopy(model.state_dict())
                weights_inner = deepcopy(model.state_dict())
                
                # accumulate loss
                batch_loss = 0
                total_loss, total_cer = 0, 0
                total_char = 0

                # print("before:", model.encoder.input_linear.weight)
                for manifest_id in range(len(train_data_list)):
                    # torch.cuda.empty_cache()

                    k_train, k_valid = args.k_train, args.k_valid
                    torch.cuda.empty_cache()
                    tr_data, val_data = train_data_list[manifest_id].sample(k_train, k_valid, manifest_id)
                    tr_inputs, tr_input_sizes, tr_percentages, tr_targets, tr_target_sizes = tr_data
                    val_inputs, val_input_sizes, val_percentages, val_targets, val_target_sizes = val_data
                    
                    if args.cuda:
                        tr_inputs = tr_inputs.cuda()
                        tr_input_sizes = tr_input_sizes.cuda()
                        tr_targets = tr_targets.cuda()
                        tr_target_sizes = tr_target_sizes.cuda()

                        val_inputs = val_inputs.cuda()
                        val_input_sizes = val_input_sizes.cuda()
                        val_targets = val_targets.cuda()
                        val_target_sizes = val_target_sizes.cuda()
                    
                    start_time = time.time()
                
                    # Before first update
                    inner_opt.zero_grad()
    #                 print('META TRAIN')
                    model.train()
                    tr_loss, tr_cer, tr_num_char = self.forward_one_batch(model, vocab, tr_inputs, tr_targets, tr_percentages, tr_input_sizes, tr_target_sizes, smoothing, loss_type, verbose=False)
    #                 print('META VALID')

                    model.eval()
                    with torch.no_grad():
                        val_loss, val_cer, val_num_char = self.forward_one_batch(model, vocab, val_inputs, val_targets, val_percentages, val_input_sizes, val_target_sizes, smoothing, loss_type, verbose=False)

                        # Update train evaluation metric                    
                        total_cer += val_cer
                        total_char += val_num_char

                    tr_loss.backward()
                    inner_opt.step()

                    if args.clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

                    # After first update
                    model.eval()
                    val_loss, val_cer, val_num_char = self.forward_one_batch(model, vocab, val_inputs, val_targets, val_percentages, val_input_sizes, val_target_sizes, smoothing, loss_type)
                    
                    # batch_loss += val_loss
                    total_loss += val_loss.item()
                    
                    end_time = time.time()
                    diff_time = end_time - start_time
                    total_time += diff_time

                    # outer loop optimization
                    outer_opt.zero_grad()
                    
                    model.load_state_dict(weights_inner)
                    batch_loss = val_loss / len(train_data_list)
                    batch_loss.backward()
                    if args.clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                    outer_opt.step()
                    weights_inner = deepcopy(model.state_dict())

                    # reset
                    model.load_state_dict(weights_original)
                    # print("inner:", model.encoder.input_linear.weight)
                    
                # outer loop optimization
                # outer_opt.zero_grad()
                # batch_loss /= len(train_data_list)
                # batch_loss.backward()
                # if args.clip:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                # outer_opt.step()
                
                last_sum_cer.append(total_cer)
                last_sum_char.append(total_char)
                last_sum_loss.append(total_loss)
                
                print("(Iteration {}) TRAIN LOSS:{:.4f} CER:{:.2f}% LR:{:.7f} TOTAL TIME:{:.7f}".format(
                    (it+1), total_loss/len(train_data_list), total_cer*100/total_char, self.get_lr(outer_opt), total_time))            
                logging.info("(Iteration {}) TRAIN LOSS:{:.4f} CER:{:.2f}% LR:{:.7f} TOTAL TIME:{:.7f}".format(
                    (it+1), total_loss/len(train_data_list), total_cer*100/total_char, self.get_lr(outer_opt), total_time))
                
                if (it + 1) % last_summary_every == 0:
                    print("(Summary Iteration {} | MA {}) TRAIN LOSS:{:.4f} CER:{:.2f}%".format(
                        (it+1), window_size, sum(last_sum_loss)/len(last_sum_loss), sum(last_sum_cer)*100/sum(last_sum_char)))
                

                # VALID
                if (it + 1) % evaluate_every == 0:
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
                            # torch.cuda.empty_cache()

                            src, trg, src_percentages, src_lengths, trg_lengths = data
                            # try:
                            if args.cuda:
                                src = src.cuda()
                                trg = trg.cuda()
                            loss, cer, num_char = self.forward_one_batch(model, vocab, src, trg, src_percentages, src_lengths, trg_lengths, smoothing, loss_type)
                            total_valid_cer += cer
                            total_valid_char += num_char

                            total_valid_loss += loss.item()
                            valid_pbar.set_description("VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format(ind,
                                total_valid_loss/(i+1), total_valid_cer*100/total_valid_char))
                            # except:
                            #     try:
                            #         # torch.cuda.empty_cache()
                            #         src = src.cpu()
                            #         trg = trg.cpu()
                            #         src_splits, src_lengths_splits, trg_lengths_splits, trg_splits, trg_transcript_splits, src_percentages_splits = iter(src.split(2, dim=0)), iter(src_lengths.split(2, dim=0)), iter(trg_lengths.split(2, dim=0)), iter(trg.split(2, dim=0)), iter(trg_transcript.split(2, dim=0)), iter(src_percentages.split(2, dim=0))
                            #         j = 0
                            #         for src, trg, src_lengths, trg_lengths, src_percentages in zip(src_splits, trg_splits, src_lengths_splits, trg_lengths_splits, src_percentages_splits):
                            #             opt.zero_grad()
                            #             # torch.cuda.empty_cache()
                            #             if args.cuda:
                            #                 src = src.cuda()
                            #                 trg = trg.cuda()

                            #             loss, cer, num_char = self.train_one_batch(model, vocab, src, trg, src_percentages, src_lengths, trg_lengths, smoothing, loss_type)
                            #             total_valid_cer += cer
                            #             total_valid_char += num_char

                            #             if args.clip:
                            #                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                                        
                            #             total_valid_loss += loss.item()
                            #             j += 1
                            #         valid_pbar.set_description("VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format(ind, total_valid_loss/(i+1), total_valid_cer*100/total_valid_char))

                            #         logging.info("probably OOM, autosplit batch. succeeded")
                            #         print("probably OOM, autosplit batch. succeeded")
                            #     except:
                            #         logging.info("probably OOM, autosplit batch. skip batch")
                            #         print("probably OOM, autosplit batch. skip batch")
                            #         continue

                        final_valid_loss = total_valid_loss/(len(valid_loader))
                        final_valid_cer = total_valid_cer*100/total_valid_char

                        final_valid_losses.append(final_valid_loss)
                        final_valid_cers.append(final_valid_cer)
                        print("VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format(ind, final_valid_loss, final_valid_cer))
                        logging.info("VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format(ind, final_valid_loss, final_valid_cer))

                    metrics = {}
                    avg_valid_loss = sum(final_valid_losses) / len(final_valid_losses)
                    avg_valid_cer = sum(final_valid_cers) / len(final_valid_cers)
                    metrics["avg_valid_loss"] = sum(final_valid_losses) / len(final_valid_losses)
                    metrics["avg_valid_cer"] = sum(final_valid_cers) / len(final_valid_cers)
                    metrics["valid_loss"] = final_valid_losses
                    metrics["valid_cer"] = final_valid_cers
                    metrics["history"] = history
                    history.append(metrics)

                    print("AVG VALID LOSS:{:.4f} AVG CER:{:.2f}%".format(sum(final_valid_losses) / len(final_valid_losses), sum(final_valid_cers) / len(final_valid_cers)))
                    logging.info("AVG VALID LOSS:{:.4f} AVG CER:{:.2f}%".format(sum(final_valid_losses) / len(final_valid_losses), sum(final_valid_cers) / len(final_valid_cers)))

                    if it % args.save_every == 0:
                        save_meta_model(model, vocab, (it+1), inner_opt, outer_opt, metrics, args, best_model=False)

                    # save the best model
                    early_stop_criteria, early_stop_val
                    if early_stop_criteria == "cer":
                        print("CRITERIA: CER")
                        if best_valid_val > avg_valid_cer:
                            count_stop = 0
                            best_valid_val = avg_valid_cer
                            save_meta_model(model, vocab, (it+1), inner_opt, outer_opt, metrics, args, best_model=True)
                        else:
                            print("count_stop:", count_stop)
                            count_stop += 1
                    else:
                        print("CRITERIA: LOSS")
                        if best_valid_val > avg_valid_loss:
                            count_stop = 0
                            best_valid_val = avg_valid_loss
                            save_meta_model(model, vocab, (it+1), inner_opt, outer_opt, metrics, args, best_model=True)
                        else:
                            count_stop += 1
                            print("count_stop:", count_stop)

                    if count_stop >= early_stop_val:
                        logging.info("EARLY STOP")
                        print("EARLY STOP\n")
                        break
            except:
                continue
