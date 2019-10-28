import time
import numpy as np
from tqdm import tqdm
from utils import constant
from utils.functions import save_model
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer
from copy import deepcopy
from torch.autograd import Variable
import torch
import torch.nn as nn
import logging

import sys

class MetaTrainer():
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

    def sample(self, data_loader):
        x1 = next(data_loader)
        x2 = next(data_loader)
        return [x1, x2]

    def train_one_batch(self, model, tr_batch, smoothing, loss_type, opt, is_train=True):
        """

        """
        tr_src, tr_trg, tr_trg_transcript, tr_src_percentages, tr_src_lengths, tr_trg_lengths, tr_langs, tr_lang_names = tr_batch
        pred, gold, hyp = model(tr_src, tr_src_lengths, tr_trg, tr_trg_transcript, tr_langs, tr_lang_names, verbose=False)
        
        # case for the last batch
        for j in range(len(pred)):
            if len(pred[j]) > 0:
                seq_length = pred[j].size(1)
                break
        sizes = Variable(tr_src_percentages.mul_(int(seq_length)).int(), requires_grad=False)

        # loss before update
        loss = None
        for j in range(len(pred)):
            # try:
            if len(pred[j]) == 0:
                continue
            t_loss, num_correct = calculate_metrics(
                pred[j], gold[j], input_lengths=sizes, target_lengths=tr_trg_lengths, smoothing=smoothing, loss_type=loss_type)
            if loss is None:
                loss = t_loss
            else:
                loss = loss + t_loss
        
        if is_train:
            loss.backward()
            opt.step()
        
        return loss

    def do_evaluation(self, model, val_batch, smoothing, loss_type, opt):
        """

        """
        model.eval()
        val_loss = model.train_one_batch(val_batch, smoothing, loss_type, opt, is_train=False)
        return val_loss

    def do_evaluation_all(self, model, valid_loader, smoothing, loss_type, opt):
        model.eval()

        losses = []
        valid_pbar = tqdm(iter(valid_loader), leave=True, total=len(valid_loader))
        for i, (data) in enumerate(valid_pbar):
            val_loss = model.train_one_batch(data, smoothing, loss_type, opt, is_train=False)
            losses.append(val_loss)
        return np.mean(losses)


    def train(self, model, train_loader_list, train_sampler, valid_loader_list, special_token_list, opt, loss_type, start_epoch, num_epochs, src_label2id, src_id2label, trg_label2ids, trg_id2labels, last_metrics=None, early_stop=10, max_grad_norm=500):
        """
        Training
        args:
            model: Model object
            train_loader_list: DataLoader object of the training set
            valid_loader_list: a list of Validation DataLoader objects
            opt: Optimizer object
            start_epoch: start epoch (> 0 if you resume the process)
            num_epochs: last epoch
            last_metrics: (if resume)
        """
        history = []
        best_valid_loss, best_valid_cer = 1000000000, 0
        smoothing = constant.args.label_smoothing
        early_stop_criteria, early_stop_val = early_stop.split(",")[0], int(early_stop.split(",")[1])
        count_stop = 0

        logging.info("name " +  constant.args.name)

        for i in range(len(train_loader_list)):
            train_loader_list[i] = iter(train_loader_list[i])

        for epoch in range(start_epoch, num_epochs):
            logging.info("META-TRAIN")
            print("META-TRAIN")
            model.train()

            weights_original = deepcopy(model.state_dict())

            num_task = len(train_loader_list)
            for i in range(num_task):
                torch.cuda.empty_cache()

                ac_tr_loss, ac_val_loss = [], []
                tr_batch, val_batch = self.sample(train_loader_list[i])
                opt.zero_grad()

                # update fast nets
                tr_loss = self.train_one_batch(model, tr_batch, smoothing, loss_type)
                ac_tr_loss.append(tr_loss)
                
                val_loss = self.do_evaluation(model, val_batch)
                ac_val_loss.append(val_loss)
                
            ac_tr_loss = np.mean(ac_tr_loss)
            ac_val_loss = np.mean(ac_val_loss)

            # reset
            model.load_state_dict({ name: weights_original[name] for name in weights_original })

            # meta update
            ac_val_loss.backward()
            
            # clip gradient
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            print("META-EVAL")
            if epoch % 10 == 0:
                valid_num_task = len(valid_loader_list)
                for i in range(valid_num_task):
                    torch.cuda.empty_cache()

                    ac_val_loss = None
                    opt.zero_grad()
                    
                    val_loss = self.do_evaluation_all(model, valid_loader_list[i])
                    ac_val_loss.append(val_loss)

                ac_val_loss = np.mean(ac_val_loss)

                metrics = {}

                if best_valid_loss > ac_val_loss:
                    save_model(model, (epoch+1), opt, metrics,
                            src_label2id, src_id2label, trg_label2ids, trg_id2labels, best_model=True)
                    best_valid_loss = ac_val_loss
                    count_stop = 0
                else:
                    count_stop += 1

                if count_stop >= early_stop_val:
                    logging.info("EARLY STOP")
                    print("EARLY STOP\n")
                    break

            if constant.args.shuffle:
                logging.info("SHUFFLE")
                print("SHUFFLE\n")
                train_sampler.shuffle(epoch)