import math
import time

import torch
import torch.nn as nn

from tqdm import tqdm
from utils import constant
from utils.functions import save_model
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer

class CharLMTrainer():
    """
    Character-based Language Model Trainer class
    """
    def __init__(self):
        print("CharLMTrainer is initialized")

    def train(self, model, train_loader, valid_loader, opt, start_epoch, num_epochs, label2id, id2label, last_metrics=None):
        """
        Training
        args:
            model: Model object
            train_loader: DataLoader object of the training set
            valid_loader: DataLoader object of the validation set
            opt: Optimizer object
            start_epoch: start epoch (> 0 if you resume the process)
            num_epochs: last epoch
            last_metrics: (if resume)
        """
        history = []
        start_time = time.time()
        best_valid_loss = 1000000000 if last_metrics is None else last_metrics['valid_loss']
        smoothing = constant.args.label_smoothing

        criterion = nn.CrossEntropyLoss()

        for epoch in range(start_epoch, num_epochs):
            total_loss, total_char = 0, 0

            start_iter = 0

            print("TRAIN")
            model.train()
            pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
            for i, (data) in enumerate(pbar, start=start_iter):
                src, src_lengths = data

                if constant.USE_CUDA:
                    src = src.cuda()

                opt.optimizer.zero_grad()

                logits, seq_in_pad, seq_out_pad = model(src, src_lengths, verbose=False)
                loss = criterion(logits.view(-1, logits.size(-1)), seq_out_pad.view(-1))

                loss.backward()
                opt.optimizer.step()

                total_loss += loss.item()
                total_char += torch.sum(src_lengths).item()

                ppl = math.exp(total_loss/(i+1))

                pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} PPL:{:.2f}".format(
                    (epoch+1), total_loss/(i+1), ppl))

            # evaluate
            print("VALID")
            model.eval()

            total_valid_loss, total_valid_char = 0, 0
            valid_pbar = tqdm(iter(valid_loader), leave=True,
                            total=len(valid_loader))
            for i, (data) in enumerate(valid_pbar):
                src, src_lengths = data

                if constant.USE_CUDA:
                    src = src.cuda()

                opt.optimizer.zero_grad()

                logits, seq_in_pad, seq_out_pad = model(src, src_lengths, verbose=False)
                loss = criterion(logits.view(-1, logits.size(-1)), seq_out_pad.view(-1))

                loss.backward()
                opt.optimizer.step()

                total_valid_loss += loss.item()
                total_valid_char += torch.sum(src_lengths).item()

                ppl = math.exp(loss)
                valid_pbar.set_description("(Epoch {}) VALID LOSS:{:.4f} PPL:{:.2f}".format(
                    (epoch+1), total_valid_loss/(i+1), ppl))

            metrics = {}
            metrics["train_loss"] = total_loss / (len(pbar))
            metrics["valid_loss"] = total_valid_loss / (len(valid_pbar))
            history.append(metrics)

            if epoch % constant.args.save_every == 0:
                save_model(model, (epoch+1), opt, metrics,
                        label2id, id2label, best_model=False)

            # save the best model
            if best_valid_loss > total_valid_loss/len(valid_loader):
                best_valid_loss = total_valid_loss/len(valid_loader)
                save_model(model, (epoch+1), opt, metrics,
                        label2id, id2label, best_model=True)