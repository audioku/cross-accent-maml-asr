import time
from tqdm import tqdm
from utils import constant
from utils.functions import save_model
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer

class EWCTrainer():
    """
    Elastic Weight Consilidation Trainer class
    """
    def __init__(self):
        print("Trainer is initialized")
        self.fisher = {}
        self.optpar = {}
        self.n_memories = constant.args.ewc_memory

        self.src_memories = []
        self.tgt_memories = []
        self.src_lengths_memories = []
        self.tgt_lengths_memories = []

    def train(self, model, train_loaders, valid_loaders, opt, start_epoch, num_epochs, label2id, id2label):
        """
        Training
        args:
            model: Model object
            train_loaders: list of DataLoader object of the training set
            valid_loaders: list of DataLoader object of the validation set
            opt: Optimizer object
            start_epoch: start epoch (> 0 if you resume the process)
            num_epochs: last epoch
        """
        start_time = time.time()
        best_valid_loss = 1000000000
        smoothing = constant.args.label_smoothing

        for task_id in range(len(train_loaders)):
            train_loader = train_loaders[task_id]
            valid_loader = valid_loaders[task_id]

            if task_id > 0:
                print("Fisher calculation")
                sum_loss = None
                for j in range(len(self.src_memories)):
                    pred, gold, hyp_seq, gold_seq = model(self.src_memories[j], self.src_lengths_memories[j], 
                        self.tgt_memories[j], verbose=False)
                    loss, num_correct = calculate_metrics(pred, gold, smoothing=smoothing)
                    if sum_loss is None:
                        sum_loss = loss
                    else:
                        sum_loss += loss    
                sum_loss /= len(self.src_memories)
                print(sum_loss)                        
                sum_loss.backward()

                self.fisher[task_id-1], self.optpar[task_id-1] = [], []
                for p in model.parameters():
                    pd = p.data.clone()
                    pg = p.grad.data.clone().pow(2)
                    self.optpar[task_id-1].append(pd)
                    self.fisher[task_id-1].append(pg)

            # clear memories
            self.src_memories = []
            self.tgt_memories = []
            self.src_lengths_memories = []
            self.tgt_lengths_memories = []

            for epoch in range(start_epoch, num_epochs):
                total_loss, total_cer, total_wer, total_char, total_word = 0, 0, 0, 0, 0

                start_iter = 0

                print("TRAIN")
                model.train()
                pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
                for i, (data) in enumerate(pbar, start=start_iter):
                    src, tgt, src_percentages, src_lengths, tgt_lengths = data

                    if constant.USE_CUDA:
                        src = src.cuda()
                        tgt = tgt.cuda()
                    
                    opt.optimizer.zero_grad()

                    pred, gold, hyp_seq, gold_seq = model(
                        src, src_lengths, tgt, verbose=False)

                    strs_gold = ["".join([id2label[int(x)] for x in gold]) for gold in gold_seq]
                    strs_hyps = ["".join([id2label[int(x)] for x in hyp]) for hyp in hyp_seq]

                    loss, num_correct = calculate_metrics(
                        pred, gold, smoothing=smoothing)

                    for j in range(len(strs_hyps)):
                        cer = calculate_cer(strs_hyps[j], strs_gold[j])
                        wer = calculate_wer(strs_hyps[j], strs_gold[j])
                        total_cer += cer
                        total_wer += wer
                        total_char += len(strs_gold[j])
                        total_word += len(strs_gold[j].split(" "))

                    # EWC LOSS
                    if task_id > 0:
                        for id in range(task_id):
                            for k, p in enumerate(model.parameters()):
                                l = constant.args.ewc_reg * self.fisher[id][k]
                                l = l * (p - self.optpar[id][k]).pow(2)
                                loss += l.sum()

                    if len(self.src_memories) < self.n_memories:
                        self.src_memories.append(src)
                        self.tgt_memories.append(tgt)
                        self.src_lengths_memories.append(src_lengths)

                    loss.backward()
                    opt.optimizer.step()

                    total_loss += loss.item()
                    non_pad_mask = gold.ne(constant.PAD_TOKEN)
                    num_word = non_pad_mask.sum().item()

                    pbar.set_description("(Epoch {}) TASK:{} TRAIN LOSS:{:.4f} CER:{:.2f}% WER:{:.2f}%".format(
                        (epoch+1), task_id, total_loss/(i+1), total_cer*100/total_char, total_wer*100/total_word))

                # print("VALID")
                # for valid_task_id in range(len(valid_loaders)):
                #     model.eval()

                #     valid_loader = valid_loaders[valid_task_id]

                #     total_valid_loss, total_valid_cer, total_valid_wer, total_valid_char, total_valid_word = 0, 0, 0, 0, 0
                #     valid_pbar = tqdm(iter(valid_loader), leave=True,
                #                     total=len(valid_loader))
                #     for i, (data) in enumerate(valid_pbar):
                #         src, tgt, src_percentages, src_lengths, tgt_lengths = data

                #         if constant.USE_CUDA:
                #             src = src.cuda()
                #             tgt = tgt.cuda()

                #         pred, gold, hyp_seq, gold_seq = model(
                #             src, src_lengths, tgt, verbose=constant.args.verbose)
                #         loss, num_correct = calculate_metrics(
                #             pred, gold, smoothing=smoothing)

                #         strs_gold = ["".join([id2label[int(x)] for x in gold]) for gold in gold_seq]
                #         strs_hyps = ["".join([id2label[int(x)] for x in hyp]) for hyp in hyp_seq]

                #         for j in range(len(strs_hyps)):
                #             cer = calculate_cer(strs_hyps[j], strs_gold[j])
                #             wer = calculate_wer(strs_hyps[j], strs_gold[j])
                #             total_valid_cer += cer
                #             total_valid_wer += wer
                #             total_valid_char += len(strs_gold[j])
                #             total_valid_word += len(strs_gold[j].split(" "))

                #         total_valid_loss += loss.item()
                #         valid_pbar.set_description("(Epoch {}) TASK:{} VALID LOSS:{:.4f} CER:{:.2f}% WER:{:.2f}%".format(
                #             (epoch+1), valid_task_id, total_valid_loss/(i+1), total_valid_cer*100/total_valid_char, total_valid_wer*100/total_valid_word))

                    # metrics = {}
                    # metrics["train_loss"] = total_loss / (epoch + 1)
                    # metrics["valid_loss"] = total_valid_loss / (len(valid_pbar))
                    # metrics["train_cer"] = total_cer
                    # metrics["train_wer"] = total_wer
                    # metrics["valid_cer"] = total_valid_cer
                    # metrics["valid_wer"] = total_valid_wer
                    # history.append(metrics)