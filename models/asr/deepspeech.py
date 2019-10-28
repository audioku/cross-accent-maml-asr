import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch import autograd
from tqdm import tqdm

from utils import constant
from models.common_layers import pad_list


class DeepSpeech(nn.Module):
    def __init__(self, dim_input, dim_model=768, num_layers=5, bidirectional=True, context=20, label2id=None, id2label=None):
        super(DeepSpeech, self).__init__()

        self.dim_input = dim_input
        self.dim_model = dim_model
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.context = context
        self.label2id = label2id
        self.id2label = id2label

        num_classes = len(self.label2id)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11),
                      stride=(2, 2), padding=(0, 10)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), ),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )

        rnn_type = nn.LSTM

        rnns = []
        rnn = BatchRNN(input_size=dim_input, hidden_size=dim_model, rnn_type=rnn_type,
                       bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(num_layers - 1):
            rnn = BatchRNN(input_size=dim_model, hidden_size=dim_model, rnn_type=rnn_type,
                           bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))

        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(dim_model, context=context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(dim_model),
            nn.Linear(dim_model, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

        self.inference_softmax = InferenceBatchSoftmax()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def preprocess(self, padded_input):
        """
        Add SOS TOKEN and EOS TOKEN into padded_input
        """
        seq = [y[y != constant.PAD_TOKEN] for y in padded_input]
        eos = seq[0].new([self.eos_id])
        sos = seq[0].new([self.sos_id])
        seq_in = [torch.cat([sos, y], dim=0) for y in seq]
        seq_out = [torch.cat([y, eos], dim=0) for y in seq]
        seq_in_pad = pad_list(seq_in, self.eos_id)
        seq_out_pad = pad_list(seq_out, constant.PAD_TOKEN)
        assert seq_in_pad.size() == seq_out_pad.size()
        return seq_in_pad, seq_out_pad

    def forward(self, padded_input, input_lengths, padded_target, verbose=False):        
        """
        padded_input: torch.Size([8, 1, 161, 428])
        input_lengths: torch.Size([8])
        padded_target: torch.Size([8, 46])
        """
        # print("padded_input:", padded_input.size())
        # print("input_lengths:", input_lengths.size())
        # print("padded_target:", padded_target.size())

        # print("padded")
        # print(padded_input)

        x = padded_input
        x = self.conv(x)
        sizes = x.size()
        # Collapse feature dimension
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        x = self.rnns(x)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        seq_logit = self.inference_softmax(x)

        # print("seq_logit", seq_logit.size())
        # print(seq_logit)
        hyp_best_scores, hyp_best_ids = torch.topk(seq_logit, 1)
        # print(hyp_best_ids)

        # gold_seq = []
        # for i in range(len(padded_target)):
        #     gold_seq.append("".join(self.id2label[char.item()] for char in padded_target[i]))

        # hyp_best_ids = hyp_best_ids.squeeze()
        # print("hyp_best_ids:", hyp_best_ids.size())

        # hyp_seq = []
        # for i in range(len(hyp_best_ids)):
        #     hyp_seq.append("".join(self.id2label[char.item()] for char in hyp_best_ids[i]))

        pred, gold = seq_logit, padded_target
        gold_seq = gold
        hyp_seq = hyp_best_ids.squeeze(2)

        # print(pred.size(), gold.size())

        return pred, gold, hyp_seq, gold_seq

    def evaluate(self, padded_input, input_lengths, padded_target, beam_search=False, beam_width=0, beam_nbest=0, verbose=False):
        """
        args:
            padded_input: B x T x D
            input_lengths: B
            padded_target: B x T
        output:
            batch_ids_nbest_hyps: list of nbest id
            batch_strs_nbest_hyps: list of nbest str
            batch_strs_gold: list of gold str
        """
        # Reshaping features

        padded_input = self.conv(padded_input)

        sizes = padded_input.size() # B x H_1 (channel?) x H_2 x T
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH
        
        sizes = padded_input.size()
        # Collapse feature dimension
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        x = self.rnns(padded_input)

        if not self._bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)

        if beam_search:
            self.beam_search(x, beam_width=beam_width, beam_nbest=beam_nbest)
        else:
            self.greedy_search(x)
            print(">>>>>>>>>>>>>", x.size())

        return x

    def beam_search(self, encoder_padded_outputs, beam_width):
        return None

    def greedy_search(self, encoder_padded_outputs):
        """
        x: batch_size, seq_len, output_vocab
        """
        return None

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(
            nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.bidirectional:
            # (TxNxH*2) -> (TxNxH) by sum
            x = x.view(x.size(0), x.size(1), 2, -
                       1).sum(2).view(x.size(0), x.size(1), -1)
        return x


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        # should we handle batch_first=True?
        super(Lookahead, self).__init__()
        self.n_features = n_features
        self.weight = Parameter(torch.Tensor(n_features, context + 1))
        assert context > 0
        self.context = context
        self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):  # what's a better way initialiase this layer?
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        seq_len = input.size(0)
        # pad the 0th dimension (T/sequence) with zeroes whose number = context
        # Once pytorch's padding functions have settled, should move to those.
        padding = torch.zeros(
            self.context, *(input.size()[1:])).type_as(input.data)
        x = torch.cat((input, Variable(padding)), 0)

        # add lookahead windows (with context+1 width) as a fourth dimension
        # for each seq-batch-feature combination
        # TxLxNxH - sequence, context, batch, feature
        x = [x[i:i + self.context + 1] for i in range(seq_len)]
        x = torch.stack(x)
        # TxNxHxL - sequence, batch, feature, context
        x = x.permute(0, 2, 3, 1)

        x = torch.mul(x, self.weight).sum(dim=3)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'n_features=' + str(self.n_features) \
            + ', context=' + str(self.context) + ')'
