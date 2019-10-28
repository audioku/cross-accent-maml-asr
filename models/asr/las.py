import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
import math

from utils import constant
from utils.metrics import calculate_metrics
from models.common_layers import DotProductAttention, pad_list


class LAS(nn.Module):
    """
    Listen-Attend-Spell class
    args:
        encoder: Encoder object
        decoder: Decoder object
    """

    def __init__(self, encoder, decoder):
        super(LAS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.id2label = decoder.id2label

        # feature embedding
        if constant.args.emb_cnn:
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11),
                          stride=(2, 2), padding=(0, 10)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), ),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True)
            )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_input, input_lengths, padded_target, verbose=False):
        """
        args:
            (TODO) padded_input: B x H_1 (channel?) x (freq?) x T
            padded_input: B x T x D
            input_lengths: B
            padded_target: B x T
        output:
            pred: B x T x vocab
            gold: B x T
        """

        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs)
        hyp_best_scores, hyp_best_ids = torch.topk(pred, 1, dim=2)

        hyp_seq = hyp_best_ids.squeeze(2)
        gold_seq = gold
        return pred, gold, hyp_seq, gold_seq


class LASEncoder(nn.Module):
    """
    Encoder class
    """

    def __init__(self, dim_input, dim_model, num_layers, dropout=0, bidirectional=False):
        super(LASEncoder, self).__init__()

        self.dim_input = dim_input
        self.dim_model = dim_model
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(dim_input, dim_model, num_layers,
                           dropout=dropout, bidirectional=bidirectional)

    def forward(self, padded_input, input_lengths):
        """
        args:
            padded_input: B x T x D
            input_lengths: B
        output:
            output: B x T x H
            hidden: (num_layers * num_directions) x B x H
        """

        if constant.args.emb_cnn:
            padded_input = self.conv(padded_input)

        # Reshaping features
        sizes = padded_input.size()  # B x H_1 (channel?) x H_2 x T
        # print(sizes)
        # if len(sizes) > 3:
        padded_input = padded_input.view(
            sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH

        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input, input_lengths,
                                            batch_first=True)
        packed_output, hidden = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)
        return output, hidden

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
        if constant.args.emb_cnn:
            padded_input = self.conv(padded_input)

        # Reshaping features
        sizes = padded_input.size()  # B x H_1 (channel?) x H_2 x T
        padded_input = padded_input.view(
            sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH

        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        hyp, gold, *_ = self.decoder(padded_target,
                                     encoder_padded_outputs, input_lengths)
        hyp_best_scores, hyp_best_ids = torch.topk(hyp, 1, dim=2)

        strs_gold = ["".join([self.id2label[int(x)]
                              for x in gold_seq]) for gold_seq in gold]

        if beam_search:
            ids_hyps, strs_hyps = self.decoder.beam_search(
                encoder_padded_outputs, beam_width=beam_width, nbest=beam_nbest)
        else:
            strs_hyps = self.decoder.greedy_search(encoder_padded_outputs)

        if verbose:
            print("GOLD", strs_gold)
            print("HYP", strs_hyps)

        return _, strs_hyps, strs_gold


class LASDecoder(nn.Module):
    """
    Decoder class
    """

    def __init__(self, id2label, num_src_vocab, num_trg_vocab, num_layers, dim_emb, dim_model, dropout=0.1, trg_max_length=1000, emb_trg_sharing=False):
        super(LASDecoder, self).__init__()
        self.sos_id = constant.SOS_TOKEN
        self.eos_id = constant.EOS_TOKEN

        self.id2label = id2label

        self.num_src_vocab = num_src_vocab
        self.num_trg_vocab = num_trg_vocab
        self.num_layers = num_layers
        self.dim_emb = dim_emb
        self.dim_model = dim_model

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

        self.trg_max_length = trg_max_length
        self.trg_embedding = nn.Embedding(num_trg_vocab, dim_emb)

        self.encoder_hidden_size = self.dim_model # NOTE:need to make it changeable
        self.layers = nn.ModuleList([
            nn.LSTMCell(self.dim_emb + self.encoder_hidden_size,
                        self.dim_model)
        ])

        for i in range(1, num_layers):
            self.layers += [nn.LSTMCell(self.dim_model, self.dim_model)]

        self.attention = DotProductAttention()
        self.mlp = nn.Sequential(
            nn.Linear(self.dim_emb + self.encoder_hidden_size, self.dim_model),
        )

        self.output_linear = nn.Linear(dim_model, num_trg_vocab, bias=False)

        if emb_trg_sharing:
            self.output_linear.weight = self.trg_embedding.weight

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

    def forward(self, padded_input, encoder_padded_outputs):
        """
        args:
            padded_input: B x T
            encoder_padded_outputs: B x T x H_enc
        output:
            output: B x T x C (vocab)
        """
        seq_in_pad, seq_out_pad = self.preprocess(padded_input)

        batch_size = padded_input.size(0)
        seq_len = padded_input.size(1)

        h_list, c_list = [], []

        for i in range(len(self.layers)):
            zero_state = encoder_padded_outputs.new_zeros(
                batch_size, self.dim_model)
            h_list.append(zero_state)  # list of B x H
            c_list.append(zero_state.clone())  # list of B x H
        att_c = encoder_padded_outputs.new_zeros(
            batch_size, encoder_padded_outputs.size(2))  # B x H_enc

        embedded = self.dropout(self.trg_embedding(seq_in_pad))

        y_list = []
        att_list = []
        for t in range(seq_len):
            rnn_input = torch.cat((embedded[:, t, :], att_c), dim=1) # B x (embedding_size + H_enc)

            for l in range(len(self.layers)):
                h_list[l], c_list[l] = self.layers[l](
                    rnn_input, h_list[l], c_list[l])
                rnn_input = h_list[l]
            rnn_output = h_list[-1]  # get the last layer output B x T x H_dec

            att_c, att_w = self.attention(
                rnn_output.unsqueeze(dim=1), encoder_padded_outputs)
            mlp_input = torch.cat((rnn_output, att_c), dim=1)
            mlp_output = nn.Tanh(self.mlp(mlp_input))
            y_list.append(mlp_output)
            att_list.append(att_w)

        y_all = torch.stack(y_list, dim=1)  # B x T x H_dec
        seq_logit = self.output_linear(y_all)  # B x T x C

        pred, gold = seq_logit, seq_out_pad

        return pred, gold, att_list

    def greedy_search(self, encoder_padded_outputs):
        """
        Greedy search, decode 1-best utterance
        args:
            encoder_padded_outputs: B x T x H
        output:
            batch_ids_nbest_hyps: list of nbest in ids (size B)
            batch_strs_nbest_hyps: list of nbest in strings (size B)
        """
        batch_size = encoder_padded_outputs.size(0)
        max_seq_len = self.trg_max_length

        h_list, c_list = [], []

        for i in range(len(self.layers)):
            zero_state = encoder_padded_outputs.new_zeros(
                batch_size, self.dim_model)
            h_list.append(zero_state)  # list of B x H
            c_list.append(zero_state.clone())  # list of B x H

        ys = torch.ones(encoder_padded_outputs.size(0), 1).fill_(
            constant.SOS_TOKEN).long()  # batch_size x 1
        att_c = encoder_padded_outputs.new_zeros(
            1, encoder_padded_outputs.size(2))

        if constant.args.cuda:
            ys = ys.cuda()
            att_c = att_c.cuda()

        decoded_words = []
        for t in range(max_seq_len):
            embedded = self.dropout(self.trg_embedding(ys))
            rnn_input = torch.cat((embedded[:, t, :], att_c), dim=1) # B x (embedding_size + H_enc)

            for l in range(len(self.layers)):
                h_list[l], c_list[l] = self.rnn[l](
                    rnn_input, h_list[l], c_list[l])
                rnn_input = h_list[l]
            rnn_output = h_list[-1]  # get the last layer output B x T x H_dec

            att_c, att_w = self.attention(
                rnn_output.unsqueeze(dim=1), encoder_padded_outputs)
            mlp_input = torch.cat((rnn_output, att_c), dim=1)
            mlp_output = self.mlp(mlp_input)

            prob = self.output_linear(mlp_output) # batch_size x t x label_size
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append([constant.EOS_CHAR if ni.item() == constant.EOS_TOKEN else self.id2label[ni.item()] for ni in next_word.view(-1)])
            next_word = next_word.unsqueeze(-1)

            if constant.args.cuda:
                ys = torch.cat([ys, next_word.cuda()], dim=1)
                ys = ys.cuda()
            else:
                ys = torch.cat([ys, next_word], dim=1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == constant.EOS_CHAR:
                    break
                else:
                    st += e
            sent.append(st)
        return sent

    def beam_search(self, encoder_padded_outputs, beam_width=2, nbest=5):
        """
        Beam search, decode nbest utterances
        args:
            encoder_padded_outputs: B x T x H
            beam_size: int
            nbest: int
        output:
            batch_ids_nbest_hyps: list of nbest in ids (size B)
            batch_strs_nbest_hyps: list of nbest in strings (size B)
        """
        batch_size = encoder_padded_outputs.size(0)
        max_len = encoder_padded_outputs.size(1)

        batch_ids_nbest_hyps = []
        batch_strs_nbest_hyps = []

        h_list, c_list = [], []

        for i in range(len(self.layers)):
            zero_state = encoder_padded_outputs.new_zeros(
                batch_size, self.dim_model)
            h_list.append(zero_state)  # list of B x H
            c_list.append(zero_state.clone())  # list of B x H

        att_c = encoder_padded_outputs.new_zeros(1, encoder_padded_outputs.size(2)) # 1 x H

        for x in range(batch_size):
            encoder_output = encoder_padded_outputs[x].unsqueeze(0) # 1 x T x H

            # add SOS_TOKEN
            ys = torch.ones(1, 1).fill_(constant.SOS_TOKEN).type_as(encoder_output).long()
            
            hyp = {'score': 0.0, 'yseq':ys}
            hyps = [hyp]
            ended_hyps = []

            for t in range(self.trg_max_length):
                hyps_best_kept = []
                for hyp in hyps:
                    ys = hyp['yseq'] # 1 x i

                    embedded = self.dropout(self.trg_embedding(ys))
                    rnn_input = torch.cat((embedded[:, t, :], att_c), dim=1) # B x (embedding_size + H_enc)

                    for l in range(len(self.layers)):
                        h_list[l], c_list[l] = self.rnn[l](
                            rnn_input, h_list[l], c_list[l])
                        rnn_input = h_list[l]
                    rnn_output = h_list[-1]  # get the last layer output B x T x H_dec

                    att_c, att_w = self.attention(
                        rnn_output.unsqueeze(dim=1), encoder_padded_outputs)
                    mlp_input = torch.cat((rnn_output, att_c), dim=1)
                    mlp_output = self.mlp(mlp_input)

                    seq_logit = self.output_linear(mlp_output[:, -1])

                    prob = self.output_linear(mlp_output) # batch_size x t x label_size
                    local_scores = F.log_softmax(seq_logit)
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam_width, dim=1)
                    
                    _, next_word = torch.max(prob[:, -1], dim=1)
                    decoded_words.append([constant.EOS_CHAR if ni.item() == constant.EOS_TOKEN else self.id2label[ni.item()] for ni in next_word.view(-1)])
                    next_word = next_word.unsqueeze(-1)

