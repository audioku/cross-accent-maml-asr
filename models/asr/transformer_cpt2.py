import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import numpy as np
import math
import kenlm
import os

from modules.common_layers import FactorizedMultiHeadAttention, PositionalEncoding, PositionwiseFeedForward, FactorizedPositionwiseFeedForward, get_subsequent_mask, get_non_pad_mask, get_attn_key_pad_mask, get_attn_pad_mask, pad_list_with_mask

from torch.autograd import Variable
from utils import constant
from utils.metrics import calculate_metrics

class TransformerCPT2(nn.Module):
    """
    TransformerCPT2 class
    args:
        encoder: Encoder object
        decoder: Decoder object
    """

    def __init__(self, encoder, cpt2, tokenizer, pad_id, sos_id, eos_id, vocab, feat_extractor='vgg_cnn', train=True, is_factorized=False, r=100):
        super(TransformerCPT2, self).__init__()
        self.encoder = encoder
        self.decoder = cpt2
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos = eos_id

        self.feat_extractor = feat_extractor
        self.is_factorized = is_factorized
        self.r = r
        self.eos_id = 0 #TODO: FUCKKKK

        print("feat extractor:", feat_extractor)

        # feature embedding
        self.conv = None
        if feat_extractor == 'emb_cnn':
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(0, 10)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), ),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True)
            )
        elif feat_extractor == 'vgg_cnn':
            self.conv = nn.Sequential(
                nn.Conv2d(1, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            )
        elif feat_extractor == "large_cnn":
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def preprocess(self, padded_input):
        """
        Add SOS TOKEN and EOS TOKEN into padded_input
        """
        seq = [y[y != self.pad_id] for y in padded_input]
        eos = seq[0].new([self.eos_id])
        sos = seq[0].new([self.sos_id])
        seq_in = [torch.cat([sos, y], dim=0) for y in seq]
        seq_out = [torch.cat([y, eos], dim=0) for y in seq]
        seq_in_pad, mask = pad_list_with_mask(seq_in, self.pad_id)
        seq_out_pad, mask = pad_list_with_mask(seq_out, self.pad_id)

        assert seq_in_pad.size() == seq_out_pad.size()
        # print(seq_in_pad.size(), seq_out_pad.size())
        return seq_in_pad, seq_out_pad

    def forward(self, padded_input, input_lengths, padded_target, verbose=False):
        """
        args:
            padded_input: B x 1 (channel for spectrogram=1) x (freq) x T
            padded_input: B x T x D
            input_lengths: B
            padded_target: B x T
        output:
            pred: B x T x vocab
            gold: B x T
        """
        if self.feat_extractor == 'emb_cnn' or self.feat_extractor == 'vgg_cnn' or self.feat_extractor == 'large_cnn':
            padded_input = self.conv(padded_input)

        # Reshaping features
        sizes = padded_input.size() # B x H_1 (channel?) x H_2 x T
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH
        
        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)

        # Padded target add sos & eos
        seq_in_pad, seq_out_pad, mask = self.preprocess(padded_input)
        final_decoded_output = []

        # Forward model
        decoder_output = self.decoder(seq_in_pad, encoder_padded_outputs, encoder_padded_outputs, attention_mask=mask)

        # Prepare output list
        pred_list = []
        gold_list = []
        for i in range(len(decoder_output)):
            pred_list.append(self.output_linear(decoder_output[i].unsqueeze(0)).squeeze())
            gold_list.append(seq_out_pad[i])
        
        pred_list = torch.stack(pred_list, dim=0)
        gold_list = torch.stack(gold_list, dim=0)

        hyp_best_scores, hyp_best_ids = torch.topk(pred_list, 1, dim=2)
        hyp_list = hyp_best_ids.squeeze(2)

        return pred_list, gold_list, hyp_list

    def evaluate(self, padded_input, input_lengths, padded_target, args, beam_search=False, beam_width=0, beam_nbest=0, lm=None, lm_rescoring=False, lm_weight=0.1, c_weight=1, start_token=-1, verbose=False):
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
        if self.feat_extractor == 'emb_cnn' or self.feat_extractor == 'vgg_cnn' or self.feat_extractor == 'large_cnn':
            padded_input = self.conv(padded_input)

        # Reshaping features
        sizes = padded_input.size() # B x H_1 (channel?) x H_2 x T
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH

        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        pred_list, gold_list, *_ = self.decoder(padded_target, encoder_padded_outputs, input_lengths)
        strs_gold = ["".join([self.vocab.id2label[int(x)] for x in gold_seq]) for gold_seq in gold_list]

        if beam_search:
            ids_hyps, strs_hyps = self.decoder.beam_search(encoder_padded_outputs, args, beam_width=beam_width, nbest=1, lm=lm, lm_rescoring=lm_rescoring, lm_weight=lm_weight, c_weight=c_weight, start_token=start_token)
            if len(strs_hyps) != sizes[0]:
                print(">>>>>>> switch to greedy")
                strs_hyps = self.decoder.greedy_search(encoder_padded_outputs, args, start_token=start_token)
        else:
            strs_hyps = self.decoder.greedy_search(encoder_padded_outputs, args, start_token=start_token)

        return _, strs_hyps, strs_gold

class Encoder(nn.Module):
    """ 
    Encoder Transformer class
    """

    def __init__(self, num_layers, num_heads, dim_model, dim_key, dim_value, dim_input, dim_inner, dropout=0.1, src_max_length=2500, is_factorized=False, r=100):
        super(Encoder, self).__init__()

        self.dim_input = dim_input
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.dim_model = dim_model
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dim_inner = dim_inner

        self.src_max_length = src_max_length

        self.is_factorized = is_factorized
        self.r = r

        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

        if is_factorized:
            self.input_linear_a = nn.Linear(dim_input, r, bias=False)
            self.input_linear_b = nn.Linear(r, dim_model)
        else:
            self.input_linear = nn.Linear(dim_input, dim_model)
        self.layer_norm_input = nn.LayerNorm(dim_model)
        self.positional_encoding = PositionalEncoding(
            dim_model, src_max_length)

        self.layers = nn.ModuleList([
            EncoderLayer(num_heads, dim_model, dim_inner, dim_key, dim_value, dropout=dropout, is_factorized=is_factorized, r=r) for _ in range(num_layers)
        ])

    def forward(self, padded_input, input_lengths):
        """
        args:
            padded_input: B x T x D
            input_lengths: B
        return:
            output: B x T x H
        """
        encoder_self_attn_list = []

        # Prepare masks
        non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)  # B x T x D
        seq_len = padded_input.size(1)
        self_attn_mask = get_attn_pad_mask(padded_input, input_lengths, seq_len)  # B x T x T

        if self.is_factorized:
            encoder_output = self.layer_norm_input(self.input_linear_b(self.input_linear_a(
                padded_input))) + self.positional_encoding(padded_input)
        else:
            encoder_output = self.layer_norm_input(self.input_linear(
                padded_input)) + self.positional_encoding(padded_input)
        
        for layer in self.layers:
            encoder_output, self_attn = layer(
                encoder_output, non_pad_mask=non_pad_mask, self_attn_mask=self_attn_mask)
            encoder_self_attn_list += [self_attn]

        return encoder_output, encoder_self_attn_list


class EncoderLayer(nn.Module):
    """
    Encoder Layer Transformer class
    """

    def __init__(self, num_heads, dim_model, dim_inner, dim_key, dim_value, dropout=0.1, is_factorized=False, r=100):
        super(EncoderLayer, self).__init__()
        self.is_factorized = is_factorized
        self.r = r
        self.self_attn = FactorizedMultiHeadAttention(num_heads, dim_model, dim_key, dim_value, dropout=dropout, r=r)
        if is_factorized:
            self.pos_ffn = FactorizedPositionwiseFeedForward(dim_model, dim_inner, dropout=dropout, r=r)
        else:
            self.pos_ffn = PositionwiseFeedForward(dim_model, dim_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, self_attn_mask=None):
        enc_output, self_attn = self.self_attn(
            enc_input, enc_input, enc_input, mask=self_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, self_attn
