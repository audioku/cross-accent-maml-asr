import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import numpy as np
import math
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
        self.eos_id = eos_id

        self.feat_extractor = feat_extractor
        self.is_factorized = is_factorized
        self.r = r

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
        seq_out_pad, _ = pad_list_with_mask(seq_out, self.pad_id)
        mask = mask.to(padded_input.device)

#         # DEBUG
#         print('seq_in_pad', seq_in_pad)
#         print('seq_out_pad', seq_out_pad)
#         print('mask', mask)
        
        assert seq_in_pad.size() == seq_out_pad.size()
        # print(seq_in_pad.size(), seq_out_pad.size())
        return seq_in_pad, seq_out_pad, mask

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
        seq_in_pad, seq_out_pad, mask = self.preprocess(padded_target)
        final_decoded_output = []

        # Forward model
        decoder_output = self.decoder(seq_in_pad, encoder_padded_outputs, encoder_padded_outputs, attention_mask=None)

        # Prepare output list
        pred_list = decoder_output
        
        gold_list = []
        for i in range(len(seq_out_pad)):
            gold_list.append(seq_out_pad[i])
        gold_list = torch.stack(gold_list, dim=0)

        hyp_best_scores, hyp_best_ids = torch.topk(pred_list, 1, dim=2)
        hyp_list = hyp_best_ids.squeeze(2)

        non_pad_mask = torch.logical_not(mask)
        
#         # DEBUG
#         print('gold_list', gold_list)
#         print('hyp_list', hyp_list)

        return pred_list, gold_list, hyp_list, non_pad_mask

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