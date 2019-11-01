import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch.autograd import Variable

import numpy as np
import math

from utils import constant
from models.common_layers import FactorizedMultiHeadAttention, PositionalEncoding, PositionwiseFeedForward, FactorizedPositionwiseFeedForward, get_subsequent_mask, get_non_pad_mask, get_attn_key_pad_mask, get_attn_pad_mask, pad_list

from utils.metrics import calculate_metrics

import kenlm
import os

from utils.lstm_utils import calculate_lm_score

class Transformer(nn.Module):
    """
    Transformer class
    args:
        encoder: Encoder object
        decoder: Decoder object
    """

    def __init__(self, encoder, decoder, feat_extractor='vgg_cnn', train=True, is_factorized=False, r=100):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trg_id2labels = decoder.trg_id2labels
        self.feat_extractor = feat_extractor
        self.is_factorized = is_factorized
        self.r = r

        print("feat extractor:", feat_extractor)

        # feature embedding
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
        else:
            self.conv = None
        if self.conv is not None:
            self.conv = self.conv if train else self.conv

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_input, input_lengths, padded_target, padded_target_transcript, langs, lang_names, verbose=False):
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
        try:                
            if self.feat_extractor == 'emb_cnn' or self.feat_extractor == 'vgg_cnn' or self.feat_extractor == 'large_cnn':
                padded_input = self.conv(padded_input)

            # Reshaping features
            sizes = padded_input.size() # B x H_1 (channel?) x H_2 x T
            padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2], sizes[3])
            padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH
            
            encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths, langs)
            pred_list, gold_list, *_ = self.decoder(padded_target, padded_target_transcript, encoder_padded_outputs, input_lengths, langs, lang_names)

            hyp_list = []
            for i in range(len(pred_list)):
                if len(pred_list[i]) > 0:
                    pred, gold = pred_list[i], gold_list[i]
                    hyp_best_scores, hyp_best_ids = torch.topk(pred, 1, dim=2)
                    hyp_seq = hyp_best_ids.squeeze(2)
                    hyp_list.append(hyp_seq)
                else:
                    hyp_list.append([])
            return pred_list, gold_list, hyp_list
        except:
            torch.cuda.empty_cache()

    def evaluate(self, padded_input, input_lengths, padded_target, padded_target_transcript, beam_search=False, beam_width=0, beam_nbest=0, lm=None, lm_rescoring=False, lm_weight=0.1, c_weight=1, start_token=constant.SOS_TOKEN, langs=None, lang_names=None, verbose=False, lang_id=0):
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
        pred_list, gold_list, *_ = self.decoder(padded_target, padded_target_transcript, encoder_padded_outputs, input_lengths, langs=langs, lang_names=lang_names)
       
        gold = gold_list[lang_id]

        if len(self.trg_id2labels) == 1:
            strs_gold = ["".join([self.trg_id2labels[0][int(x)] for x in gold_seq]) for gold_seq in gold]
        else:
            strs_gold = ["".join([self.trg_id2labels[lang_id][int(x)] for x in gold_seq]) for gold_seq in gold]

        if beam_search:
            ids_hyps, strs_hyps = self.decoder.beam_search(encoder_padded_outputs, beam_width=beam_width, nbest=1, lm=lm, lm_rescoring=lm_rescoring, lm_weight=lm_weight, c_weight=c_weight, start_token=start_token,lang_id=lang_id)
            if len(strs_hyps) != sizes[0]:
                print(">>>>>>> switch to greedy")
                strs_hyps = self.decoder.greedy_search(encoder_padded_outputs, start_token=start_token, lang_id=lang_id)
        else:
            strs_hyps = self.decoder.greedy_search(encoder_padded_outputs, start_token=start_token, lang_id=lang_id)

        return _, strs_hyps, strs_gold

class Encoder(nn.Module):
    """ 
    Encoder Transformer class
    """

    def __init__(self, num_layers, num_heads, dim_model, dim_key, dim_value, dim_input, dim_inner, dropout=0.1, src_max_length=2500, num_lang=0, is_factorized=False, r=100):
        super(Encoder, self).__init__()

        self.dim_input = dim_input
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_lang = num_lang

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

    def forward(self, padded_input, input_lengths, langs=None):
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

class Decoder(nn.Module):
    """
    Decoder Layer Transformer class
    """

    def __init__(self, src_label2id, src_id2label, trg_label2ids, trg_id2labels, num_layers, num_heads, dim_emb, dim_model, dim_inner, dim_key, dim_value, dropout=0.1, trg_max_length=1000, emb_trg_sharing=False, num_lang=0, combine_decoder=False, is_factorized=False, r=100):
        super(Decoder, self).__init__()
        self.sos_id = constant.SOS_TOKEN
        self.eos_id = constant.EOS_TOKEN

        self.src_label2id = src_label2id
        self.src_id2label = src_id2label
        self.trg_label2ids = trg_label2ids
        self.trg_id2labels = trg_id2labels

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_lang = num_lang

        self.dim_emb = dim_emb
        self.dim_model = dim_model
        self.dim_inner = dim_inner
        self.dim_key = dim_key
        self.dim_value = dim_value

        self.dropout_rate = dropout
        self.emb_trg_sharing = emb_trg_sharing

        self.trg_max_length = trg_max_length

        self.is_factorized = is_factorized
        self.r = r

        self.trg_embedding = nn.Embedding(len(self.src_label2id), dim_emb, padding_idx=constant.PAD_TOKEN)
        self.positional_encoding = PositionalEncoding(dim_model, max_length=trg_max_length)

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(dim_model, dim_inner, num_heads,
                         dim_key, dim_value, dropout=dropout, is_factorized=is_factorized, r=r)
            for _ in range(num_layers)
        ])

        # self.output_linear = nn.Linear(dim_model, num_trg_vocab, bias=False)
        self.combine_decoder = combine_decoder
        if combine_decoder:
            self.output_linears = nn.ModuleList([nn.Linear(dim_model, len(self.trg_label2ids[0]), bias=False)])
        else:
            self.output_linears = nn.ModuleList([nn.Linear(dim_model, len(self.trg_label2ids[i]), bias=False) for i in range(num_lang)])

        for i in range(len(self.output_linears)):
            nn.init.xavier_normal_(self.output_linears[i].weight)
        # nn.init.xavier_normal_(self.output_linear.weight)

        if emb_trg_sharing:
            # self.output_linear.weight = self.trg_embedding.weight
            # self.x_logit_scale = (dim_model ** -0.5)
            self.x_logit_scale = 1.0
        else:
            self.x_logit_scale = 1.0

    def preprocess(self, padded_input, langs, lang_names):
        """
        Add SOS TOKEN and EOS TOKEN into padded_input
        """
        seq = [y[y != constant.PAD_TOKEN] for y in padded_input]
        eos = seq[0].new([constant.EOS_TOKEN])

        sos = []
        # print(self.trg_label2ids[2])
        for i in range(len(lang_names)):
            if len(self.trg_label2ids) == 1:
                sos.append(seq[0].new([self.trg_label2ids[0][lang_names[i]]]))
            else:
            # print(langs[i], lang_names[i], self.trg_label2ids[langs[i]][lang_names[i]])
                sos.append(seq[0].new([self.trg_label2ids[langs[i]][lang_names[i]]]))
            # sos.append(seq[0].new([constant.SOS_TOKEN]))
        
        seq_in = [torch.cat([sos[i], seq[i]], dim=0) for i in range(len(seq))]
        seq_out = [torch.cat([y, eos], dim=0) for y in seq]
        seq_in_pad = pad_list(seq_in, self.eos_id)
        seq_out_pad = pad_list(seq_out, constant.PAD_TOKEN)

        assert seq_in_pad.size() == seq_out_pad.size()
        # print(seq_in_pad.size(), seq_out_pad.size())
        return seq_in_pad, seq_out_pad

    def forward(self, padded_input, padded_input_transcript, encoder_padded_outputs, encoder_input_lengths, langs=None, lang_names=None):
        """
        args:
            padded_input: B x T
            encoder_padded_outputs: B x T x H
            encoder_input_lengths: B
            langs: B
            lang_names: B
        returns:
            pred: B x T x vocab
            gold: B x T
        """
        decoder_self_attn_list, decoder_encoder_attn_list = [], []
        seq_in_pad, _ = self.preprocess(padded_input, langs, lang_names)
        _, seq_out_pad = self.preprocess(padded_input_transcript, langs, lang_names)

        # Prepare masks
        non_pad_mask = get_non_pad_mask(seq_in_pad, pad_idx=constant.EOS_TOKEN)
        self_attn_mask_subseq = get_subsequent_mask(seq_in_pad)
        self_attn_mask_keypad = get_attn_key_pad_mask(
            seq_k=seq_in_pad, seq_q=seq_in_pad, pad_idx=constant.EOS_TOKEN)
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0)

        output_length = seq_in_pad.size(1)
        dec_enc_attn_mask = get_attn_pad_mask(
            encoder_padded_outputs, encoder_input_lengths, output_length)
        
        decoder_output = self.dropout(self.trg_embedding(
            seq_in_pad) * self.x_logit_scale + self.positional_encoding(seq_in_pad))

        for layer in self.layers:
            decoder_output, decoder_self_attn, decoder_enc_attn = layer(
                decoder_output, encoder_padded_outputs, non_pad_mask=non_pad_mask, self_attn_mask=self_attn_mask, dec_enc_attn_mask=dec_enc_attn_mask)

            decoder_self_attn_list += [decoder_self_attn]
            decoder_encoder_attn_list += [decoder_enc_attn]

        final_decoded_output = []
        final_gold = []
        for i in range(self.num_lang):
            final_decoded_output.append([])
            final_gold.append([])

        for i in range(len(decoder_output)):
            if self.combine_decoder: # single decoder
                idx = 0
            else:
                idx = langs[i]
            if len(self.output_linears) == 1:
                idx = 0
            final_decoded_output[langs[i]].append(self.output_linears[idx](decoder_output[i].unsqueeze(0)).squeeze())
            final_gold[langs[i]].append(seq_out_pad[i])
        
        for i in range(len(final_decoded_output)):
            if len(final_decoded_output[i]) > 0:
               final_decoded_output[i] = torch.stack(final_decoded_output[i], dim=0)
            if len(final_gold[i]) > 0:
                final_gold[i] = torch.stack(final_gold[i], dim=0)

        return final_decoded_output, final_gold, decoder_self_attn_list, decoder_encoder_attn_list

    def post_process_hyp(self, hyp):
        """
        args: 
            hyp: list of hypothesis
        output:
            list of hypothesis (string)>
        """
        return "".join([self.src_id2label[int(x)] for x in hyp['yseq'][1:]])

    def greedy_search(self, encoder_padded_outputs, beam_width=2, lm_rescoring=False, lm=None, lm_weight=0.1, c_weight=1, start_token=constant.SOS_TOKEN, lang_id=0):
        """
        Greedy search, decode 1-best utterance
        args:
            encoder_padded_outputs: B x T x H
        output:
            batch_ids_nbest_hyps: list of nbest in ids (size B)
            batch_strs_nbest_hyps: list of nbest in strings (size B)
        """
        max_seq_len = self.trg_max_length
        
        ys = torch.ones(encoder_padded_outputs.size(0),1).fill_(start_token).long() # batch_size x 1
        if constant.args.cuda:
            ys = ys.cuda()

        decoded_words = []
        for t in range(300):
        # for t in range(max_seq_len):
            # print(t)
            # Prepare masks
            non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) # batch_size x t x 1
            self_attn_mask = get_subsequent_mask(ys) # batch_size x t x t

            decoder_output = self.dropout(self.trg_embedding(ys) * self.x_logit_scale 
                                        + self.positional_encoding(ys))

            for layer in self.layers:
                decoder_output, _, _ = layer(
                    decoder_output, encoder_padded_outputs,
                    non_pad_mask=non_pad_mask,
                    self_attn_mask=self_attn_mask,
                    dec_enc_attn_mask=None
                )

            final_decoded_output = []
            
            for i in range(len(decoder_output)):
                if len(self.output_linears) == 1:
                    final_decoded_output.append(self.output_linears[0](decoder_output[i]))
                else:
                    final_decoded_output.append(self.output_linears[lang_id](decoder_output[i]))
            prob = torch.stack(final_decoded_output, dim=0)
            # prob = self.output_linear(decoder_output) # batch_size x t x label_size
            
            # _, next_word = torch.max(prob[:, -1], dim=1)
            # decoded_words.append([constant.EOS_CHAR if ni.item() == constant.EOS_TOKEN else self.id2label[ni.item()] for ni in next_word.view(-1)])
            # next_word = next_word.unsqueeze(-1)

            # local_best_scores, local_best_ids = torch.topk(local_scores, beam_width, dim=1)

            if lm_rescoring:
                local_scores = F.log_softmax(prob, dim=1)
                local_best_scores, local_best_ids = torch.topk(local_scores, beam_width, dim=1)

                best_score = -1
                best_word = None

                # calculate beam scores
                for j in range(beam_width):
                    cur_seq = " ".join(word for word in decoded_words)
                    lm_score, num_words, oov_token = calculate_lm_score(cur_seq, lm, self.id2label)
                    score = local_best_scores[0, j] + lm_score
                    if best_score < score:
                        best_score = score
                        best_word = local_best_ids[0, j]
                        next_word = best_word
                        for k in range(len(next_word)):
                            if len(self.trg_label2ids) == 1:
                                next_word[k] = torch.LongTensor([self.src_label2id[self.trg_id2labels[0][next_word[k].item()]]]).cuda()
                            else:
                                next_word[k] = torch.LongTensor([self.src_label2id[self.trg_id2labels[lang_id][next_word[k].item()]]]).cuda()
                        next_word = next_word.unsqueeze(-1)
                if len(self.trg_label2ids) == 1:
                    decoded_words.append(self.trg_id2labels[0][int(best_word)])
                else:
                    decoded_words.append(self.trg_id2labels[lang_id][int(best_word)])
            else:
                _, next_word = torch.max(prob[:, -1], dim=1)
                if len(self.trg_id2labels) == 1:
                    decoded_words.append([constant.EOS_CHAR if ni.item() == constant.EOS_TOKEN else self.trg_id2labels[0][ni.item()] for ni in next_word.view(-1)])
                else:
                    decoded_words.append([constant.EOS_CHAR if ni.item() == constant.EOS_TOKEN else self.trg_id2labels[lang_id][ni.item()] for ni in next_word.view(-1)])
                # convert target index to source index
                for k in range(len(next_word)):
                    if len(self.trg_id2labels) == 1:
                        next_word[k] = torch.LongTensor([self.src_label2id[self.trg_id2labels[0][next_word[k].item()]]]).cuda()
                    else:
                        next_word[k] = torch.LongTensor([self.src_label2id[self.trg_id2labels[lang_id][next_word[k].item()]]]).cuda()
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

    def beam_search(self, encoder_padded_outputs, beam_width=2, nbest=5, lm_rescoring=False, lm=None, lm_weight=0.1, c_weight=1, prob_weight=1.0, start_token=constant.SOS_TOKEN, lang_id=0):
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

        for x in range(batch_size):
            encoder_output = encoder_padded_outputs[x].unsqueeze(0) # 1 x T x H

            # add SOS_TOKEN
            ys = torch.ones(1, 1).fill_(start_token).type_as(encoder_output).long()
            
            hyp = {'score': 0.0, 'yseq':ys}
            hyps = [hyp]
            ended_hyps = []

            for i in range(300):
            # for i in range(self.trg_max_length):
                hyps_best_kept = []
                for hyp in hyps:
                    ys = hyp['yseq'] # 1 x i

                    # Prepare masks
                    non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) # 1xix1
                    self_attn_mask = get_subsequent_mask(ys)

                    decoder_output = self.dropout(self.trg_embedding(ys) * self.x_logit_scale 
                                                + self.positional_encoding(ys))

                    for layer in self.layers:
                        decoder_output, _, _ = layer(
                            decoder_output, encoder_output,
                            non_pad_mask=non_pad_mask,
                            self_attn_mask=self_attn_mask,
                            dec_enc_attn_mask=None
                        )

                    if len(self.output_linears) == 1:
                        seq_logit = self.output_linears[0](decoder_output[:, -1])
                    else:
                        seq_logit = self.output_linears[lang_id](decoder_output[:, -1])
                    local_scores = F.log_softmax(seq_logit, dim=1)
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam_width, dim=1)

                    # calculate beam scores
                    for j in range(beam_width):
                        new_hyp = {}
                        new_hyp["score"] = hyp["score"] + local_best_scores[0, j]

                        new_hyp["yseq"] = torch.ones(1, (1+ys.size(1))).type_as(encoder_output).long()
                        new_hyp["yseq"][:, :ys.size(1)] = hyp["yseq"].cpu()
                        new_word = int(local_best_ids[0, j])

                        # convert target index to source index
                        if len(self.trg_id2labels) == 1:
                            new_word = torch.LongTensor([self.src_label2id[self.trg_id2labels[0][new_word]]]).cuda()
                        else:
                            new_word = torch.LongTensor([self.src_label2id[self.trg_id2labels[lang_id][new_word]]]).cuda()
                        new_hyp["yseq"][:, ys.size(1)] = new_word # adding new word
                        
                        hyps_best_kept.append(new_hyp)

                    hyps_best_kept = sorted(hyps_best_kept, key=lambda x:x["score"], reverse=True)[:beam_width]
                
                hyps = hyps_best_kept

                # add EOS_TOKEN
                if i == max_len - 1:
                    for hyp in hyps:
                        hyp["yseq"] = torch.cat([hyp["yseq"], torch.ones(1,1).fill_(constant.EOS_TOKEN).type_as(encoder_output).long()], dim=1)

                # add hypothesis that have EOS_TOKEN to ended_hyps list
                unended_hyps = []
                for hyp in hyps:
                    if hyp["yseq"][0, -1] == constant.EOS_TOKEN:
                        if lm_rescoring:
                            # seq_str = "".join(self.id2label[char.item()] for char in hyp["yseq"][0]).replace(constant.PAD_CHAR,"").replace(constant.SOS_CHAR,"").replace(constant.EOS_CHAR,"")
                            # seq_str = seq_str.replace("  ", " ")
                            # num_words = len(seq_str.split())

                            hyp["lm_score"], hyp["num_words"], oov_token = calculate_lm_score(hyp["yseq"], lm, self.id2label)
                            num_words = hyp["num_words"]
                            hyp["lm_score"] -= oov_token * 2
                            hyp["final_score"] = hyp["score"] + lm_weight * hyp["lm_score"] + math.sqrt(num_words) * c_weight
                        else:
                            seq_str = "".join(self.src_id2label[char.item()] for char in hyp["yseq"][0]).replace(constant.PAD_CHAR,"").replace(constant.SOS_CHAR,"").replace(constant.EOS_CHAR,"")
                            seq_str = seq_str.replace("  ", " ")
                            num_words = len(seq_str.split())
                            hyp["final_score"] = hyp["score"] + math.sqrt(num_words) * c_weight
                        
                        ended_hyps.append(hyp)
                        
                    else:
                        unended_hyps.append(hyp)
                hyps = unended_hyps

                if len(hyps) == 0:
                    # decoding process is finished
                    break
                
            num_nbest = min(len(ended_hyps), nbest)
            nbest_hyps = sorted(ended_hyps, key=lambda x:x["final_score"], reverse=True)[:num_nbest]

            a_nbest_hyps = sorted(ended_hyps, key=lambda x:x["final_score"], reverse=True)[:beam_width]

            if lm_rescoring:
                for hyp in a_nbest_hyps:
                    seq_str = "".join(self.src_id2label[char.item()] for char in hyp["yseq"][0]).replace(constant.PAD_CHAR,"").replace(constant.SOS_CHAR,"").replace(constant.EOS_CHAR,"")
                    seq_str = seq_str.replace("  ", " ")
                    num_words = len(seq_str.split())
                    # print("{}  || final:{} e2e:{} lm:{} num words:{}".format(seq_str, hyp["final_score"], hyp["score"], hyp["lm_score"], hyp["num_words"]))

            for hyp in nbest_hyps:                
                hyp["yseq"] = hyp["yseq"][0].cpu().numpy().tolist()
                hyp_strs = self.post_process_hyp(hyp)

                batch_ids_nbest_hyps.append(hyp["yseq"])
                batch_strs_nbest_hyps.append(hyp_strs)
                # print(hyp["yseq"], hyp_strs)
        return batch_ids_nbest_hyps, batch_strs_nbest_hyps

class DecoderLayer(nn.Module):
    """
    Decoder Transformer class
    """

    def __init__(self, dim_model, dim_inner, num_heads, dim_key, dim_value, dropout=0.1, is_factorized=False, r=100):
        super(DecoderLayer, self).__init__()
        self.is_factorized = is_factorized
        self.r = r
        self.self_attn = FactorizedMultiHeadAttention(
            num_heads, dim_model, dim_key, dim_value, dropout=dropout, r=r)
        self.encoder_attn = FactorizedMultiHeadAttention(
            num_heads, dim_model, dim_key, dim_value, dropout=dropout, r=r)
        if is_factorized:
            self.pos_ffn = FactorizedPositionwiseFeedForward(dim_model, dim_inner, dropout=dropout, r=r)
        else:
            self.pos_ffn = PositionwiseFeedForward(dim_model, dim_inner, dropout=dropout)

    def forward(self, decoder_input, encoder_output, non_pad_mask=None, self_attn_mask=None, dec_enc_attn_mask=None):
        decoder_output, decoder_self_attn = self.self_attn(
            decoder_input, decoder_input, decoder_input, mask=self_attn_mask)
        decoder_output *= non_pad_mask

        decoder_output, decoder_encoder_attn = self.encoder_attn(
            decoder_output, encoder_output, encoder_output, mask=dec_enc_attn_mask)
        decoder_output *= non_pad_mask

        decoder_output = self.pos_ffn(decoder_output)
        decoder_output *= non_pad_mask

        return decoder_output, decoder_self_attn, decoder_encoder_attn        