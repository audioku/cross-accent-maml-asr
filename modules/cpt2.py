from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import json
import logging
import os
import regex as re
from io import open
import pickle
import six
import math

import collections
import logging
import os
import unicodedata
from io import open

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from transformers.modeling_gpt2 import GPT2LMHeadModel, GPT2Model, MLP, gelu
from transformers.modeling_bert import BertModel, BertEmbeddings

class ContextualAttention(nn.Module):
    @classmethod
    def _get_future_mask(cls, size, device):
        if not hasattr(cls, '_future_mask') or cls._future_mask.device != device or cls._future_mask.shape < size:
            cls._future_mask = torch.triu(torch.ones(size[0], size[1], dtype=torch.uint8, device=device), 1)
        mask = cls._future_mask[:size[0], :size[1]]
        return mask
    
    def __init__(self, nx, n_ctx, config, scale=False):
        super(ContextualAttention, self).__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2*self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, apply_future_mask=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        
        if attention_mask is not None:
            # Apply the attention mask
            # w = w + attention_mask
            w.masked_fill_(attention_mask, float('-inf'))

        if apply_future_mask:
            future_mask = ContextualAttention._get_future_mask(w.shape[-2:], w.device).unsqueeze(0).unsqueeze(0).bool()
            w.masked_fill_(future_mask, float('-inf'))

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, q, k, v, layer_past=None, attention_mask=None, head_mask=None):
        qkv_same = (q.data_ptr() == k.data_ptr() == v.data_ptr())
        kv_same = (k.data_ptr() == v.data_ptr())

        if qkv_same:
            # Project q to q, k, v
            x = self.c_attn(q)
            query, key, value = x.split(self.split_size, dim=2)
            apply_future_mask = True  # self-attention
        elif kv_same:
            # Perform linear on q, k, v independently
            q_w, q_b = self.c_attn.weight[:, :self.split_size], self.c_attn.bias[:self.split_size]
            query = F.linear(q, q_w.t(), q_b) # We need to transpose the weight because of Conv1D
            kv_w, kv_b = self.c_attn.weight[:, self.split_size:], self.c_attn.bias[self.split_size:]
            key, value = F.linear(k, kv_w.t(), kv_b).split(self.split_size, dim=-1) # We need to transpose the weight because of Conv1D
            apply_future_mask = False # non self-attention
        else:
            assert False
            
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
            
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, apply_future_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)

# class MLP(nn.Module):
#     def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
#         super(MLP, self).__init__()
#         nx = config.n_embd
#         self.c_fc = Conv1D(n_state, nx)
#         self.c_proj = Conv1D(nx, n_state)
#         self.act = gelu
#         self.dropout = nn.Dropout(config.resid_pdrop)

#     def forward(self, x):
#         h = self.act(self.c_fc(x))
#         h2 = self.c_proj(h)
#         return self.dropout(h2)

class ContextualBlock(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(ContextualBlock, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = ContextualAttention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, q, k=None, v=None, layer_past=None, attention_mask=None, head_mask=None):
        q_norm = self.ln_1(q)
        if k is None or v is None:
            output_attn = self.attn(q_norm, q_norm, q_norm,
                                    layer_past=layer_past,
                                    attention_mask=attention_mask,
                                    head_mask=head_mask)
        else:
            output_attn = self.attn(q_norm, k, v,
                                    layer_past=layer_past,
                                    attention_mask=attention_mask,
                                    head_mask=head_mask)
        a = output_attn[0]  # output_attn: a, present, (attentions)

        q = q + a
        m = self.mlp(self.ln_2(q))
        q = q + m

        outputs = [q] + output_attn[1:]
        return outputs  # x, present, (attentions)

class CPT2Model(GPT2Model):
    def __init__(self, config, mha_block):
        super(CPT2Model, self).__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.output_past = config.output_past
        self.mha_block = mha_block

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([ContextualBlock(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        self.wte = self._get_resized_embeddings(self.wte, new_num_tokens)
        return self.wte

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(self, input_ids, key=None, value=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        # TODO: Configure / specify layer for injecting key & value (MHA)
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            if self.mha_block[i] == 0:
                outputs = block(hidden_states, None, None,
                                layer_past=layer_past,
                                attention_mask=attention_mask,
                                head_mask=head_mask[i])
            else:
                outputs = block(hidden_states, key, value,
                                layer_past=layer_past,
                                attention_mask=attention_mask,
                                head_mask=head_mask[i])

            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)
                                  
class CPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super(CPT2LMHeadModel, self).__init__(config)
        self.transformer = CPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size

        self.init_weights()
        self.tie_weights()

    def extend_embedding(self, embedding_layer):
        vocab_size = self.vocab_size + embedding_layer.weight.shape[0]
        en_weights = self.transformer.wte.weight
        cn_weights = embedding_layer.weight
        
        self.lm_head = nn.Linear(self.n_embd, vocab_size, bias=False)
        self.transformer.wte = nn.Embedding(vocab_size, self.n_embd)
        self.transformer.wte.weight.data[:self.vocab_size,:] = en_weights
        self.transformer.wte.weight.data[self.vocab_size:,:] = cn_weights
        
        self.tie_weights()
        
        print('en_weights', en_weights[0,:10])
        print('wte_en_0', self.transformer.wte.weight[0,:10])
        print('lm_head_en_0', self.lm_head.weight[0,:10])

        print('cn_weights', cn_weights[0,:10])
        print('wte_cn_0', self.transformer.wte.weight[self.vocab_size,:10])
        print('lm_head_cn_0', self.lm_head.weight[self.vocab_size,:10])

    def forward(self, input_ids, key=None, value=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        transformer_outputs = self.transformer(input_ids, key, value,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

if __name__ == '__main__':
    from attrdict import AttrDict
    nx = 8
    n_ctx = 512
    config = AttrDict({
        'output_attentions': False,
        'n_head' : 2,
        'attn_pdrop' : 0,
        'resid_pdrop' : 0
    })
    
    q = torch.randn((1,3,8))
    k = torch.randn((1,4,8))
    v = torch.randn((1,4,8))
    
    attn = ContextualAttention(nx, n_ctx, config, scale=False)
    # Test assert q k v different
    # y = attn(q,k,v)
    
    # Test self attention
    print('Test self-attention')
    y = attn(q,q,q)
    print('y', y)
    
    # Test non self-attention
    print('Test non self-attention')
    y = attn(q,v,v)
    print('y', y)
    
    # Test CPT2
    q = torch.randint(0, 2, (8,4))
    y = torch.randint(0, 2, (8,4)).long()
    k = torch.randn((8, 4, 768))
    
    # Test self attention
    print('Test CPT2 self-attention')
    cpt2 = CPT2LMHeadModel.from_pretrained('distilgpt2')
    optimizer = torch.optim.Adam(cpt2.parameters())
    for i in range(3):
        result = cpt2(q, labels=y)
        loss = result[0]
        
        print('loss',loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Test non self-attention
    print('Test CPT2 non self-attention')
    cpt2 = CPT2LMHeadModel.from_pretrained('distilgpt2')
    optimizer = torch.optim.Adam(cpt2.parameters())
    for i in range(3):
        result = cpt2(q, k, k,labels=y)
        loss = result[0]
        
        print('loss',loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Test BERT Embeddine
    print('=BEFORE=')
    print('cpt2.lm_head.weight.shape', cpt2.lm_head.weight.shape)
    print('self.transformer.wte.weight.shape', cpt2.transformer.wte.weight.shape)
    
    cpt2 = CPT2LMHeadModel.from_pretrained('distilgpt2')
    bert_model = BertModel.from_pretrained('bert-base-chinese')
    bert_word_embedding = bert_model.embeddings.word_embeddings
    cpt2.extend_embedding(bert_word_embedding)
    
    print('=AFTER=')
    print('cpt2.lm_head.weight.shape', cpt2.lm_head.weight.shape)
    print('self.transformer.wte.weight.shape', cpt2.transformer.wte.weight.shape)
    
