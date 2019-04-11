import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Xiang Gao"

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_relative_position_table(dim,max_position):
    pass

def get_absolute_position_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        
        self.absolute_position_enc = nn.Embedding.from_pretrained(
            get_absolute_position_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):
        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.absolute_position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        
        self.absolute_position_enc = nn.Embedding.from_pretrained(
            get_absolute_position_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.absolute_position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class CCN(nn.Module):
    def __init__(self,len_max_seq,kernel_sizes,d_position,in_channels,out_channels,num_class): 
        super().__init__()
        self.relative_position = nn.Embedding(
            len_max_seq*2+1,d_position,padding_idx=Constants.PAD)
        self.cnn1ds = []
        for i in range(len(kernel_sizes)):
            cnn1d = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_sizes[i]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=len_max_seq-kernel_sizes[i]+1))
            self.cnn1ds.append(cnn1d)
        self.linear = nn.Linear(in_features=out_channels*len(kernel_sizes),
                            out_features=num_class)

    def forward(self, dec_output,pos1s,pos2s):
        try:
            en1_positions=self.relative_position(pos1s)
        except:
            print(pos1s)
        en2_positions=self.relative_position(pos2s)
        dec_output=torch.cat([dec_output,en1_positions,en2_positions],dim=2)
        dec_output = dec_output.permute(0, 2, 1)
        outs = [conv(dec_output) for conv in self.cnn1ds]
        outs = torch.cat(outs, dim=1) 
        outs = outs.view(-1, outs.size(1)) 
        outs = self.linear(outs)
        return outs


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab , len_max_seq=10,
            d_word_vec=32, d_model=32, d_inner=256,
            n_layers=1, n_head=2, d_k=8, d_v=8, dropout=0.1,d_position=10,
            out_channels=16,num_class=50,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        # self.decoder = Decoder(
        #     n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
        #     d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
        #     n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
        #     dropout=dropout)

        self.cnn = CCN(
            in_channels=d_word_vec+d_position*2,
            len_max_seq=len_max_seq,
            kernel_sizes=kernel_sizes,
            d_position=d_position,
            out_channels=out_channels,
            num_class=num_class)


    def forward(self, src_seq, src_pos,pos1s,pos2s):
        enc_output, *_ = self.encoder(src_seq, src_pos)
        reslut = self.cnn(enc_output,pos1s,pos2s)
        return reslut 



