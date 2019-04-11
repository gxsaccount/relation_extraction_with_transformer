import numpy as np 
from transformer.Models import Transformer
import torch
import time
from tqdm import tqdm
import transformer.Constants as Constants
import data
import torch.optim as optim
from transformer.Optim import ScheduledOptim
import math

class Config():
    def __init__(self):
        self.batch_size=128
        self.num_class=50
        self.max_seq_len=198##根据语料设定
        self.dim_pos=10
        self.d_word_vec=32
        self.d_model=32
        self.d_inner=256
        self.n_layers=1#层数
        self.n_head=2##多头注意力要复制几次
        self.d_k=8
        self.d_v=8
        self.dropout=0.1
        self.out_channels=16 ##
        self.kernel_sizes = [3,5,7]
config = Config()


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''
    lossf = torch.nn.CrossEntropyLoss()##cal_loss(pred, gold, smoothing)
    gold = gold.contiguous().view(-1)
    loss = lossf(pred,gold)
    non_pad_mask = gold.ne(Constants.PAD)
    pred = pred.max(1)[1]
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return loss, n_correct


def prepare_dataloaders(word2idx,ints,en1_pos,en2_pos,predicates):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        data.Dataset(
            word2idx=word2idx,
            insts=ints,
            en1_pos=en1_pos,
            en2_pos=en2_pos,
            predicates=predicates),
        batch_size=128,
        collate_fn=collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        data.Dataset(
            word2idx=word2idx,
            insts=ints,
            en1_pos=en1_pos,
            en2_pos=en2_pos,
            predicates=predicates),
        batch_size=128,
        collate_fn=collate_fn)
    return train_loader, valid_loader

def collate_fn(items):
    ''' Pad the instance to the max seq length in batch '''
    max_len = max(len(inst[0]) for inst in items)
    batch_seq = np.array([
        (inst[0] + [Constants.PAD] * (max_len - len(inst[0])))
        for inst in items])
    batch_abs_pos =  np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])
    batch_rel_pos_en1 =[]
    for i in range(len(batch_abs_pos)):
        en1 = items[i][1]
        pos=batch_abs_pos[i]
        batch_rel_pos_en1.append([
            pos_i+config.max_seq_len-i+1 if pos_i != 0 else 0 for pos_i in pos ])
    batch_rel_pos_en1 = np.array(batch_rel_pos_en1)
    batch_rel_pos_en2 =[]
    for i in range(len(batch_abs_pos)):
        en2 = items[i][2]
        pos=batch_abs_pos[i]
        batch_rel_pos_en2.append([
            pos_i+config.max_seq_len-i+1 if pos_i != 0 else 0 for pos_i in pos ])
    batch_rel_pos_en2 = np.array(batch_rel_pos_en2)
    batch_pred= np.array([pre[3] for pre in items])
    batch_seq = torch.LongTensor(batch_seq)
    batch_abs_pos = torch.LongTensor(np.array(batch_abs_pos))
    batch_pred = torch.LongTensor(batch_pred)
    batch_rel_pos_en1 =torch.LongTensor(batch_rel_pos_en1)
    batch_rel_pos_en2 = torch.LongTensor(batch_rel_pos_en2)
    return batch_seq, batch_abs_pos,batch_rel_pos_en1,batch_rel_pos_en2,batch_pred

def train_epoch(model, training_data, optimizer, predicates, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
        # prepare data
        src_seq, src_pos,src_pos1,src_pos2,src_pred = map(lambda x: x.to(device), batch)
        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos,src_pos1,src_pos2)
        # backward
        loss, n_correct = cal_performance(pred,src_pred, smoothing=smoothing)
        loss.backward()
        # update parameters
        optimizer.step_and_update_lr()
        # note keeping
        total_loss += loss.item()
        non_pad_mask = src_pred.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy
    
def eval_epoch(model, validation_data,predicates):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Training)   ', leave=False):
            # prepare data
            src_seq, src_pos,src_pos1,src_pos2,src_pred = map(lambda x: x.to(device), batch)

            # forward
            pred = model(src_seq, src_pos,src_pos1,src_pos2)

            # backward
            loss, n_correct = cal_performance(pred,src_pred)

            # update parameters

            # note keeping
            total_loss += loss.item()
            non_pad_mask = src_pred.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, training_data, validation_data, optimizer,predicates):
    valid_accus = []
    for epoch_i in range(10):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, predicates, smoothing='store_true')
        print('  - (Training)   accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, predicates)
        print('  - (Validation)  accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

device = torch.device('cpu')


word2idx,ints,en1_pos,en2_pos,predicates,relation2idx = data.build_sentences()

training_data, validation_data = prepare_dataloaders(word2idx,ints,en1_pos,en2_pos,predicates)
model = Transformer(
    n_src_vocab=len(word2idx),
    len_max_seq=config.max_seq_len).to(device)

optimizer = ScheduledOptim(
    optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        betas=(0.9, 0.98), eps=1e-09),
    512, 1000)

train(model, training_data, validation_data, optimizer,predicates)