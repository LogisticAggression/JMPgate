
import random

import math
import six
import sys
import time
import h5py
import os
import pickle
import json
import base64

import numpy as np

import chainer
from chainer.initializers import Constant, Orthogonal, GlorotUniform
from chainer import cuda, Function, gradient_check, Variable, optimizers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from chainer import serializer
from chainer import serializers

import argparse

from tqdm import tqdm

EOF = 256

def getSeq(path):
    if type(path) == np.ndarray:  #
        return path  # its already in memory as a numpy ndarray
    else:  # assume its a full path and load it
        seq_list = []
        with open(path, "r") as f:
            # f_bytes = f.read()
            buff = json.load(f)
            # print buff
            raw_bytes = base64.b64decode(buff["Normalized_Code"])
            for func in buff["Funcs"]:
                start = buff["Funcs"][func][0]
                end = buff["Funcs"][func][0] + buff["Funcs"][func][1]
                f_bytes = raw_bytes[start:end]
                data = load_sequence(f_bytes)
                seq_list.append(data)
        return seq_list

def getShuffledChunks(l, b):
    chunks = []
    for i in range(0, len(l), b):
        chunks.append( l[i:min(i+b, len(l))] )
    random.shuffle(chunks)
    return chunks

def getByte(seq, index, EOF=EOF, marker=-1):
    if index == len(seq):
        return EOF
    elif index < len(seq):
        return seq[index]
    else:
        return marker

def load_sequence(seq):
    buffer = np.zeros(len(seq), dtype=np.int32)
    for i, token in enumerate(seq):
        buffer[i] = token
    return buffer

class RNN(chainer.Chain):

    def __init__(self, n_vocab, n_embed, n_units, n_lstm_layers):
        super(RNN, self).__init__(
            embed=L.EmbedID(n_vocab, n_embed),
            lstms=chainer.ChainList(),
            out=L.Linear(n_units, n_vocab)
        )

        self.lstms.add_link(L.LSTM(n_embed, n_units, lateral_init=Orthogonal(),
                                   upward_init=GlorotUniform(), forget_bias_init=Constant(5)))
        for _ in range(n_lstm_layers-1):
            self.lstms.add_link(L.LSTM(n_units, n_units, lateral_init=Orthogonal(),
                                       upward_init=GlorotUniform(), forget_bias_init=Constant(5)))

    def reset_state(self):
        for lstm in self.lstms:
            lstm.reset_state()

    def forward(self, x):
        h0 = self.embed(x)
        hs = [h0]
        for lstm in self.lstms:
            h_i = lstm(F.dropout(hs[-1]))
            hs.append(h_i)
        y = self.out(F.dropout(hs[-1]))
        return y


def train(seqs, num_iters):
    accum_loss = 0
    cur_log_perp = xp.zeros(())
    for epoch in range(num_iters):
        #batchs wont have even length, so count invidualy
        log_perps = 0

        random.shuffle(order)
        for batch in tqdm(getShuffledChunks(order_by_size, batchsize)):
            seq_batch = [seqs[i] for i in batch]

            max_len = max([s.shape[0] for s in seq_batch])

            batch_log_perp = xp.zeros(())
            batch_log_perps = 0
            #start working on this batch
            model.predictor.reset_state()

            for t in range(max_len):
                #for each item in batch, we get the value at t, or -1 if one of the seqsuences it out of items
                #we use -1 for too many b/c loss will ignore -1 targets in loss computation
                #use EOF instead of -1 for input b/c we still need to push it through our network, and -1 would err as input
                x_raw = [getByte(s, t, marker=EOF)  for s in seq_batch ]
                y_raw = [getByte(s, t+1, marker=-1)  for s in seq_batch ]

                x = chainer.Variable(xp.asarray(x_raw, dtype=np.int32))
                y = chainer.Variable(xp.asarray(y_raw, dtype=np.int32))

                loss_i = model(x, y)
                accum_loss += loss_i
                batch_log_perp += loss_i.data
                batch_log_perps += y_raw.count(-1)

                if (t + 1) % bprop_len == 0 or t == (max_len-1):  # Run truncated BPTT
                    model.cleargrads()
                    accum_loss.backward()
                    accum_loss.unchain_backward()  # truncate
                    accum_loss = 0
                    optimizer.update()
            #end of processing this batch
            cur_log_perp += batch_log_perp
            log_perps += batch_log_perps
        #end of all batches in epoch
        print("Epoch {}, perp: {}".format(epoch, cur_log_perp/log_perps))
        #logFile.write("Epoch {}, perp: {}\n".format(epoch, cur_log_perp/log_perps))

        #if epoch % 10 == 0:
        #serializers.save_hdf5("/home/rob/"+modelName+"-{}.model".format(epoch), model)
        #serializers.save_hdf5("/home/rob/x86/"+modelName+"-{}.optimizer".format(epoch), optimizer)
        cur_log_perp.fill(0)
        log_perps = 0        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-file', '-t', default='data',
                        help='File containing sequences')
    parser.add_argument('--pre-load', '-pl', default=False, type=bool,
                        help='Whether or not to read all data into memory or load from disk as needed')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--start-epoch-from', '-a', default=0, type=int,
                        help='Starts the epoch from the given number (starts from 0)')
    parser.add_argument('--num-lstm-layers', '-L', default=1, type=int,
                        help='The number of LSTM layers to use')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batch', '-b', default=128, type=int,
                        help='Batch Size for training')
    parser.add_argument('--epochs', '-e', default=10, type=int,
                        help='Number of training epochs')
    parser.add_argument('--bprop-len', '-bl', default=100, type=int,
                        help='The number of steps to take before truncating the backprop chain')
    parser.add_argument('--reset-state-every', '-rs', default=10000, type=int,
                        help='The number of steps to take before reseting the hidden state of the LSTM')
    parser.add_argument('--grad-clip', '-gc', default=1.0, type=float,
                        help='Norm for gradient clipping')
    parser.add_argument('--hard-clip', '-gh', default=False, type=bool,
                        help='Whether or not to perform hard clipping or L2 norm rescaling')
    parser.add_argument('--lstm-layer-size', '-s', default=100, type=int,
                        help='Number of hidden units for each layer of the LSTM')
    parser.add_argument('--embed-size', '-es', default=25, type=int,
                        help='Size of the embedding layer')
    parser.add_argument('--adam-alpha', '-aa', default=0.001, type=float,
                        help='The learning rate used for Adam')
    parser.add_argument('--adam-beta1', '-ab', default=0.9, type=float,
                        help='The momentum term for Adam')

    args = parser.parse_args()

    num_layers = args.num_lstm_layers
    n_epoch = args.epochs      # 10 # number of epochs
    batchsize = args.batch     # 128   # minibatch size
    bprop_len = args.bprop_len # 100   # length of truncated BPTT
    reset_state_every = args.reset_state_every   #  10000//bprop_len
    grad_clip = args.grad_clip # 20    # gradient norm threshold to clip
    tokens = 256
    lstm_layer_size = args.lstm_layer_size # 2048
    embed_size = args.embed_size # 512
    adam_alpha = args.adam_alpha # 0.001
    adam_beta1 = args.adam_beta1 # 0.9
    useGPU = args.gpu
    xp = cuda.cupy if useGPU >= 0 else np

    print(args)

    seqs = []
    seqSizeMap = {}  # maps from a size in bytes to a list of every sequence that has that length
    
    print("Loading sequences...")

    #seqFile = args.sequence_dir + '/' + seqFile
    newseqs = [load_sequence(line) for line in open(args.sequence_file,'rb')]
    for s in newseqs:
        length = len(s)
        indx = len(seqs)
        seqs.append(s)
        if length in seqSizeMap:
            seqSizeMap[length].append(indx)
        else:
            seqSizeMap[length] = [indx]

    #index into seqs by file size
    order_by_size = []
    for length in sorted(seqSizeMap.keys()):
        for indx in seqSizeMap[length]:
            order_by_size.append(indx)


    order_by_size = sorted(order_by_size, reverse=True)


    lm = RNN(257, embed_size, lstm_layer_size, num_layers)
    model = L.Classifier(lm)
    model.compute_accuracy = False  # we only want the perplexity

    if useGPU >= 0:
        cuda.get_device(useGPU).use()
        model.to_gpu()

    # Setup optimizer
    optimizer = optimizers.Adam(alpha=adam_alpha, beta1=adam_beta1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

    n_seqs = len(seqs)
    jump = n_seqs // batchsize

    epoch = args.start_epoch_from

    batch_idxs = list(range(batchsize))
    print("going to start from epoch {}".format(epoch))

    start_at = time.time()
    cur_at = start_at
    epoch_start_time = start_at

    order = [ i for i in range(n_seqs)]

    train(seqs, 10)

    #end of epochs



