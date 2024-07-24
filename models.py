import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv 
from dgi_models import DGIforSAGE, DGIforGR
from node2vec import Node2Vec
import os.path as osp
import pandas as pd
import numpy as np
import collections
from pandas.core.common import flatten
import networkx as nx
import math
from copy import deepcopy
from args import *
from transformers import pipeline

args = make_args()

torch.autograd.set_detect_anomaly(True)

class SimpleNeuralNets(nn.Module):
    '''
    This is specifically for LLM experiments
    '''
    def __init__(self, device, num_classes, embed_dim = args.in_size, hidden_dim = args.hidden_size1, dropout = 0.1):
        super(SimpleNeuralNets, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.linear_author = nn.Linear(embed_dim, hidden_dim)
        self.linear_sections = nn.Linear(embed_dim, hidden_dim)
        self.linear_combine = nn.Linear(hidden_dim * 2 , hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, num_classes) # 2 for author and sections input linears

        self.dropout_layer = nn.Dropout(dropout) # will be disabled in model.eval()

    def forward(self, author_batch, section_batch, author_features, section_features):
        batch_representation = []
        for i in range(len(author_batch)):
            curr_author = author_batch[i]
            curr_section = section_batch[i]
            author_embedding = author_features[curr_author]
            section_embeddings = section_features[curr_section]
            section_embeddings_mean = torch.mean(section_embeddings, dim = 0, keepdim=True)

            x_author = F.relu(self.linear_author(author_embedding))
            x_author = self.dropout_layer(x_author)
            x_sections = F.relu(self.linear_sections(section_embeddings_mean))
            x_sections = self.dropout_layer(x_sections)
            combine_x = torch.cat((x_author, section_embeddings_mean), dim = 1)
            batch_representation.append(combine_x)

        combined_rep = self.linear_combine(
            torch.stack(batch_representation).squeeze()
        )
        combined_rep = self.dropout_layer(combined_rep)

        out = F.relu(self.linear_out(combined_rep))
        if len(out.shape) == 1:
            out = out.unsqueeze(0)

        return out

class MultiHeadAttention(nn.Module):
    '''
    This multi-head attention is exclusively for transformer model
    code is copied from https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
    '''
    def __init__(self, embed_dim = args.in_size, n_heads = 4, dropout = 0.1):
        '''
        embed_dim : the dimension of embedding vector output
        n_heads : number of self attention heads, set to 4 by default
        '''
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads)

        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias = False)
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias = False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

    def forward(self, key, query, value, mask = None):
        batch_size = key.size(0)
        seq_length = key.size(1)
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose (1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_adjusted = k.transpose(-1, -2)
        product = torch.matmul(q, k_adjusted)

        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        product = product / math.sqrt(self.single_head_dim)
        scores = F.softmax(product, dim = -1)
        scores = torch.mathmul(scores, v)
        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)
        output = self.out(concat)

        return output

class MHALayer(nn.Module):
    def __init__(self, node_emb_dim, hidden_dim1, hidden_dim2, n_head, device, dropout = 0.1):
        super(MHALayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(node_emb_dim, n_head, dropout = dropout)
        self.hidden_query = nn.Linear(hidden_dim1, hidden_dim2) # for query
        self.hidden_key = nn.Linear(hidden_dim1, hidden_dim2) # for key
        self.hidden_value = nn.Linear(hidden_dim1, hidden_dim2) # for value
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim2, hidden_dim2)

    def forward(self, gru_embedding):
        '''
        Query q = embedding dari GRU, dimension [batch_size, longest_sequence_len, embedding_dimension]
        Key k = semacam weight, learnable, transform
        Value v = semacam weight, learnable, samaain sama q
        '''
        gru_embedding = gru_embedding.to(self.device) # torch.Size([1, 384])
        q = self.hidden_query(gru_embedding)
        k = self.hidden_key(gru_embedding)
        v = self.hidden_value(gru_embedding)
        out, _ = self.multihead_attn(q, k, v)

        q = q.to(self.device)
        k = k.to(self.device)
        v = v.to(self.device)
        out = out.to(self.device)

        return out

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, device, max_seq_len = 80, dropout = 0.1):
        '''
        > d_model = dimension of the embeddings used for the model
        code copied and modified from https://blog.floydhub.com/the-transformer-in-pytorch/
        '''
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        pe = torch.squeeze(self.pe)
        pe = torch.unsqueeze(torch.mean(pe, dim = 0), 0)
        x = x + Variable(pe, requires_grad=False).to(self.device)
        return self.dropout(x)
    '''
    Simple feedforward used to create section position embedding.
    Reference tutorial: https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb
    For each paper, there exist several sections, this section embedding is used for additional information for the classification model regarding in which section the author is currently writing their paper.
    > d_model = dimension of the embeddings used for the model
    '''
    def __init__(self, d_model, device, max_seq_len, dropout = 0.1):
        super().__init__()
        self.input_size = d_model + max_seq_len
        self.max_seq_len = max_seq_len
        self.device = device
        self.hidden_size = d_model 
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, sequence_rep):
        '''
        > x = paper embedding
        > sequence_rep = sequence representation of the current paper embedding
        '''
        inpt = []
        for n in range(len(sequence_rep)):
            one_hot_encoding = np.zeros(self.max_seq_len)
            section_index = sequence_rep[n] - 1
            try:
                one_hot_encoding[section_index] = 1
            except IndexError:
                print('INDEX ERROR!')
                exit()
            curr_seq_rep = torch.from_numpy(one_hot_encoding).float()
            # print(curr_seq_rep, curr_seq_rep.shape)
            curr_seq_rep = curr_seq_rep.to(self.device)
            curr_x = x[n]
            curr_x = curr_x.to(self.device)
            inpt.append(torch.cat((curr_seq_rep, curr_x), 0))
        inpt = torch.stack(inpt)
        inpt = inpt.to(self.device)
        hidden = self.fc1(inpt)
        hidden = hidden.to(self.device)
        relu = self.relu(hidden)
        relu = relu.to(self.device)
        out = self.fc2(relu)
        # out = self.sigmoid(out)
        out = out.to(self.device)

        return out

class GRUModel(nn.Module):
    '''
    This is using PyTorch's GRU cell implementation
    '''
    def __init__(self, node_emb_dim, hidden_dim1, num_layers, output_dim, device, bias = True, dropout_prob = 0.2, MHA = False, use_position_encoding = False):

        super(GRUModel, self).__init__()

        self.hidden_dim = hidden_dim1
        self.num_layers = num_layers
        self.gru = nn.GRU(node_emb_dim, hidden_dim1, num_layers, batch_first = True, dropout = dropout_prob)
        self.device = device

        self.fc = nn.Linear(hidden_dim1, output_dim)
        self.MHA = MHA

        if self.MHA:
            self.n_head = 4
            self.mha_layer = MHALayer(node_emb_dim, hidden_dim1, hidden_dim1, self.n_head, self.device)
            self.use_position_encoding = use_position_encoding
        else:
            self.use_position_encoding = False

        if self.use_position_encoding:
            self.positional_encoder = PositionalEncoder(node_emb_dim, self.device)

    def get_position_encoding(self, len_curr_instace, d, n = 10000):
        '''
        code is a combination of codes from:
        > https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
        > https://blog.floydhub.com/the-transformer-in-pytorch/
        TO DO, need to check again about learnable positional encoding

        len_curr_instace = the current sequence length
        d = Dimension of the output embedding space
        n = User defined scalar. Set to 10,000 by the authors of Attention is all You Need.
        '''
        P = np.zeros((len_curr_instace, d))
        for pos in range(len_curr_instace):
            for i in range(0, d, 2):
                P[pos, i] = math.sin(pos / (n ** ((2 * i)/d)))
                P[pos, i + 1] = math.cos(pos / (n ** ((2 * (i + 1))/d)))
        return P

    def forward(self, x, len_curr_instace):
        '''
        len_curr_instance: the length of the current sequence before padding
        '''
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        x = x.to(self.device)
        if not self.MHA:
            out, _ = self.gru(x, h0)
            out = out[:, len_curr_instace - 1, :]
        else:
            if self.use_position_encoding:
                self.mha_layer = self.mha_layer.to(torch.float)
                x = torch.squeeze(x)
                x = x[:len_curr_instace]
                position_encoding = self.positional_encoder(x)
                try:
                    assert x.shape == position_encoding.shape
                except Exception as e:
                    print(e)
                position_encoding = torch.unsqueeze(position_encoding, 0).to(self.device)
                out = self.mha_layer(position_encoding.to(torch.float))
                out = out[:, len_curr_instace - 1, :]
            else:
                out = self.mha_layer(x)
                out = torch.mean(out, dim = 1)

        out = self.fc(out)

        return out

class GRUCell(nn.Module):
    '''
    Copied from https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
    not yet debugged -- not sure if the code can run
    '''
    def __init__(self, input_size, hidden_size, bias = True):
        super(GRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias = bias) # feature layer to hidden layer
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias = bias) # hidden layer to hidden layer
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1) 

        inputgate = F.sigmoid(i_i + h_i) 
        resetgate = F.sigmoid(i_r + h_r) 
        newgate = F.tanh(i_n + (resetgate * h_n)) 
        hy = newgate + inputgate * (hidden - newgate) 

        return hy

class GRUModelScratch(nn.Module):
    '''
    copied from https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
    not yet debugged -- not sure if the code can run
    '''
    def __init__(self, node_emb_dim, hidden_dim1, num_layers, output_dim, device, bias = True):
        super(GRUModel, self).__init__()

        self.hidden_dim = hidden_dim1
        self.layer_dim = num_layers
        self.gru_cell = GRUCell(node_emb_dim, hidden_dim1, num_layers)
        self.fc = nn.Linear(hidden_dim1, output_dim)

    def forward(self, x):
        outs = []
        hn = h0[0,:,:]

        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:, seq, :], hn)
            outs.append(hn)
        out = outs[-1].squeeze()
        out = self.fc(out)

        return out

class GRUCellLite(nn.Module):
    '''
    re-implementation of GRU-Li from paper https://arxiv.org/abs/1803.10225
    Ravanelli, M., Brakel, P., Omologo, M. and Bengio, Y., 2018. Light gated recurrent units for speech recognition. IEEE Transactions on Emerging Topics in Computational Intelligence, 2(2), pp.92-102.
    '''
    def __init__(self, node_emb_dim, hidden_dim1, bias = True):
        super(GRUCellLite, self).__init__()

        self.input_size = node_emb_dim
        self.hidden_size = hidden_dim1
        self.bias = bias
        self.x2h = nn.Linear(node_emb_dim, 3 * hidden_dim1, bias = bias)
        self.h2h = nn.Linear(hidden_dim1, 3 * hidden_dim1, bias = bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        inputgate = F.sigmoid(nn.BatchNorm1d(i_i) + h_i)
        newgate = nn.ReLU(nn.BatchNorm1d(i_n) + h_n)

        hy = newgate + inputgate * (hidden - newgate)

        return hy

class GRUModelLite(nn.Module):

    def __init__(self, node_emb_dim, hidden_dim1, num_layers, output_dim, device, bias = True, MHA = False):
        super(GRUModelLite, self).__init__()

        self.hidden_dim = hidden_dim1
        self.num_layers = num_layers
        self.gru = GRUCellLite(node_emb_dim, hidden_dim1, bias = bias)
        self.device = device
        self.fc = nn.Linear(hidden_dim1, output_dim)
        self.MHA = MHA

        if MHA:
            self.n_head = 4
            self.mha_layer = MHALayer(node_emb_dim, hidden_dim1, hidden_dim1, self.n_head, self.device)
        else:
            pass

    def forward(self, x, len_curr_instace):
        '''
        len_cur_instance: the length of the current sequence before padding
        '''
        x = x.to(self.device)
        if not self.MHA:
            out, _ = self.gru(x, self.hidden_dim)
            out = out[:, len_cur_instance - 1, :]
        else:
            out = self.mha_layer(x)
            out = torch.mean(out, dim = 1)
        out = self.fc(out)
        return out

class Transformer(nn.Module):
    '''
    Text classifier based on a pytorch TransformerEncoder
    '''
    def __init__(self, d_model, device, n_head = 8, d_ff = 2048, num_layers = 6, dropout = 0.1, activation = 'relu', classifier_dropout = 0.1,):
        super().__init__()
        assert d_model % n_head == 0, 'n_heads must divide evenly into d_model'

        self.pos_encoder = PositionalEncoder(d_model, device)

        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = n_head, dim_feedforward = d_ff, dropout = dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

        self.fc = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim = 1)
        x = self.fc(x)

        return x

class EncoderSVAE(nn.Module):
    def __init__(self, in_size, hidden_dim1):
        super(EncoderSVAE, self).__init__()
        self.linear1 = nn.Linear(in_size, hidden_dim1)
        nn.init.xavier_normal(self.linear1.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x

class DecoderSVAE(nn.Module):
    def __init__(self, in_size, hidden_dim1, output_dim):
        super(DecoderSVAE, self).__init__()
        self.linear1 = nn.Linear(in_size, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, output_dim)
        nn.init.xavier_normal(self.linear1.weight)
        nn.init.xavier_normal(self.linear2.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class SVAE(nn.Module):
    '''
    code copied and modified from its original: https://github.com/noveens/svae_cf/blob/master/main_svae.ipynb
    '''
    def __init__(self, in_size, hidden_dim1, output_dim, gru_layer, device):
        super(SVAE, self).__init__()

        self.in_size = in_size
        self.hidden_dim1 = hidden_dim1
        self.device = device

        self.encoder = EncoderSVAE(in_size, hidden_dim1)
        self.decoder = DecoderSVAE(in_size, hidden_dim1, output_dim)

        self.gru = nn.GRU(in_size, in_size, batch_first = True, num_layers = gru_layer)

        self.linear1 = nn.Linear(hidden_dim1, 2 * in_size)
        nn.init.xavier_normal(self.linear1.weight)

        self.tanh = nn.Tanh()

    def sample_latent(self, h_enc):
        temp_out = self.linear1(h_enc)

        mu = temp_out[:, :self.in_size]
        log_sigma = temp_out[:, self.in_size:]

        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size = sigma.size())).float()
        std_z = std_z.to(self.device)

        self.z_mean = mu
        self.z_log_sigma = log_sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, x):
        in_shape = x.shape
        rnn_out, _ = self.gru(x) # gru output (gru is rnn)
        rnn_out = rnn_out.view(in_shape[0] * in_shape[1], -1)
        enc_out = self.encoder(rnn_out) # encoder output
        sampled_z = self.sample_latent(enc_out)

        dec_out = self.decoder(sampled_z)
        dec_out = dec_out.view(in_shape[0], in_shape[1], -1)
        out = torch.mean(dec_out, dim = 1)

        return out # we dont need the z_mean and z_log_sigma

class SAGE(nn.Module):
    def __init__(self, node_emb_dim, hidden_dim1, hidden_dim2, output_dim, device, feature_pre = True, num_layers = 10, dropout = True):
        '''
        Not including graphSAGE layer. GraphSAGE layer is from torch_geometric.nn
        node_emb_dim = size of input layer
        hidden_dim1 = size of hidden layer in SAGE
        hidden_dim2 = size of feature dim in SAGE
        output_dim = size of output layer in SAGE -- in my architecture before averaging the node embeddings
        '''
        super(SAGE, self).__init__()

        self.feature_pre = feature_pre
        self.num_layers = num_layers # or layer_num
        self.dropout = dropout
        self.device = device
        self.sigmoid = nn.Sigmoid()

        if feature_pre:
            self.linear_pre = nn.Linear(node_emb_dim, hidden_dim1)
            self.conv_first = SAGEConv(hidden_dim1, hidden_dim2, aggr = 'max').to(self.device)
        else:
            self.conv_first = SAGEConv(node_emb_dim, hidden_dim2, aggr = 'max').to(self.device) # first convolution layer

        self.hidden_convs = nn.ModuleList()
        for _ in range(num_layers - 2): # hidden layers
            self.hidden_convs.append(SAGEConv(hidden_dim2, hidden_dim2, aggr = 'max').to(self.device))

        self.conv_out = SAGEConv(hidden_dim2, output_dim, aggr = 'max').to(self.device) # output layer

    def forward(self, x, adjs):
        '''
        x = pasangan current node dan neighbors, formatnya x0 = source, x1 = target
        adjs = graph adjacency

        [train_loader] computes the k-hop neighborhood of a batch of nodes and returns,
        for each layer, a bipartite graph object, holding the bipartite edges [edge_index],
        the index [e_id] of the original edges, and the size/shape [size] of the bipartite graph
        Target nodes are also included in the source nodes so that one can easily apply skip-connections and add
        self-loops
        '''
        for i, (edge_index, _, size) in enumerate(adjs):
            edge_index = edge_index.to(self.device) # no problem
            x_target = x[:size[1]] # target nodes
            x_target = x_target.to(self.device)
            x = x.to(self.device)
            if self.feature_pre:
                x = self.linear_pre(x)
            x = self.conv_first(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training = self.training)
            x = self.hidden_convs[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training = self.training)
            x = self.conv_out(x, edge_index)
            # x = self.sigmoid(x)
            x = F.normalize(x, p = 2, dim =- 1)
            if i == 0:
                layer_1_embeddings = x
            elif i == 1:
                layer_2_embeddings = x
            elif i == 2:
                layer_3_embeddings = x 
        return layer_1_embeddings, layer_2_embeddings, layer_3_embeddings


class MultiHeadExternalAttention(nn.Module):
    def __init__(self, embed_dim = args.in_size, n_heads = 4, dropout = 0.1):
        super(MultiHeadExternalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_head = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_head)

        self.query_matrix = nn.Linear(self.embed_dim, self.n_head * self.single_head_dim)

        self.memory_unit_k = nn.Linear(self.embed_dim, self.n_head * self.single_head_dim) # memory unit for k

        self.memory_unit_v = nn.Linear(self.embed_dim, self.n_head * self.single_head_dim) # memory unit for v

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.n_head * self.single_head_dim, self.embed_dim)

    def forward(self, input_array):
        batch_size = input_array.size(0)
        seq_length = input_array.size(1)
        key = input_array

        x = self.query_matrix(input_array)

        attn = self.memory_unit_k(x) # a linear layer M_k
        attn = F.softmax(attn) # normalize attention map

        attn = self.dropout(attn) # dropout

        out = self.memory_unit_v(attn) # a linear layer M_v
        out = out.view(batch_size, seq_length)
        out = self.out(out)
        out = self.dropout(out) # dropout

        return out


class RecModelDGI(nn.Module):
    def __init__(self,
                num_classes,
                device,
                paper_section_position = None,
                similarity_feat = None,
                neighbor_sizes = [15, 10, 5],
                nn_model = 'sage',
                node_emb_dim = 300,
                hidden_dim1 = 300,
                hidden_dim2 = 300,
                output_dim = 300,
                use_dgi = False,
                GRU = args.use_gru,
                GRU_MHA = False,
                LSTM = False,
                MHA = False,
                position_encoding = False,
                ablation_no_sage = args.ablation_no_sage,
                use_similarity = args.use_similarity,
                use_cf = args.use_cf,
                use_n2v = args.use_n2v,
                use_transformer = args.use_transformer,
                use_hftransformer = args.use_hftransformer,
                use_svae = args.use_svae,
                use_external_attention = args.use_external_attention,
                max_seq_len = None, # this one is counted in main.py (max_section)
                idx_to_ppr = None, # for getting paper text (abstract, title, and keywords) used for huggingface's transformer feature extractor
                idx_to_author = None, # for getting author keywords used for huggingface's transformer feature extractor
                ):

        '''
        num_instances = total instances in both train and test (train + test data)
        nn_model = sage/gr/pgnn/... (need to add more)
        num_classes = number of potential papers to be cited (depends on training data)
        uniq_labels = worded version of the classes --> for example, ppr_1234
        node_emb_dim = number of features in node embeddings
        hidden_dim1 = size of hidden layer in SAGE
        hidden_dim2 = size of output layer in SAGE -- in my architecture before averaging the node embeddings
        GRU = whether to use GRU (True) or not (False)
        GRU_MHA = whether to use multihead attention in GRU (True) or not (False)
        transfomer = whether to use Transfomer (True) or not (False)
        '''

        super().__init__()

        self.model = nn_model
        self.use_dgi = use_dgi
        self.device = device
        self.GRU = GRU
        if use_transformer or use_hftransformer:
            self.GRU = False
        self.GRU_MHA = GRU_MHA
        self.MHA = MHA
        self.position_encoding = position_encoding
        self.ablation_no_sage = ablation_no_sage
        self.use_similarity = use_similarity
        self.similarity_feat = similarity_feat
        self.use_cf = use_cf
        self.use_n2v = use_n2v
        self.use_svae = use_svae
        self.use_external_attention = use_external_attention
        self.use_transformer = use_transformer
        self.use_hftransformer = use_hftransformer
        if self.use_hftransformer or self.use_external_attention:
            self.use_transformer = False
            self.use_gru = False
            self.MHA = False
        self.max_seq_len = max_seq_len
        self.idx_to_ppr = idx_to_ppr
        self.idx_to_author = idx_to_author
        self.node_emb_dim = node_emb_dim

        if self.use_similarity and self.similarity_feat == None:
            print('Using similarity features however there is no similarity features given to the model!!')
            exit()

        if self.MHA and self.GRU:
            self.GRU = False
            self.GRU_MHA = False
        else:
            pass

        if self.model == 'sage':
            self.coauth_SAGE = SAGE(node_emb_dim, hidden_dim1, hidden_dim2, output_dim, self.device).to(self.device)
            self.coauth_SAGE = self.coauth_SAGE.to(self.device)

            self.citation_SAGE = SAGE(node_emb_dim, hidden_dim1, hidden_dim2, output_dim, self.device).to(self.device)
            self.citation_SAGE =  self.citation_SAGE.to(self.device)

            self.authorship_SAGE = SAGE(node_emb_dim, hidden_dim1, hidden_dim2, output_dim, self.device).to(self.device)
            self.authorship_SAGE = self.authorship_SAGE.to(self.device)

            self.section_SAGE = SAGE(node_emb_dim, hidden_dim1, hidden_dim2, output_dim, self.device).to(self.device)
            self.section_SAGE = self.section_SAGE.to(self.device)

            if args.train_onto_sage:
                self.onto_SAGE = SAGE(node_emb_dim, hidden_dim1, hidden_dim2, output_dim, self.device).to(self.device)
                self.active_sage_layers = [self.coauth_SAGE, self.authorship_SAGE, self.citation_SAGE, self.section_SAGE, self.onto_SAGE]
            else:
                self.active_sage_layers = [self.coauth_SAGE, self.authorship_SAGE, self.citation_SAGE, self.section_SAGE] # sage layers yang dipake saat ini, ini buat ngitung wide of input for prediction layer

            if self.use_dgi: # DGI part for SAGE
                self.coauth_DGI = DGIforSAGE(self.coauth_SAGE, node_emb_dim)
                self.coauth_DGI = self.coauth_DGI.to(self.device)
                self.citation_DGI = DGIforSAGE(self.citation_SAGE, node_emb_dim)
                self.citation_DGI = self.citation_DGI.to(self.device)
                self.authorship_DGI = DGIforSAGE(self.authorship_SAGE, node_emb_dim)
                self.authorship_DGI = self.authorship_DGI.to(self.device)
                self.section_DGI = DGIforSAGE(self.section_SAGE, node_emb_dim)
                self.section_DGI = self.section_DGI.to(self.device)

            if self.use_cf:
                self.prediction_layer = nn.Linear(hidden_dim2 * (len(self.active_sage_layers) + 1), num_classes) # +1 is for the neighborhood representation
            else:
                self.prediction_layer = nn.Linear(hidden_dim2 * len(self.active_sage_layers), num_classes)

            if args.section_embedding:
                if self.max_seq_len == None:
                    print('Error! Max section should be declared in main. Exiting...')
                    exit()
                else:
                    pass
                self.citation_se = SectionEncoder(node_emb_dim, self.device, self.max_seq_len)
                self.section_se = SectionEncoder(node_emb_dim, self.device, self.max_seq_len)
                self.authorship_se = SectionEncoder(node_emb_dim, self.device, self.max_seq_len)
                if args.train_onto_sage:
                    self.onto_se = SectionEncoder(node_emb_dim, self.device, self.max_seq_len)

            if self.GRU:
                self.n_layers = 3
                self.citation_GRU = GRUModel(node_emb_dim, hidden_dim1, self.n_layers, output_dim, self.device, MHA = self.GRU_MHA, use_position_encoding = self.position_encoding).to(self.device)
                self.section_GRU = GRUModel(node_emb_dim, hidden_dim1, self.n_layers, output_dim, self.device, MHA = self.GRU_MHA, use_position_encoding = self.position_encoding).to(self.device)
                self.authorship_GRU = GRUModel(node_emb_dim, hidden_dim1, self.n_layers, output_dim, self.device, MHA = self.GRU_MHA, use_position_encoding = self.position_encoding).to(self.device)
                if args.train_onto_sage:
                    self.onto_GRU = GRUModel(node_emb_dim, hidden_dim1, self.n_layers, output_dim, self.device, MHA = self.GRU_MHA, use_position_encoding = self.position_encoding).to(self.device)
            else:
                pass

            if self.MHA:
                n_head = 2
                self.mha_citation = MHALayer(node_emb_dim, hidden_dim1, hidden_dim2, n_head, self.device)
                self.mha_authorship = MHALayer(node_emb_dim, hidden_dim1, hidden_dim2, n_head, self.device)
                self.mha_section = MHALayer(node_emb_dim, hidden_dim1, hidden_dim2, n_head, self.device)
                if args.train_onto_sage:
                    self.mha_onto = mha_layer(node_emb_dim, hidden_dim1, hidden_dim2. n_head, self.device)
            else:
                pass

            if self.use_svae: # using Sequential VAE
                self.n_layers = 3
                self.citation_SVAE = SVAE(node_emb_dim, hidden_dim1, output_dim, self.n_layers, self.device).to(self.device)
                self.section_SVAE = SVAE(node_emb_dim, hidden_dim1, output_dim, self.n_layers, self.device).to(self.device)
                self.section_SVAE = SVAE(node_emb_dim, hidden_dim1, output_dim, self.n_layers, self.device).to(self.device)
                self.authorship_SVAE = SVAE(node_emb_dim, hidden_dim1, output_dim, self.n_layers, self.device).to(self.device)
            else:
                pass

            if self.use_transformer:
                n_head = 2
                dropout = 0.0
                self.citation_transformer = Transformer(node_emb_dim, self.device, n_head = n_head, dropout = dropout, classifier_dropout = dropout).to(self.device)
                self.authorship_transformer = Transformer(node_emb_dim, self.device, n_head = n_head, dropout = dropout, classifier_dropout = dropout).to(self.device)
                self.section_transformer = Transformer(node_emb_dim, self.device, n_head = n_head, dropout = dropout, classifier_dropout = dropout).to(self.device)
            else:
                pass

            if self.use_external_attention:
                n_head = 2
                dropout = 0.1
                self.citation_ea = MultiHeadExternalAttention(embed_dim = node_emb_dim, n_heads = n_head, dropout = dropout).to(self.device)
                self.authorship_ea = MultiHeadExternalAttention(embed_dim = node_emb_dim, n_heads = n_head, dropout = dropout).to(self.device)
                self.section_ea = MultiHeadExternalAttention(embed_dim = node_emb_dim, n_heads = n_head, dropout = dropout).to(self.device)
            else:
                pass

    def neighborhood_sampling(self, edge_index, curr_inputs, neighbor_sizes):
        '''
        Current inputs = node index di current instance
        '''
        node_sampler = NeighborSampler(torch.as_tensor(edge_index, device = self.device).to(torch.long),
                            node_idx = torch.as_tensor(curr_inputs, device = self.device).to(torch.long),
                            sizes = neighbor_sizes,
                            batch_size = len(curr_inputs), # since we are doing sampling per input instance to keep dependency
                            shuffle = True)

        return node_sampler

    def feature_mean(self, mat1, mat2):
        '''
        Averaging feature matrices for heterogeneous graph feature initialization
        - inputs 2 matrices, later will update to 3 for connection to topics
        '''
        result = []
        for i in range(len(mat1)):
            adding = mat1[i].add(mat2[i])
            averaging = torch.div(adding, 2)
            result.append(averaging)
        return torch.stack(result)

    def forward(self,
                author_batch,
                node_types_dict,
                number_of_authors, number_of_papers, neighbor_sizes,
                coauth_features,
                coauth_adj,
                citation_features,
                citation_adj,
                authorship_pca,
                authorship_adj,
                section_batch,
                section_pos_batch,
                section_features,
                section_adj,
                neighbor_rep_batch,
                onto_adj = None,
                ):
        '''
        author_batch = authors_batches[b] in main.py, i.e., the current authors for the current batch
        section_bach = inputs_batches[b] in main.py, i.e., the current sequences of papers in the current batch
        neighbor_rep_batch = neighbor_rep_batches[b] in main.py, i.e., the current neighborhood representations in the current batch
        '''
        assert(len(author_batch) == len(section_batch))

        if args.train_onto_sage:
            if onto_adj is None:
                print('Error: onto_adj is not supposed to be None. Exitting...')
                exit()

        num_authors = number_of_authors # jumlah authors in the whole data
        num_papers = number_of_papers # jumlah papers in the whole data

        '''
        OBTAINING HETEROGENEOUS GRAPH FEATURES
        - By averaging node features obtained from all related homogeneous graphs
        - Steps:
          1. averaging the feature per node type
          2. concat all averaged features from each node type (e.g., concat author nodes with paper nodes)
          3. average the features from step 2 with the PCA from hetero graph's adjacency matrix (contains edge info)
        '''
        authorship_author_features = coauth_features 
        authorship_paper_features = self.feature_mean(citation_features, section_features) 
        authorship_paper_and_author = torch.cat((authorship_author_features, authorship_paper_features))
        authorship_features = authorship_paper_and_author

        if neighbor_rep_batch == None:
            cf_mode = False 
        else:
            cf_mode = True

        if self.model == 'sage':

            dgi_coauth_losses = []
            dgi_citation_losses = []
            dgi_section_losses = []
            dgi_authorship_losses = []
            batch_representation = [] 
            batch_short_term_representation = []

            longest_section_batch = max([len(b) for b in section_batch])

            for i in range(len(author_batch)): 

                curr_author = author_batch[i] 
                curr_section = section_batch[i] 
                curr_section_pos = section_pos_batch[i] 
                if args.section_embedding:
                    try:
                        assert len(curr_section) == len(curr_section_pos)
                    except AssertionError as e:
                        print(e)

                else:
                    pass
                if cf_mode:
                    curr_neigh_rep = neighbor_rep_batch[i] 
                if args.train_onto_sage:
                    onto_sampler = self.neighborhood_sampling(onto_adj, curr_section, neighbor_sizes, ontology_graph = True)
                    for _, n_id, adjs in onto_sampler:
                        curr_inputs = citation_features[n_id]
                        onto_SAGE_input = None 
                        _, _, onto_embeddings = self.onto_SAGE(onto_SAGE_input, adjs)

                if not self.ablation_no_sage:
                    coauth_sampler = self.neighborhood_sampling(coauth_adj, curr_author, neighbor_sizes)
                    for _, n_id, adjs in coauth_sampler:
                        _, _, coauth_author_embeddings = self.coauth_SAGE(coauth_features[n_id], adjs)
                elif self.ablation_no_sage and self.use_n2v:
                    n_id = author_batch[i]
                    coauth_author_embeddings = []
                    for item in n_id: 
                        try:
                            n2v_embed = coauth_n2v.wv[str(item)] 
                        except:
                            n2v_embed = np.zeros(args.in_size)
                        author_embed = coauth_features[item]
                        n2v_embed = torch.Tensor(n2v_embed)
                        embed = torch.stack([n2v_embed, author_embed])
                        embed = torch.mean(embed, dim = 0)
                        coauth_author_embeddings.append(embed)
                    coauth_author_embeddings = torch.stack(coauth_author_embeddings)
                else:
                    n_id = author_batch[i]
                    coauth_author_embeddings = coauth_features[n_id] # tensor of tensor
                rep_coauth = torch.mean(coauth_author_embeddings, dim = 0)

                if not self.ablation_no_sage:
                    citation_sampler = self.neighborhood_sampling(citation_adj, curr_section, neighbor_sizes)
                    for _, n_id, adjs in citation_sampler:
                        _, _, citation_embeddings = self.citation_SAGE(citation_features[n_id], adjs)
                elif self.ablation_no_sage and self.use_n2v:
                    n_id = section_batch[i]
                    citation_embeddings = []
                    for item in n_id:
                        try:
                            n2v_embed = citation_n2v.wv[str(item)]
                        except:
                            n2v_embed = np.zeros(args.in_size)
                        citation_embed = citation_features[item]
                        n2v_embed = torch.Tensor(n2v_embed)
                        embed = torch.stack([n2v_embed, citation_embed])
                        embed = torch.mean(embed, dim = 0)
                        citation_embeddings.append(embed)
                    citation_embeddings = torch.stack(citation_embeddings)
                else:
                    n_id = section_batch[i]
                    assert n_id == curr_section
                    citation_embeddings = citation_features[n_id]

                if args.section_embedding:
                    n_id_order = {}
                    for x in curr_section:
                        try:
                            n_id_order[x] = n_id.tolist().index(x)
                        except AttributeError:
                            n_id_order[x] = n_id.index(x)

                    citation_se_input = torch.stack([citation_embeddings[y] for y in n_id_order.values()])
                    citation_se_input = citation_se_input.to(self.device)
                    citation_se_position = curr_section_pos

                    citation_se_output = self.citation_se(citation_se_input, citation_se_position)
                    temp = []
                    citation_se_output_counter = 0
                    for n in range(citation_embeddings.shape[0]):
                        if n not in n_id_order.values():
                            temp.append(citation_embeddings[n])
                        else:
                            try:
                                temp.append(citation_se_output[citation_se_output_counter])
                                citation_se_output_counter += 1
                            except Exception as e:
                                print(e)
                                exit()
                    citation_embeddings = torch.stack(temp)

                if self.use_similarity:
                    similarity_embeddings = []
                    for index in n_id:
                        try:
                            index = index.item()
                        except AttributeError:
                            pass
                        cur_sim_emb = self.similarity_feat.get(index)
                        if isinstance(cur_sim_emb, type(None)):
                            print(index)
                            continue
                        else:
                            similarity_embeddings.append(torch.from_numpy(cur_sim_emb))
                    similarity_embeddings = torch.stack(similarity_embeddings)
                    similarity_embeddings = torch.mean(similarity_embeddings, dim = 0)
                else:
                    pass

                if self.GRU or self.use_transformer or self.use_svae:
                    n_id_order = {} 
                    for x in curr_section:
                        try:
                            n_id_order[x] = n_id.tolist().index(x)
                        except AttributeError:
                            n_id_order[x] = n_id.index(x)
                    citation_gru_input = torch.stack([citation_embeddings[y] for y in n_id_order.values()])
                    citation_gru_input = citation_gru_input.to(self.device)
                    padding = torch.zeros(longest_section_batch - len(curr_section), citation_embeddings.shape[1])
                    padding = padding.to(self.device)
                    citation_padded_input = torch.cat((citation_gru_input, padding), 0) 
                    citation_padded_input = citation_padded_input.to(self.device)
                    citation_padded_input = torch.unsqueeze(citation_padded_input, 0) 
                    if self.GRU:
                        citation_gru_embeddings = self.citation_GRU(citation_padded_input, len(curr_section))
                        rep_citation = torch.squeeze(citation_gru_embeddings) 
                    elif self.use_svae:
                        citation_svae_embeddings = self.citation_SVAE(citation_padded_input)
                        rep_citation = torch.squeeze(citation_svae_embeddings)

                    elif self.use_transformer:
                        citation_transformer_embeddings = self.citation_transformer(citation_padded_input)
                        rep_citation = torch.squeeze(citation_transformer_embeddings)

                elif self.MHA and not self.use_transformer:
                    n_id_order = {} 
                    for x in curr_section:
                        try: 
                            n_id_order[x] = n_id.tolist().index(x)
                        except: 
                            n_id_order[x] = n_id.index(x)
                    citation_mha_input = torch.stack([citation_embeddings[y] for y in n_id_order.values()])
                    citation_mha_output = self.mha_citation(citation_mha_input)
                    rep_citation = torch.mean(citation_mha_output, dim = 0)

                elif self.use_external_attention:
                    n_id_order = {}
                    for x in curr_section:
                        try: 
                            n_id_order[x] = n_id.tolist().index(x)
                        except: 
                            n_id_order[x] = n_id.index(x)
                    citation_ea_input = torch.stack([citation_embeddings[y] for y in n_id_order.values()])
                    citation_ea_output = self.citation_ea(citation_ea_input)
                    rep_citation = torch.mean(citation_ea_output, dim = 0)

                elif self.use_hftransformer and not self.use_transformer and not self.GRU and not self.MHA:
                    curr_sequence_texts = []
                    for index in n_id:
                        try:
                            index = index.item()
                        except AttributeError:
                            pass
                        worded_index = self.idx_to_ppr.get(index)
                        list_of_text = self.paper_text.get(worded_index)
                        curr_text = []
                        for item in list_of_text:
                            if type(item) != list:
                                if item != '':
                                    curr_text.append(item)
                                else:
                                    pass
                            else:
                                for itm in item:
                                    curr_text.append(itm)
                        input_text = '. '.join(curr_text)
                        curr_sequence_texts.append(input_text)
                    rep_citation = self.citation_hftransformer(curr_sequence_texts)
                else:
                    rep_citation = torch.mean(citation_embeddings, dim = 0)

                if self.use_similarity:
                    temp = torch.stack([rep_citation, similarity_embeddings]) 
                    rep_citation = torch.mean(temp, dim = 0) 
                else:
                    pass

                if self.ablation_no_sage == False:
                    section_sampler = self.neighborhood_sampling(section_adj, curr_section, neighbor_sizes)
                    for _, n_id, adjs in section_sampler:
                        _, _, section_embeddings = self.section_SAGE(section_features[n_id], adjs)
                elif self.ablation_no_sage and self.use_n2v:
                    n_id = section_batch[i]
                    section_embeddings = []
                    for item in n_id: 
                        try:
                            n2v_embed = section_n2v.wv[str(item)] 
                        except:
                            n2v_embed = np.zeros(args.in_size)
                        section_embed = section_features[item]
                        n2v_embed = torch.Tensor(n2v_embed)
                        embed = torch.stack([n2v_embed, section_embed])
                        embed = torch.mean(embed, dim = 0)
                        section_embeddings.append(embed)
                    section_embeddings = torch.stack(section_embeddings)
                else:
                    n_id = section_batch[i]
                    section_embeddings = section_features[n_id]

                if args.section_embedding: 
                    n_id_order = {}
                    for x in curr_section:
                        try:
                            n_id_order[x] = n_id.tolist().index(x)
                        except AttributeError:
                            n_id_order[x] = n_id.index(x)

                    section_se_input = torch.stack([section_embeddings[y] for y in n_id_order.values()])
                    section_se_input = section_se_input.to(self.device)
                    section_se_position = curr_section_pos
                    section_se_output = self.section_se(section_se_input, section_se_position)
                    temp = []
                    section_se_output_counter = 0
                    for n in range(section_embeddings.shape[0]): 
                        if n not in n_id_order.values():
                            temp.append(section_embeddings[n])
                        else:
                            temp.append(section_se_output[section_se_output_counter])
                            section_se_output_counter += 1
                    section_embeddings = torch.stack(temp)

                if self.use_similarity:
                    similarity_embeddings = []
                    for index in n_id:
                        try:
                            index = index.item()
                        except AttributeError:
                            pass
                        cur_sim_emb = self.similarity_feat.get(index) 
                        if isinstance(cur_sim_emb, type(None)):
                            continue
                        else:
                            similarity_embeddings.append(torch.from_numpy(cur_sim_emb))
                    similarity_embeddings = torch.stack(similarity_embeddings) 
                    similarity_embeddings = torch.mean(similarity_embeddings, dim = 0) 

                if self.GRU or self.use_transformer or self.use_svae:
                    n_id_order = {}
                    for x in curr_section:
                        try:
                            n_id_order[x] = n_id.tolist().index(x)
                        except AttributeError:
                            n_id_order[x] = n_id.index(x)
                    section_gru_input = torch.stack([section_embeddings[y] for y in n_id_order.values()])
                    section_gru_input = section_gru_input.to(self.device)
                    padding = torch.zeros(longest_section_batch - len(curr_section), section_embeddings.shape[1])
                    padding = padding.to(self.device)
                    section_padded_input = torch.cat((section_gru_input, padding), 0)
                    section_padded_input = section_padded_input.to(self.device)
                    section_padded_input = torch.unsqueeze(section_padded_input, 0)
                    if self.GRU:
                        section_gru_embeddings = self.section_GRU(section_padded_input, len(curr_section))
                        rep_section = torch.squeeze(section_gru_embeddings)

                    elif self.use_svae:
                        section_svae_embeddings = self.section_SVAE(section_padded_input)
                        rep_section = torch.squeeze(section_svae_embeddings)

                    elif self.use_transformer:
                        section_transformer_embeddings = self.section_transformer(section_padded_input)
                        rep_section = torch.squeeze(section_transformer_embeddings)

                elif self.MHA and not self.use_transformer:
                    n_id_order = {}
                    for x in curr_section:
                        try:
                            n_id_order[x] = n_id.tolist().index(x)
                        except:
                            n_id_order[x] = n_id.index(x)
                    section_mha_input = torch.stack([section_embeddings[y] for y in n_id_order.values()])
                    section_mha_output = self.mha_section(section_mha_input)
                    rep_section = torch.mean(section_mha_output, dim = 0)

                elif self.use_external_attention:
                    n_id_order = {}
                    for x in curr_section:
                        try: 
                            n_id_order[x] = n_id.tolist().index(x)
                        except: 
                            n_id_order[x] = n_id.index(x)
                    section_ea_input = torch.stack([section_embeddings[y] for y in n_id_order.values()])
                    section_ea_output = self.section_ea(section_ea_input)
                    rep_section = torch.mean(section_ea_output, dim = 0)

                elif self.use_hftransformer and not self.use_transformer and not self.GRU and not self.MHA:
                    curr_sequence_texts = []
                    for index in n_id:
                        try:
                            index = index.item()
                        except AttributeError:
                            pass
                            worded_index = self.idx_to_ppr.get(index)
                            list_of_text = self.paper_text.get(worded_index)
                            curr_text = []
                        for item in list_of_text:
                            if type(item) != list:
                                if item != '':
                                    curr_text.append(item)
                                else:
                                    pass
                            else:
                                for itm in item:
                                    curr_text.append(itm)
                        input_text = '. '.join(curr_text)
                        curr_sequence_texts.append(input_text)
                    rep_section = self.section_hftransformer(curr_sequence_texts)
                else:
                    rep_section = torch.mean(section_embeddings, dim = 0)

                if self.use_similarity:
                    temp = torch.stack([rep_section, similarity_embeddings]) 
                    rep_section = torch.mean(temp, dim = 0) 

                else:
                    pass

                if self.ablation_no_sage == False:
                    authorship_sampler = self.neighborhood_sampling(coauth_adj, curr_section, neighbor_sizes)
                    for _, n_id, adjs in authorship_sampler:
                        _, _, authorship_paper_embeddings = self.authorship_SAGE(authorship_features[n_id], adjs)
                elif self.ablation_no_sage and self.use_n2v:
                    n_id = section_batch[i]
                    authorship_paper_embeddings = []
                    for item in n_id: 
                        try:
                            n2v_embed = authorship_n2v.wv[str(item)] 
                        except:
                            n2v_embed = np.zeros(args.in_size)
                        authorship_embed = authorship_features[item]
                        n2v_embed = torch.Tensor(n2v_embed)
                        embed = torch.stack([n2v_embed, authorship_embed])
                        embed = torch.mean(embed, dim = 0)
                        authorship_paper_embeddings.append(embed)
                    authorship_paper_embeddings = torch.stack(authorship_paper_embeddings)
                else:
                    n_id = section_batch[i]
                    authorship_paper_embeddings = authorship_features[n_id]

                if args.section_embedding: 
                    n_id_order = {}
                    for x in curr_section:
                        try:
                            n_id_order[x] = n_id.tolist().index(x)
                        except AttributeError:
                            n_id_order[x] = n_id.index(x)

                    authorship_paper_se_input = torch.stack([authorship_paper_embeddings[y] for y in n_id_order.values()])
                    authorship_paper_se_input = authorship_paper_se_input.to(self.device)
                    authorship_paper_se_position = curr_section_pos
                    authorship_paper_se_output = self.authorship_se(authorship_paper_se_input, authorship_paper_se_position)
                    temp = []
                    authorship_paper_se_output_counter = 0
                    for n in range(authorship_paper_embeddings.shape[0]):
                        if n not in n_id_order.values():
                            temp.append(authorship_paper_embeddings[n])
                        else:
                            temp.append(authorship_paper_se_output[authorship_paper_se_output_counter])
                            authorship_paper_se_output_counter += 1
                    authorship_paper_embeddings = torch.stack(temp)

                if self.use_similarity:
                    similarity_embeddings = []
                    for index in n_id:
                        try:
                            index = index.item()
                        except AttributeError:
                            pass
                        cur_sim_emb = self.similarity_feat.get(index)
                        if isinstance(cur_sim_emb, type(None)):
                            continue
                        else:
                            similarity_embeddings.append(torch.from_numpy(cur_sim_emb))
                    similarity_embeddings = torch.stack(similarity_embeddings) 
                    similarity_embeddings = torch.mean(similarity_embeddings, dim = 0) 
                if self.GRU or self.use_transformer or self.use_svae:
                    n_id_order = {}
                    for x in curr_section:
                        try:
                            n_id_order[x] = n_id.tolist().index(x)
                        except AttributeError:
                            n_id_order[x] = n_id.index(x)
                    authorship_gru_input = torch.stack([authorship_paper_embeddings[y] for y in n_id_order.values()])
                    authorship_gru_input = authorship_gru_input.to(self.device)
                    padding = torch.zeros(longest_section_batch - len(curr_section), authorship_paper_embeddings.shape[1])
                    padding = padding.to(self.device)
                    authorship_padded_input = torch.cat((authorship_gru_input, padding), 0)
                    authorship_padded_input = authorship_padded_input.to(self.device)
                    authorship_padded_input = torch.unsqueeze(authorship_padded_input, 0)
                    if self.GRU:
                        authorship_gru_embeddings = self.authorship_GRU(authorship_padded_input, len(curr_section))
                        rep_authorship = torch.squeeze(authorship_gru_embeddings)
                    elif self.use_svae:
                        authorship_svae_embeddings = self.authorship_SVAE(authorship_padded_input)
                        rep_authorship = torch.squeeze(authorship_svae_embeddings)
                    elif self.use_transformer:
                        authorship_transformer_embeddings = self.authorship_transformer(authorship_padded_input)
                        rep_authorship = torch.squeeze(authorship_transformer_embeddings)

                elif self.MHA and not self.use_transformer and not self.use_external_attention:
                    n_id_order = {}
                    for x in curr_section:
                        try:
                            n_id_order[x] = n_id.tolist().index(x)
                        except:
                            n_id_order[x] = n_id.index(x)
                    authorship_mha_input = torch.stack([authorship_paper_embeddings[y] for y in n_id_order.values()])
                    authorship_mha_output = self.mha_section(authorship_mha_input)
                    rep_authorship = torch.mean(authorship_mha_output, dim = 0)

                elif self.use_external_attention:
                    n_id_order = {}
                    for x in curr_section:
                        try: 
                            n_id_order[x] = n_id.tolist().index(x)
                        except: 
                            n_id_order[x] = n_id.index(x)
                    authorship_ea_input = torch.stack([authorship_paper_embeddings[y] for y in n_id_order.values()])
                    authorship_ea_output = self.authorship_ea(authorship_ea_input)
                    rep_authorship = torch.mean(authorship_ea_output, dim = 0)

                elif self.use_hftransformer and not self.use_transformer and not self.GRU and not self.MHA:
                    curr_sequence_texts = []
                    for index in n_id:
                        try:
                            index = index.item()
                        except AttributeError:
                            pass
                        worded_index = self.idx_to_ppr.get(index)
                        list_of_text = self.paper_text.get(worded_index)
                        curr_text = []
                        for item in list_of_text:
                            if type(item) != list:
                                if item != '':
                                    curr_text.append(item)
                                else:
                                    pass
                            else:
                                for itm in item:
                                    curr_text.append(itm)
                        input_text = '. '.join(curr_text)
                        curr_sequence_texts.append(input_text)
                    rep_authorship = self.authorship_hftransformer(curr_sequence_texts)
                else:
                    rep_authorship = torch.mean(authorship_paper_embeddings, dim = 0)

                if self.use_similarity:
                    temp = torch.stack([rep_authorship, similarity_embeddings]) 
                    rep_authorship = torch.mean(temp, dim = 0) 
                else:
                    pass

                if cf_mode:
                    input_instance_rep = torch.cat((rep_coauth, rep_authorship, rep_citation, rep_section, curr_neigh_rep)) 
                else:
                    input_instance_rep = torch.cat((rep_coauth, rep_authorship, rep_citation, rep_section)) 

                batch_representation.append(input_instance_rep)

            batch_representation = torch.stack(batch_representation) 
            batch_representation = batch_representation.to(self.device)

            distribution_score = self.prediction_layer(batch_representation)

            if self.use_dgi and self.training:
                dgi_coauth_losses = torch.stack(dgi_coauth_losses)
                dgi_citation_losses = torch.stack(dgi_citation_losses)
                dgi_section_losses = torch.stack(dgi_section_losses)
                dgi_authorship_losses = torch.stack(dgi_authorship_losses)

                dgi_coauth_sum_loss = torch.mean(dgi_coauth_losses)
                dgi_citation_sum_loss = torch.mean(dgi_citation_losses)
                dgi_section_sum_loss = torch.mean(dgi_section_losses)
                dgi_authorship_sum_loss = torch.mean(dgi_authorship_losses)

                dgi_mtl_loss = dgi_coauth_sum_loss + dgi_citation_sum_loss + dgi_section_sum_loss + dgi_authorship_sum_loss

                return distribution_score, dgi_mtl_loss
            else: 
                return distribution_score

class RecModelLite(nn.Module):
    def __init__(self,
                num_classes,
                device,
                paper_section_position = None,
                similarity_feat = None,
                neighbor_sizes = [15, 10, 5],
                nn_model = 'sage',
                node_emb_dim = 300,
                hidden_dim1 = 300,
                hidden_dim2 = 300,
                output_dim = 300,
                use_dgi = False,
                GRU = args.use_gru,
                lite_GRU = False,
                GRU_MHA = False,
                LSTM = False,
                MHA = False,
                position_encoding = False,
                ablation_no_sage = args.ablation_no_sage,
                use_similarity = args.use_similarity,
                use_cf = args.use_cf,
                use_n2v = args.use_n2v,
                use_transformer = args.use_transformer,
                use_hftransformer = args.use_hftransformer,
                use_svae = args.use_svae,
                use_external_attention = args.use_external_attention,
                max_seq_len = None,
                idx_to_ppr = None, 
                idx_to_author = None, 
                ):

        super().__init__()

        self.model = nn_model
        self.use_dgi = use_dgi
        self.device = device
        self.GRU = GRU
        if use_transformer or use_external_attention:
            self.GRU = False
        self.GRU_MHA = GRU_MHA
        self.MHA = MHA
        self.position_encoding = position_encoding
        self.ablation_no_sage = ablation_no_sage
        self.use_similarity = use_similarity
        self.similarity_feat = similarity_feat
        self.use_cf = use_cf
        self.use_n2v = use_n2v
        self.use_external_attention = use_external_attention
        self.use_transformer = use_transformer
        self.use_hftransformer = use_hftransformer
        self.use_svae = use_svae
        self.max_seq_len = max_seq_len
        self.lite_GRU = lite_GRU
        self.idx_to_ppr = idx_to_ppr
        self.idx_to_author = idx_to_author

        if self.use_similarity and self.similarity_feat == None:
            print('Using similarity features however there is no similarity features given to the model!!')
            exit()

        if self.MHA and self.GRU:
            self.GRU = False
            self.GRU_MHA = False
        else:
            pass

        if self.model == 'sage':
            # these SAGE model inits will output embeddings
            self.coauth_SAGE = SAGE(node_emb_dim, hidden_dim1, hidden_dim2, output_dim, self.device).to(self.device)
            self.coauth_SAGE = self.coauth_SAGE.to(self.device)

            self.citation_SAGE = SAGE(node_emb_dim, hidden_dim1, hidden_dim2, output_dim, self.device).to(self.device)
            self.citation_SAGE =  self.citation_SAGE.to(self.device)

            self.authorship_SAGE = SAGE(node_emb_dim, hidden_dim1, hidden_dim2, output_dim, self.device).to(self.device)
            self.authorship_SAGE = self.authorship_SAGE.to(self.device)

            self.section_SAGE = SAGE(node_emb_dim, hidden_dim1, hidden_dim2, output_dim, self.device).to(self.device)
            self.section_SAGE = self.section_SAGE.to(self.device)

            self.active_sage_layers = [self.coauth_SAGE, self.authorship_SAGE, self.citation_SAGE, self.section_SAGE]

            if self.use_cf:
                self.prediction_layer = nn.Linear(hidden_dim2 * (2 + 1), num_classes)
            else:
                self.prediction_layer = nn.Linear(hidden_dim2 * 2, num_classes)

            if args.section_embedding:
                if self.max_seq_len == None:
                    print('Error! Max section should be declared in main. Exiting...')
                    exit()
                else:
                    pass
                self.citation_se = SectionEncoder(node_emb_dim, self.device, self.max_seq_len)
                self.section_se = SectionEncoder(node_emb_dim, self.device, self.max_seq_len)
                self.authorship_se = SectionEncoder(node_emb_dim, self.device, self.max_seq_len)

            if self.GRU:
                self.n_layers = 3
                if not self.lite_GRU:
                    self.paper_GRU = GRUModel(node_emb_dim, hidden_dim1, self.n_layers, output_dim, self.device, MHA = self.GRU_MHA, use_position_encoding = self.position_encoding).to(self.device)
                else:
                    self.paper_GRU = GRUModelLite(node_emb_dim, hidden_dim1, self.n_layers, output_dim, self.device, MHA = self.GRU_MHA).to(self.device)

            elif self.use_transformer:
                n_head = 2
                node_emb_dim = args.in_size
                dropout = 0.1
                self.paper_transformer = Transformer(node_emb_dim, self.device, n_head = n_head, dropout = dropout, classifier_dropout = dropout).to(self.device)
            else:
                pass

            if self.use_external_attention:
                n_head = 2
                dropout = 0.1
                self.paper_ea = MultiHeadExternalAttention(embed_dim = node_emb_dim, n_heads = n_head, dropout = dropout).to(self.device)

            if self.use_svae:
                self.n_layers = 3
                self.paper_SVAE = SVAE(node_emb_dim, hidden_dim1, output_dim, self.n_layers, self.device).to(self.device)

            if self.MHA:
                self.n_head = 2
                self.mha_paper = MHALayer(node_emb_dim, hidden_dim1, hidden_dim2, self.n_head, self.device)

    def neighborhood_sampling(self, edge_index, curr_inputs, neighbor_sizes):
        '''
        Current inputs = node index di current instance
        '''
        node_sampler = NeighborSampler(torch.as_tensor(edge_index, device = self.device).to(torch.long),
                            node_idx = torch.as_tensor(curr_inputs, device = self.device).to(torch.long),
                            sizes = neighbor_sizes,
                            batch_size = len(curr_inputs), # since we are doing sampling per input instance to keep dependency
                            shuffle = True)
        return node_sampler

    def feature_mean(self, mat1, mat2):
        '''
        Averaging feature matrices for heterogeneous graph feature initialization
        - inputs 2 matrices, later will update to 3 for connection to topics
        '''
        result = []
        # topic_part = []
        for i in range(len(mat1)):
            adding = mat1[i].add(mat2[i])
            averaging = torch.div(adding, 2)
            result.append(averaging)
        return torch.stack(result)

    def forward(self,
                author_batch,
                node_types_dict,
                number_of_authors, number_of_papers, neighbor_sizes,
                coauth_features,
                coauth_adj,
                citation_features,
                citation_adj,
                authorship_pca, 
                authorship_adj,
                section_batch,
                section_pos_batch, 
                section_features,
                section_adj,
                neighbor_rep_batch,
                ):
        '''
        author_batch = authors_batches[b] in main.py, i.e., the current authors for the current batch
        section_bach = inputs_batches[b] in main.py, i.e., the current sequences of papers in the current batch
        neighbor_rep_batch = neighbor_rep_batches[b] in main.py, i.e., the current neighborhood representations in the current batch
        '''
        assert(len(author_batch) == len(section_batch))

        num_authors = number_of_authors # jumlah authors in the whole data
        num_papers = number_of_papers # jumlah papers in the whole data

        '''
        OBTAINING HETEROGENEOUS GRAPH FEATURES
        - By averaging node features obtained from all related homogeneous graphs
        - Steps:
          1. averaging the feature per node type
          2. concat all averaged features from each node type (e.g., concat author nodes with paper nodes)
          3. average the features from step 2 with the PCA from hetero graph's adjacency matrix (contains edge info)
        '''
        authorship_author_features = coauth_features 
        authorship_paper_features = self.feature_mean(citation_features, section_features)
        authorship_paper_and_author = torch.cat((authorship_author_features, authorship_paper_features))
        authorship_features = authorship_paper_and_author

        if neighbor_rep_batch == None:
            cf_mode = False 
        else:
            cf_mode = True 
        if self.model == 'sage':

            batch_representation = [] # jadinya list of tensors, for long term representation
            batch_short_term_representation = []

            try:
                longest_section_batch = max([len(b) for b in section_batch])
            except Exception as e:
                print(e)
                exit()

            for i in range(len(author_batch)):

                curr_author = author_batch[i]
                curr_section = section_batch[i]
                curr_section_pos = section_pos_batch[i]
                if args.section_embedding:
                    try:
                        assert len(curr_section) == len(curr_section_pos)
                    except AssertionError as e:
                        print(e)
                        print(curr_section, len(curr_section))
                        print(curr_section_pos, len(curr_section_pos))
                else:
                    pass
                if cf_mode:
                    curr_neigh_rep = neighbor_rep_batch[i] 

                coauth_sampler = self.neighborhood_sampling(coauth_adj, curr_author, neighbor_sizes)
                for _, n_id, adjs in coauth_sampler:
                    _, _, coauth_author_embeddings = self.coauth_SAGE(coauth_features[n_id], adjs)
                rep_coauth = torch.mean(coauth_author_embeddings, dim = 0)

                if self.ablation_no_sage == False:
                    citation_sampler = self.neighborhood_sampling(citation_adj, curr_section, neighbor_sizes)
                    section_sampler = self.neighborhood_sampling(section_adj, curr_section, neighbor_sizes)
                    authorship_sampler = self.neighborhood_sampling(authorship_adj, curr_section, neighbor_sizes)
                    coauth_sampler = self.neighborhood_sampling(coauth_adj, curr_author, neighbor_sizes)
                    for _, n_id, adjs in citation_sampler:
                        _, _, citation_embeddings = self.citation_SAGE(citation_features[n_id], adjs)
                    for _, n_id, adjs in section_sampler:
                        _, _, section_embeddings = self.section_SAGE(section_features[n_id], adjs)
                    for _, n_id, adjs in authorship_sampler:
                        _, _, authorship_paper_embeddings = self.authorship_SAGE(authorship_features[n_id], adjs)
                    for _, n_id, adjs in coauth_sampler:
                        _, _, coauth_author_embeddings = self.coauth_SAGE(coauth_features[n_id], adjs)

                elif self.ablation_no_sage:
                    n_id = section_batch[i]
                    n_id_auth = author_batch[i]
                    assert n_id == curr_section
                    assert n_id_auth == curr_author
                    if self.use_n2v:
                        citation_embeddings, section_embeddings, authorship_paper_embeddings, coauth_author_embeddings = [], [], [], []
                        for item in n_id:
                            try:
                                n2v_embed_citation = citation_n2v.wv[str(item)]
                            except:
                                n2v_embed_citation = np.zeros(args.in_size)
                            citation_text_embed = citation_features[item]
                            n2v_embed_citation = torch.Tensor(n2v_embed_citation)
                            embed_citation = torch.stack([n2v_embed_citation, citation_text_embed])

                            try:
                                n2v_embed_section = section_n2v.wv[str(item)]
                            except:
                                n2v_embed_section = np.zeros(args.in_size)
                            section_text_embed = section_features[item]
                            n2v_embed_section = torch.Tensor(n2v_embed_section)
                            embed_section = torch.stack([n2v_embed_section, section_text_embed])

                            try:
                                n2v_embed_authorship = authorship_n2v[str(item)]
                            except:
                                n2v_embed_authorship = np.zeros(args.in_size)
                            authorship_text_embed = authorship_features[item]
                            n2v_embed_authorship = torch.Tensor(n2v_embed_authorship)
                            embed_authorship = torch.stack([n2v_embed_authorship, authorship_text_embed])

                            citation_embeddings.append(torch.mean(embed_citation, dim = 0))
                            section_embeddings.append(torch.mean(embed_section, dim = 0))
                            authorship_paper_embeddings.append(torch.mean(embed_authorship, dim = 0))

                        for item in n_id_auth:
                            try:
                                n2v_embed_author = coauth_n2v.wv[str(item)]
                            except:
                                n2v_embed_author = np.zeros(args.in_size)
                            coauth_auth_embed = coauth_features[item]
                            n2v_embed_author = torch.Tensor(n2v_embed_author)
                            embed_author = torch.stack([n2v_embed_author, coauth_auth_embed])

                            coauth_author_embeddings.append(torch.mean(embed_author, dim = 0))

                        citation_embeddings, section_embeddings, authorship_paper_embeddings, coauth_author_embeddings = torch.stack(citation_embeddings), torch.stack(section_embeddings), torch.stack(authorship_paper_embeddings), torch.stack(coauth_author_embeddings)
                    else:
                        citation_embeddings = citation_features[n_id]
                        section_embeddings = section_features[n_id]
                        authorship_paper_embeddings = authorship_features[n_id]
                        coauth_author_embeddings = coauth_features[n_id_auth]

                rep_author = torch.mean(coauth_author_embeddings, dim = 0) 
                rep_paper = []

                for ti in range(section_embeddings.shape[0]):
                    curr_citation_emb = citation_embeddings[ti]
                    curr_section_emb = section_embeddings[ti]
                    curr_authorship_emb = authorship_paper_embeddings[ti]
                    combi_tensor = []
                    for ei in range(curr_citation_emb.shape[-1]):
                        combi_tensor.append(curr_citation_emb[ei] * curr_section_emb[ei] * curr_authorship_emb[ei]) 
                    combi_tensor = torch.Tensor(combi_tensor)
                    rep_paper.append(combi_tensor)

                rep_paper = torch.stack(rep_paper)

                if self.use_hftransformer:
                    paper_embeddings = []
                    for index in n_id:
                        try:
                            index = index.item()
                        except AttributeError:
                            pass
                        worded_index = self.idx_to_ppr.get(index)
                        list_of_text = self.paper_text.get(worded_index)
                        curr_text = []
                        for item in list_of_text:
                            if type(item) != list:
                                if item != '':
                                    curr_text.append(item)
                                else:
                                    pass
                            else:
                                for itm in item:
                                    curr_text.append(itm)
                        input_text = '. '.join(curr_text)
                        curr_emb = self.paper_hftransformer(input_text)
                        paper_embeddings.append(curr_emb)
                    paper_embeddings_hf = torch.stack(paper_embeddings)
                    rep_paper = torch.mean(paper_embeddings_hf, dim = 0)
                else:
                    pass

                if self.GRU or self.use_transformer or self.use_svae:
                    n_id_order = {} 
                    for x in curr_section:
                        try:
                            n_id_order[x] = n_id.tolist().index(x)
                        except AttributeError:
                            n_id_order[x] = n_id.index(x)
                    gru_input = torch.stack([rep_paper[y] for y in n_id_order.values()]) 
                    gru_input = gru_input.to(self.device)
                    padding = torch.zeros(longest_section_batch - len(curr_section), rep_paper.shape[1])
                    padding = padding.to(self.device)
                    padded_input = torch.cat((gru_input, padding), 0) 
                    padded_input = padded_input.to(self.device)
                    padded_input = torch.unsqueeze(padded_input, 0) 
                    if self.GRU:
                        gru_embeddings = self.paper_GRU(padded_input, len(curr_section))
                        rep_paper = torch.squeeze(gru_embeddings) 
                    elif self.use_svae:
                        svae_embeddings = self.paper_SVAE(padded_input)
                        rep_paper = torch.squeeze(svae_embeddings)

                    elif self.use_transformer:
                        transformer_embeddings = self.paper_transformer(padded_input)
                        rep_paper = torch.squeeze(transformer_embeddings)

                elif self.MHA and not self.use_transformer:
                    n_id_order = {} 
                    for x in curr_section:
                        try:
                            n_id_order[x] = n_id.tolist().index(x)
                        except:
                            n_id_order[x] = n_id.index(x)
                    mha_input = torch.stack([rep_paper[y] for y in n_id_order.values()])
                    mha_output = self.mha_paper(mha_input)
                    rep_paper = torch.mean(mha_output, dim = 0)

                elif self.use_external_attention:
                    n_id_order = {}
                    for x in curr_section:
                        try:
                            n_id_order[x] = n_id.tolist().index(x)
                        except:
                            n_id_order[x] = n_id.index(x)
                    ea_input = torch.stack([rep_paper[y] for y in n_id_order.values()])
                    ea_output = self.paper_ea(ea_input)
                    rep_paper = torch.mean(ea_output, dim = 0)

                elif self.use_transformer:
                    pass

                else:
                    rep_paper = torch.mean(rep_paper, dim = 0)

                rep_paper = rep_paper.to(self.device)
                rep_author = rep_author.to(self.device)
                if cf_mode:
                    input_instance_rep = torch.cat((rep_author, rep_paper)) 
                else:
                    input_instance_rep = torch.cat((rep_author, rep_paper)) 

                batch_representation.append(input_instance_rep)

            batch_representation = torch.stack(batch_representation) 
            batch_representation = batch_representation.to(self.device)

            distribution_score = self.prediction_layer(batch_representation)

            return distribution_score
