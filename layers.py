import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn.functional as F
from IPython import embed


class MLP_Match(nn.Module):
    def __init__(self , feat):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(256, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y


def row_wise_matrix_mul(X , Y , rows , embedding): # X and Y are of shape [clips X 1024] , [sentences X 1024]
 return torch.mul(X.reshape(-1 , 1) , Y.reshape(-1 , 1)).reshape(rows , embedding).sum(dim=1)

class ScoresMatch(nn.Module):
    def __init__(self, embed_x_size, x_spatial_dim=None, embed_y_size=None, y_spatial_dim=None):
      super(ScoresMatch, self).__init__()
      print(x_spatial_dim, y_spatial_dim)
      embed_y_size = embed_y_size if embed_y_size is not None else embed_x_size
      self.y_spatial_dim = y_spatial_dim if y_spatial_dim is not None else x_spatial_dim

      self.embed_size = max(embed_x_size, embed_y_size)
      self.x_spatial_dim = x_spatial_dim

      self.embed_X = nn.Conv1d(embed_x_size, self.embed_size, 1)
      self.embed_Y = nn.Conv1d(embed_y_size, self.embed_size, 1)
      if x_spatial_dim is not None:
        self.normalize_S = nn.BatchNorm1d(self.x_spatial_dim * self.y_spatial_dim)

        self.margin_X = nn.Conv1d(self.y_spatial_dim, 1, 1)
        self.margin_Y = nn.Conv1d(self.x_spatial_dim, 1, 1)

    def forward(self, X, Y=None):
      X_t = X.transpose(1, 2)
      Y_t = Y.transpose(1, 2) if Y is not None else X_t

      X_embed = self.embed_X(X_t)
      Y_embed = self.embed_Y(Y_t)

      X_norm = F.normalize(X_embed).squeeze().transpose(0,1)
      Y_norm = F.normalize(Y_embed).squeeze().transpose(0,1)

      X_norm_shape = X_norm.shape[0]
      Y_norm_shape = Y_norm.shape[0]
      embedding = X_norm.shape[1]

      #S = X_norm.transpose(1, 2).bmm(Y_norm)
      return row_wise_matrix_mul(X_norm , Y_norm , X_norm_shape , embedding)
#      if self.x_spatial_dim is not None:
#          S = self.normalize_S(S.view(-1, self.x_spatial_dim * self.y_spatial_dim)) \
#              .view(-1, self.x_spatial_dim, self.y_spatial_dim)


class GroupMLP(nn.Module):
  def __init__(self, in_features, mid_features, out_features, drop=0.5, groups=1):
    super(GroupMLP, self).__init__()

    self.conv1 = nn.Conv1d(in_features, mid_features, 1)
    self.drop  = nn.Dropout(p=drop)
    self.relu  = nn.ReLU()
    self.conv2 = nn.Conv1d(mid_features, out_features, 1, groups=groups)

  def forward(self, a):
    N, C = a.size()
    h = self.relu(self.conv1(a.view(N, C, 1)))
    return self.conv2(self.drop(h)).view(N, -1)

class Seq2Seq(nn.Module):
  def __init__(self, embedding_features, rnn_features, rnn_bidirectional=False):
    super(Seq2Seq, self).__init__()
    self.bidirectional = rnn_bidirectional

    self.rnn = nn.GRU(input_size=embedding_features,
              hidden_size=rnn_features,
              num_layers=1, batch_first=True,
              bidirectional=rnn_bidirectional)

    self.features = rnn_features

    self._init_rnn(self.rnn.weight_ih_l0)
    self._init_rnn(self.rnn.weight_hh_l0)
    self.rnn.bias_ih_l0.data.zero_()
    self.rnn.bias_hh_l0.data.zero_()

  def _init_rnn(self, weight):
    for w in weight.chunk(3, 0):
      init.xavier_uniform(w)

  def forward(self, q_emb, q_len, hidden=None , pooling=False):
    lengths = q_len.long()
    lens, indices = torch.sort(lengths, 0, True)

    packed_batch = pack_padded_sequence(q_emb[indices.cuda()], lens.tolist(), batch_first=True)
    if hidden is not None:
      N_, H_ = hidden.size()
      _, outputs = self.rnn(packed_batch, hidden[indices.cuda()].view(1, N_, H_))
    else:
      _, outputs = self.rnn(packed_batch)

    if self.bidirectional:
      outputs = torch.cat([ outputs[0, :, :], outputs[1, :, :] ], dim=1)
    else:
      outputs = outputs.squeeze(0)
    print(outputs.shape)
    _, _indices = torch.sort(indices, 0)
    outputs = outputs[_indices.cuda()]
    print(outputs.shape)
    exit(-1)
    return outputs


class Attention(nn.Module):
  def __init__(self, embedding_features, rnn_features, rnn_bidirectional=False):
    super(Attention, self).__init__()
    hid_size = rnn_features
    natt = rnn_features

    self.rnn = nn.GRU(input_size=embedding_features,
                    hidden_size=rnn_features,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=rnn_bidirectional)
    self.lin = nn.Linear(hid_size,natt)
    self.att_w = nn.Linear(natt,1,bias=False)
    self.tanh = nn.Tanh()

    self._init_rnn(self.rnn.weight_ih_l0)
    self._init_rnn(self.rnn.weight_hh_l0)
    self.rnn.bias_ih_l0.data.zero_()
    self.rnn.bias_hh_l0.data.zero_()

  def _init_rnn(self, weight):
    for w in weight.chunk(3, 0):
      init.xavier_uniform(w)

  def forward(self, q_emb, q_len, hidden=None , pooling=False):
    lengths = q_len.long()
    lens, indices = torch.sort(lengths, 0, True)

    packed_batch = pack_padded_sequence(q_emb[indices.cuda()], lens.tolist(), batch_first=True)
    if hidden is not None:
      N_, H_ = hidden.size()
      hs, _ = self.rnn(packed_batch, hidden[indices.cuda()].view(1, N_, H_))
    else:
      hs, _ = self.rnn(packed_batch)
    enc_sents, len_s = pad_packed_sequence(hs, batch_first=True)

    emb_h = self.tanh(self.lin(enc_sents.contiguous().view(enc_sents.size(0)*enc_sents.size(1),-1)))  # Nwords * Emb
    attend = self.att_w(emb_h).view(enc_sents.size(0),
                                    enc_sents.size(1))
    mask = self._list_to_bytemask(list(len_s))
    all_att = self._masked_softmax(attend, mask)
    try:
      attended = all_att.unsqueeze(2).expand_as(enc_sents) * enc_sents
    except:
      embed()
      raise

    _, _indices = torch.sort(indices, 0)
    outputs = attended.sum(1,True).squeeze(1)[_indices.cuda()]

    return outputs

  def forward_att(self, q_emb, q_len, hidden=None):
    lengths = q_len.long()
    lens, indices = torch.sort(lengths, 0, True)

    packed_batch = pack_padded_sequence(q_emb[indices.cuda()], lens.tolist(), batch_first=True)
    if hidden is not None:
      N_, H_ = hidden.size()
      hs, _ = self.rnn(packed_batch, hidden[indices.cuda()].view(1, N_, H_))
    else:
      hs, _ = self.rnn(packed_batch)
    enc_sents, len_s = pad_packed_sequence(hs, batch_first=True)

    _, _indices = torch.sort(indices, 0)
    enc_sents = rnn_sents[_indices.cuda()]

    emb_h = self.tanh(self.lin(enc_sents.contiguous().view(enc_sents.size(0)*enc_sents.size(1),-1)))  # Nwords * Emb
    attend = self.att_w(emb_h).view(enc_sents.size(0),
                                    enc_sents.size(1))
    mask = self._list_to_bytemask(list(lens.tolist))
    all_att = self._masked_softmax(attend, mask)
    try:
      attended = all_att.unsqueeze(2).expand_as(enc_sents) * enc_sents
    except:
      embed()
      raise

    _, _indices = torch.sort(indices, 0)
    return attended.sum(1,True).squeeze(1)[_indices.cuda()], all_att

  def _list_to_bytemask(self,l):
    mask = torch.FloatTensor(len(l),l[0]).fill_(1)

    for i,j in enumerate(l):
      if j != l[0]: mask[i,j:l[0]] = 0

    return mask.cuda()

  def _masked_softmax(self,mat,mask):
    exp = torch.exp(mat) * Variable(mask,requires_grad=False)
    sum_exp = exp.sum(1,True)+0.0001

    return exp/sum_exp.expand_as(exp)

class Maxout(nn.Module):
  def __init__(self, embedding_features, rnn_features, rnn_bidirectional=False):
    super(Maxout, self).__init__()
    self.bidirectional = rnn_bidirectional

    self.rnn = nn.GRU(input_size=embedding_features,
              hidden_size=rnn_features,
              num_layers=1, batch_first=True,
              bidirectional=False)

    self.features = rnn_features

    self._init_rnn(self.rnn.weight_ih_l0)
    self._init_rnn(self.rnn.weight_hh_l0)
    self.rnn.bias_ih_l0.data.zero_()
    self.rnn.bias_hh_l0.data.zero_()

  def _init_rnn(self, weight):
    for w in weight.chunk(3, 0):
      init.xavier_uniform(w)

  def forward(self, q_emb, q_len, hidden=None,pooling=False):
    lengths = q_len.long()
    lens, indices = torch.sort(lengths, 0, True)

    packed_batch = pack_padded_sequence(q_emb[indices.cuda()], lens.tolist(), batch_first=True)
    if hidden is not None:
      N_, H_ = hidden.size()
      hs, _ = self.rnn(packed_batch, hidden[indices.cuda()].view(1, N_, H_))
    else:
      hs, _ = self.rnn(packed_batch)
    outputs, _ = pad_packed_sequence(hs, batch_first=True)
    _, _indices = torch.sort(indices, 0)
    outputs = outputs[_indices.cuda()]
    if pooling:
     return outputs
    N, L, H = outputs.size()
    maxout = []
    for batch_ind, length in enumerate(lengths.tolist()):
      maxout.append( F.max_pool1d(outputs[batch_ind, :length, :].view(1, length, H).transpose(1, 2), length).squeeze().view(1, -1) )

    return torch.cat(maxout)
