import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from IPython import embed
from atten import *
from layers import *
from loss import *
from decoder.loss import *
from decoder.model import *
from decoder.layers import *
import time

class EncoderImage(nn.Module):
  def __init__(self, img_dim, embed_size, bidirectional=False, rnn_type='maxout'):
    super(EncoderImage, self).__init__()
    self.embed_size = embed_size
    self.bidirectional = bidirectional

    if rnn_type == 'attention':
      self.rnn = Attention(img_dim, embed_size, rnn_bidirectional=bidirectional)
    elif rnn_type == 'seq2seq':
      self.rnn = Seq2Seq(img_dim, embed_size, rnn_bidirectional=bidirectional)
    elif rnn_type == 'maxout':
      self.rnn = Maxout(img_dim, embed_size, rnn_bidirectional=bidirectional)
    else:
      raise ValueError('Unsupported RNN type')

  def forward(self, x, lengths , pooling=False):
    """Extract image feature vectors."""
    outputs = self.rnn(x, lengths,None , pooling)

    # normalization in the joint embedding space
    # return F.normalize(outputs)
    return outputs

class EncoderSequence(nn.Module):
  def __init__(self, img_dim, embed_size, bidirectional=False, rnn_type='maxout'):
    super(EncoderSequence, self).__init__()
    self.embed_size = embed_size
    self.bidirectional = bidirectional

    if rnn_type == 'attention':
      self.rnn = Attention(img_dim, embed_size, rnn_bidirectional=bidirectional)
    elif rnn_type == 'seq2seq':
      self.rnn = Seq2Seq(img_dim, embed_size, rnn_bidirectional=bidirectional)
    elif rnn_type == 'maxout':
      self.rnn = Maxout(img_dim, embed_size, rnn_bidirectional=bidirectional)
    else:
      raise ValueError('Unsupported RNN type')

  def forward(self, x, lengths, hidden=None , pooling2=False):
    """Extract image feature vectors."""
    outputs = self.rnn(x, lengths,hidden ,pooling2)

    # normalization in the joint embedding space
    # return F.normalize(outputs)
    return outputs

class EncoderText(nn.Module):
  def __init__(self, vocab_size, word_dim, embed_size,
      bidirectional=False, rnn_type='maxout', data_name='anet_precomp'):
    super(EncoderText, self).__init__()
    self.embed_size = embed_size
    self.bidirectional = bidirectional

    # word embedding
    self.embed   = nn.Embedding(vocab_size, word_dim)

    # caption embedding
    if rnn_type == 'attention':
      self.rnn = Attention(word_dim, embed_size, rnn_bidirectional=bidirectional)
    elif rnn_type == 'seq2seq':
      self.rnn = Seq2Seq(word_dim, embed_size, rnn_bidirectional=bidirectional)
    elif rnn_type == 'maxout':
      self.rnn = Maxout(word_dim, embed_size, rnn_bidirectional=bidirectional)
    else:
      raise ValueError('Unsupported RNN type')

    self.init_weights(data_name)

  def init_weights(self, data_name):
    self.embed.weight.data = torch.from_numpy(np.load('vocab/{}_w2v_total.npz'.format(data_name))['arr_0'].astype(float)).float()

  def forward(self, x, lengths , pooling=False):
    # Embed word ids to vectors
    cap_emb = self.embed(x)
    outputs = self.rnn(cap_emb, lengths,None , pooling)

    # normalization in the joint embedding space
    # return F.normalize(outputs), cap_emb
    return outputs #, cap_emb


class VSE(nn.Module):
  def __init__(self, opt):
    super(VSE, self).__init__()
    self.losses = []
    self.norm = opt.norm
    self.grad_clip = opt.grad_clip
    # Ameen : Try Lower Space for the M matrix.
    self.M = ScoresMatch(embed_x_size=1024 , embed_y_size=1024)
    self.attention = Atten(util_e = [1024 ,1024])
    self.attention_context = Atten(util_e = [1024 , 1024])

    self.clip_enc = EncoderImage(opt.img_dim, opt.img_first_size,
                  rnn_type=opt.rnn_type)
    self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim, opt.cap_first_size,
                  rnn_type=opt.rnn_type, data_name = opt.data_name)
    self.vid_seq_enc = EncoderSequence(opt.img_first_size, opt.embed_size,
                  rnn_type=opt.rnn_type)
    self.txt_seq_enc = EncoderSequence(opt.cap_first_size, opt.embed_size,
                  rnn_type=opt.rnn_type)
    #self.mlp = MLP_Match(opt.img_first_size)
    #self.mlp_criterion = nn.BCELoss()

    if torch.cuda.is_available():
      self.M.cuda()
      self.clip_enc.cuda()
      self.attention.cuda()
      self.attention_context.cuda()
      self.txt_enc.cuda()
      self.vid_seq_enc.cuda()
      self.txt_seq_enc.cuda()
      cudnn.benchmark = True

    # Loss and Optimizer
    self.criterion = ContrastiveLoss(margin=opt.margin,
                     measure=opt.measure,
                     max_violation=opt.max_violation, norm=self.norm)

    self.weak_criterion = GroupWiseContrastiveLoss(margin=opt.margin,
                     measure=opt.measure,
                     max_violation=opt.max_violation, norm=self.norm)

    self.negs_criterion = Score_Loss(margin=opt.margin , max_violation=opt.max_violation)

    params = list(self.txt_enc.parameters())
    params += list(self.M.parameters())
    params += list(self.clip_enc.parameters())
    params += list(self.vid_seq_enc.parameters())
    params += list(self.txt_seq_enc.parameters())
    params += list(self.attention.parameters())
    params += list(self.attention_context.parameters())

    self.params = params

    self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

    self.Eiters = 0

  def state_dict(self, opt):
    state_dict = [self.txt_enc.state_dict(), self.M.state_dict(), \
                  self.clip_enc.state_dict(), self.vid_seq_enc.state_dict() ,  self.txt_seq_enc.state_dict() , self.attention.state_dict() , self.attention_context.state_dict()]
    return state_dict

  def load_state_dict(self, state_dict, opt):
    self.txt_enc.load_state_dict(state_dict[0])
    self.M.load_state_dict(state_dict[1])
    self.clip_enc.load_state_dict(state_dict[2])
    self.vid_seq_enc.load_state_dict(state_dict[3])
    self.txt_seq_enc.load_state_dict(state_dict[4])
    self.attention.load_state_dict(state_dict[5])
    self.attention_context.load_state_dict(state_dict[6])

  def train_start(self, opt):
    """switch to train mode
    """
    self.attention.train()
    self.attention_context.train()
    self.M.train()
    self.clip_enc.train()
    self.txt_enc.train()
    self.vid_seq_enc.train()
    self.txt_seq_enc.train()


  def val_start(self, opt):
    """switch to evaluate mode
    """
    self.clip_enc.eval()
    self.M.eval()
    self.attention.eval()
    self.attention_context.eval()
    self.txt_enc.eval()
    self.vid_seq_enc.eval()
    self.txt_seq_enc.eval()

  def forward_emb(self, clips, captions, lengths_clip, lengths_cap):
    clips    = Variable(clips)
    captions = Variable(captions)
    if torch.cuda.is_available():
      clips = clips.cuda()
      captions = captions.cuda()

    # Forward
    clip_emb = self.clip_enc(clips, Variable(lengths_clip) , pooling=True)
    cap_emb = self.txt_enc(captions, Variable(lengths_cap) , pooling=True)
    attented = self.attention(utils=[clip_emb , cap_emb])
    clip_emb_attented = attented[0]
    cap_emb_attented = attented[1]
    return clip_emb_attented , cap_emb_attented , clip_emb , cap_emb

  def forward_emb_context(self, videos, paragraphs, lengths_video , lengths_paragraph):
    videos     = Variable(videos)
    paragraphs = Variable(paragraphs)
    if torch.cuda.is_available():
      videos = videos.cuda()
      paragraphs = paragraphs.cuda()

    # Forward
    video_emb = self.clip_enc(videos, Variable(lengths_video) , pooling=True)
    paragraph_emb = self.txt_enc(paragraphs, Variable(lengths_paragraph) , pooling=True)

    attented = self.attention_context(utils=[video_emb , paragraph_emb])
    video_emb = attented[0]
    paragraph_emb = attented[1]

    return video_emb, paragraph_emb

  def structure_emb(self, clip_emb, cap_emb, num_clips, num_caps, vid_context=None, para_context=None):
    img_reshape_emb = Variable(torch.zeros(len(num_clips), max(num_clips), clip_emb.shape[1])).cuda()
    cap_reshape_emb = Variable(torch.zeros(len(num_caps),  max(num_caps),  cap_emb.shape[1])).cuda()

    cur_displace = 0
    for i, end_place in enumerate(num_clips):
      img_reshape_emb[i, 0:end_place, :] = clip_emb[cur_displace : cur_displace + end_place, :]
      cur_displace = cur_displace + end_place

    cur_displace = 0
    for i, end_place in enumerate(num_caps):
      cap_reshape_emb[i, 0:end_place, :] = cap_emb[cur_displace : cur_displace + end_place, :]
      cur_displace = cur_displace + end_place

    vid_emb  = self.vid_seq_enc(img_reshape_emb, Variable(torch.Tensor(num_clips).long()), vid_context)
    para_emb = self.txt_seq_enc(cap_reshape_emb, Variable(torch.Tensor(num_caps).long()), para_context)

    return vid_emb, para_emb

  def forward_loss(self, clip_emb, cap_emb, name, **kwargs):
    """Compute the loss given pairs of image and caption embeddings
    """
    loss = self.criterion(clip_emb, cap_emb)
    self.logger.update('Le'+name, loss.item(), clip_emb.size(0))
    return loss


  def cosine_similarity(self , im, s):
   return im.mm(s.t())

  def scores(self , clips, captions, cap_lens , clip_lens , vids , paras , vid_len , para_lens , num_clips , num_caps):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    num_segments_per_video = num_clips[0]
    clips    = Variable(clips)
    captions = Variable(captions)
#    vids = Variable(vids)
#    paras = Variable(paras)
    if torch.cuda.is_available():
      clips = clips.cuda()
      captions = captions.cuda()
      vids = vids.cuda()
      paras = paras.cuda()

    clips     = self.clip_enc(clips  , Variable(clip_lens) , pooling=True)
    captions  = self.txt_enc(captions, Variable(cap_lens)   , pooling=True)

    similarities_clips_captions    = []
    similarities_videos_paragraphs = []
    n_clip = clips.size(0)
    n_caption = captions.size(0)
    embedding_dim = clips.shape[-1]
    num_videos     = vid_len.shape[0]
    num_paragraphs = para_lens.shape[0]
    paragraph_iter = 1
    mask = torch.zeros(n_clip)
    for iter in range(n_clip):
     if iter % num_segments_per_video == 0:
      mask[iter] = 1

    video_clips = []
    video_sentences = []

    for i in range(n_caption):
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        cap_i_expand = cap_i.repeat(n_clip, 1, 1)
        clips_attented , caps_attented = self.attention(utils=[clips , cap_i_expand.cuda()])

        tmp_clips     = clips_attented[mask == 1]
        tmp_sentences = caps_attented[mask == 1]
        video_clips.append(tmp_clips.unsqueeze(1))
        video_sentences.append(tmp_sentences.unsqueeze(1))


        if (i+1) % num_segments_per_video == 0 :
         n_para = para_lens[paragraph_iter-1]
         len_para_expand = n_para.repeat(num_videos)
         target_para = paras[paragraph_iter-1 , : n_para].unsqueeze(0).contiguous()
         target_para = target_para.repeat(num_videos , 1)

         video_clips = tuple(video_clips)
         video_sentences = tuple(video_sentences)

         new_clip_emb_attented = torch.cat(video_clips     , dim=1).view(n_clip,embedding_dim)
         new_cap_emb_attented  = torch.cat(video_sentences , dim=1).view(n_clip,embedding_dim)
         #new_clip_emb_attented = torch.stack(video_clips,dim=0).view(n_clip,embedding_dim).t().contiguous().view(n_clip,embedding_dim)
         #new_cap_emb_attented  = torch.stack(video_sentences,dim=0).view(n_caption,embedding_dim).t().contiguous().view(n_clip,embedding_dim)

         vid_context, para_context = self.forward_emb_context(vids, target_para , vid_len, len_para_expand)
         vid_emb, para_emb = self.structure_emb(new_clip_emb_attented, new_cap_emb_attented, num_clips, num_caps, vid_context, para_context)
         vids_paras_scores = row_wise_matrix_mul(vid_emb , para_emb , num_videos , embedding_dim).unsqueeze(1)
         similarities_videos_paragraphs.append(vids_paras_scores)
         video_clips = []
         video_sentences = []
         paragraph_iter += 1

        mask = torch.roll(mask , 1 , 0)
        clips_captions_score = self.M(clips_attented.unsqueeze(0) , caps_attented.unsqueeze(0).cuda()).unsqueeze(1)
        similarities_clips_captions.append(clips_captions_score)
    similarities_clips_captions    = torch.cat(similarities_clips_captions, 1)
    similarities_videos_paragraphs = torch.cat(similarities_videos_paragraphs, 1)
    return similarities_clips_captions , similarities_videos_paragraphs

  def train_emb(self, opts, clips, captions, videos, paragraphs,
      lengths_clip, lengths_cap, lengths_video, lengths_paragraph,
      num_clips, num_caps, ind, cur_vid, *args):
    """One training step given clips and captions.
    """
    self.Eiters += 1
    self.logger.update('Eit', self.Eiters)
    self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

    # compute the embeddings
#    clip_emb_attented, cap_emb_attented , clip_emb , cap_emb = self.forward_emb(clips, captions, lengths_clip, lengths_cap)
#    vid_context, para_context = self.forward_emb_context(videos, paragraphs, lengths_video, lengths_paragraph)
#    vid_emb, para_emb = self.structure_emb(clip_emb_attented, cap_emb_attented, num_clips, num_caps, vid_context, para_context)

    # measure accuracy and record loss
    self.optimizer.zero_grad()

    #loss_1 = self.forward_loss(F.normalize(vid_emb), F.normalize(para_emb), '_vid')
    #loss = loss + loss_1
    loss = 0
    scores_clips_sentences , scores_videos_paragraphs  = self.scores(clips , captions , lengths_cap , lengths_clip ,
                                                                    videos , paragraphs , lengths_video, lengths_paragraph,
                                                                    num_clips, num_caps)
    loss_1 = self.negs_criterion(scores_clips_sentences)
    loss_2 = self.negs_criterion(scores_videos_paragraphs)
    self.logger.update('M_Loss ', loss_1.item())
    self.logger.update('Videos_And_Pargraphs_Loss ', loss_2.item())
    loss = loss + loss_1 + loss_2

    # compute gradient and do SGD step
    loss.backward()
    if self.grad_clip > 0: clip_grad_norm(self.params, self.grad_clip)
    self.optimizer.step()
