import torch
from random import randrange
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import os.path as osp
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
import h5py
import copy
from IPython import embed

this_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))
class PrecompDataset(data.Dataset):
  """
  Load precomputed captions and image features
  Possible options: f8k, f30k, coco, 10crop
  """

  def __init__(self, data_path, data_split, vocab, opt):
    self.vocab = vocab
    json_path = osp.join(this_dir, 'data', 'captions', '{}.json'.format(data_split) )

    # Captions
    self.jsondict = jsonmod.load(open(json_path, 'r'))
    self.ann_id = {}
    for i, keys in enumerate(self.jsondict.keys()):
      if i == 10000:break
      self.ann_id[i] = keys

    # Image features
#    self.video_emb = h5py.File(osp.join(this_dir, 'data', 'anet_precomp', 'sub_activitynet_v1-3.c3d.hdf5-0'),'r')

    self.length = len(self.ann_id)
    self.feat_name = opt.feat_name

  def img_cap_feat_combine(self, video_feat, caption_feat, cur_vid):
    #### Videos ####
    frame_duration = self.jsondict[cur_vid]['duration']
    segment_number = len(self.jsondict[cur_vid]['timestamps'])
    video_feat_len = video_feat.shape[0]

    clips = []
    for seg_id in range(segment_number):
      picked_segment = self.jsondict[cur_vid]['timestamps'][seg_id]
      start_frame = int(np.floor(picked_segment[0] / frame_duration * video_feat_len))
      end_frame = int(np.ceil(picked_segment[1] / frame_duration * video_feat_len))

      if start_frame > end_frame:end_frame = start_frame

      current_feat = video_feat[start_frame: end_frame+1, :]
      max_frames = 80.0
      if current_feat.shape[0] > max_frames:
        ind = np.arange(0, current_feat.shape[0], current_feat.shape[0]/max_frames).astype(int).tolist()
        current_feat = current_feat[ind,:]

      clips.append(current_feat)

    video = video_feat
    if video.shape[0] > max_frames:
      ind = np.arange(0, video.shape[0], video.shape[0]/max_frames).astype(int).tolist()
      video = video[ind,:]

    #### Captions ####
    vocab = self.vocab

    captions = []
    length_cap = []
    for cap in caption_feat:
#      n = randrange(self.length)
#      tmp = list(self.jsondict.keys())[n]
#      tmp = self.jsondict[tmp]['sentences'][0]
#      print(tmp)
      tokens_cap = nltk.tokenize.word_tokenize(cap.lower())
     # tokens_cap = nltk.tokenize.word_tokenize(tmp)
      _cap = []
      _cap.extend([vocab(token) for token in tokens_cap])
#      print(type(_cap))
      target_cap = torch.Tensor(_cap)
#      print(target_cap)
#      exit(-1)
      captions.append(target_cap)

    paragraph = torch.cat(captions, 0)

    lengths_clip = [len(vid) for vid in clips]
    lengths_cap = [len(cap) for cap in captions]

    num_clip    = len(clips)
    num_caption = len(captions)
    assert num_clip == num_caption, 'something is wrong: {} vs. {}'.format(num_clip, num_caption)

    lengths_clip = torch.Tensor(lengths_clip).long()
    lengths_cap = torch.Tensor(lengths_cap).long()

    clips_stack = torch.cat(clips,0)
    word_stack = torch.cat(captions,0)

    return clips, captions, video, paragraph, lengths_clip, lengths_cap, num_clip, num_caption

  def __getitem__(self, index):
    # handle the image redundancy
    cur_vid = self.ann_id[index]
    if self.feat_name == 'c3d':
        image_data = self.video_emb[cur_vid]['c3d_features'].value
    if self.feat_name == 'icep':
        image_data = np.load(os.path.join(this_dir,'data', 'anet_precomp', 'ICEP_V3_global_pool_skip_8_direct_resize/'+cur_vid+'.npz'))['frame_scores'].squeeze()
#    image_data = self.video_emb[cur_vid]['c3d_features'].value
    image = torch.Tensor(image_data)
    caption_json = self.jsondict[cur_vid]['sentences']

    clips, captions, video, paragraph, lengths_clip, lengths_cap, \
      num_clip, num_caption = self.img_cap_feat_combine(image, caption_json, cur_vid)

    return clips, captions, video, paragraph, lengths_clip, lengths_cap, num_clip, num_caption, index, cur_vid

  def __len__(self):
    return self.length

def collate_fn(data_batch):
  _clips, _captions, _video, _paragraph, _lengths_clip, _lengths_cap, _num_clip, _num_caption, _index, _cur_vid = zip(*data_batch)

  # Merge images
  lengths_clip = torch.cat(_lengths_clip, 0)
  clips = torch.zeros(len(lengths_clip), lengths_clip.max(), _clips[0][0].shape[1])
  _cur_ind = 0
  for i, _vid_seg in enumerate(_clips):
    for j, vid in enumerate(_vid_seg):
      end = lengths_clip[_cur_ind]
      clips[_cur_ind, :end, :] = vid[:end, :]
      _cur_ind += 1

  lengths_video = torch.Tensor([len(x) for x in _video]).long()
  videos = torch.zeros(len(_video), lengths_video.max(), _video[0].shape[1])
  for i, vid in enumerate(_video):
    end = lengths_video[i]
    videos[i, :end, :] = vid[:end, : ]

  # Merget captions
  lengths_cap = torch.cat(_lengths_cap, 0)
  captions = torch.zeros(len(lengths_cap), lengths_cap.max()).long()
  _cur_ind = 0
  for i, _cap_seg in enumerate(_captions):
    for j, cap in enumerate(_cap_seg):
      end = lengths_cap[_cur_ind]
      captions[_cur_ind, :end] = cap[:end]
      _cur_ind += 1

  lengths_paragraph = torch.Tensor([len(x) for x in _paragraph]).long()
  paragraphs = torch.zeros(len(_paragraph), lengths_paragraph.max()).long()
  for i, cap in enumerate(_paragraph):
    end = lengths_paragraph[i]
    paragraphs[i, :end] = cap[:end ]
  return clips, captions, videos, paragraphs, lengths_clip, lengths_cap, lengths_video, lengths_paragraph, _num_clip, _num_caption, _index, _cur_vid

def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=64,
             shuffle=True, num_workers=2):
  """Returns torch.utils.data.DataLoader for custom coco dataset."""
  dset = PrecompDataset(data_path, data_split, vocab, opt)

  data_loader = torch.utils.data.DataLoader(dataset=dset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True,
                        collate_fn=collate_fn,
                        num_workers=num_workers)
  return data_loader

def get_loaders(data_name, vocab, batch_size, workers, opt):
  dpath = os.path.join(opt.data_path, data_name)
  if opt.data_name.endswith('_precomp'):
      if opt.feat_name == 'c3d':
          workers = 1
      train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                      batch_size, True, workers)
      val_loader   = get_precomp_loader(dpath, 'val_1', vocab, opt,
                    batch_size, False, workers)
  return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
          workers, opt):
  dpath = os.path.join(opt.data_path, data_name)
  if opt.data_name.endswith('_precomp'):
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                     batch_size, False, workers)
  return test_loader

