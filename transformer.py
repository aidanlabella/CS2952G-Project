import math
import gc
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from fastai.vision.all import *
import fastai

# Source: https://www.kaggle.com/code/iafoss/rna-starter-0-186-lb

# Globals
PATH = '/users/mzuo6/genomics/data/'
OUT = './'
bs = 256
num_workers = 2
SEED = 2023
nfolds = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_everything(seed: int):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True


def flatten(o):
  "Concatenate all collections and items as a generator"
  for item in o:
    if isinstance(o, dict):
      yield o[item]
      continue
    elif isinstance(item, str):
      yield item
      continue
    try:
      yield from flatten(item)
    except TypeError:
      yield item


@delegates(GradScaler)
class MixedPrecision(Callback):
  "Mixed precision training using Pytorch's `autocast` and `GradScaler`"
  order = 10

  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def before_fit(self):
    self.autocast, self.learn.scaler, self.scales = autocast(), GradScaler(
        **self.kwargs), L()

  def before_batch(self):
    self.autocast.__enter__()

  def after_pred(self):
    if next(flatten(self.pred)).dtype == torch.float16:
      self.learn.pred = to_float(self.pred)

  def after_loss(self):
    self.autocast.__exit__(None, None, None)

  def before_backward(self):
    self.learn.loss_grad = self.scaler.scale(self.loss_grad)

  def before_step(self):
    "Use `self` as a fake optimizer. `self.skipped` will be set to True `after_step` if gradients overflow. "
    self.skipped = True
    self.scaler.step(self)
    if self.skipped:
      raise CancelStepException()
    self.scales.append(self.scaler.get_scale())

  def after_step(self):
    self.learn.scaler.update()

  @property
  def param_groups(self):
    "Pretend to be an optimizer for `GradScaler`"
    return self.opt.param_groups

  def step(self, *args, **kwargs):
    "Fake optimizer step to detect whether this batch was skipped from `GradScaler`"
    self.skipped = False

  def after_fit(self):
    self.autocast, self.learn.scaler, self.scales = None, None, None


fastai.callback.fp16.MixedPrecision = MixedPrecision


class RNA_Dataset(Dataset):

  def __init__(
      self,
      df: pd.DataFrame,
      mode: str = 'train',
      seed: int = 2023,
      fold: int = 0,
      nfolds: int = 4,
      mask_only: bool = False,
      **kwargs,
  ):
    self.seq_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    self.Lmax = 206
    df['L'] = df.sequence.apply(len)
    df_2A3 = df.loc[df.experiment_type == '2A3_MaP']
    df_DMS = df.loc[df.experiment_type == 'DMS_MaP']

    split = list(
        KFold(n_splits=nfolds, random_state=seed,
              shuffle=True).split(df_2A3))[fold][0 if mode == 'train' else 1]
    df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
    df_DMS = df_DMS.iloc[split].reset_index(drop=True)

    m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
    df_2A3 = df_2A3.loc[m].reset_index(drop=True)
    df_DMS = df_DMS.loc[m].reset_index(drop=True)

    self.seq = df_2A3['sequence'].values
    self.L = df_2A3['L'].values

    self.react_2A3 = df_2A3[[c for c in df_2A3.columns if \
                             'reactivity_0' in c]].values
    self.react_DMS = df_DMS[[c for c in df_DMS.columns if \
                             'reactivity_0' in c]].values
    self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if \
                             'reactivity_error_0' in c]].values
    self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if \
                            'reactivity_error_0' in c]].values
    self.sn_2A3 = df_2A3['signal_to_noise'].values
    self.sn_DMS = df_DMS['signal_to_noise'].values
    self.mask_only = mask_only

  def __len__(self):
    return len(self.seq)

  def __getitem__(self, idx):
    seq = self.seq[idx]
    if self.mask_only:
      mask = torch.zeros(self.Lmax, dtype=torch.bool)
      mask[:len(seq)] = True
      return {'mask': mask}, {'mask': mask}
    seq = [self.seq_map[s] for s in seq]
    seq = np.array(seq)
    mask = torch.zeros(self.Lmax, dtype=torch.bool)
    mask[:len(seq)] = True
    seq = np.pad(seq, (0, self.Lmax - len(seq)))

    react = torch.from_numpy(
        np.stack([self.react_2A3[idx], self.react_DMS[idx]], -1))
    react_err = torch.from_numpy(
        np.stack([self.react_err_2A3[idx], self.react_err_DMS[idx]], -1))
    sn = torch.FloatTensor([self.sn_2A3[idx], self.sn_DMS[idx]])

    return {'seq': torch.from_numpy(seq), 'mask': mask}, \
           {'react': react, 'react_err': react_err,
            'sn': sn, 'mask': mask}


class LenMatchBatchSampler(torch.utils.data.BatchSampler):

  def __iter__(self):
    buckets = [[]] * 100
    yielded = 0

    for idx in self.sampler:
      s = self.sampler.data_source[idx]
      if isinstance(s, tuple):
        L = s[0]["mask"].sum()
      else:
        L = s["mask"].sum()
      L = max(1, L // 16)
      if len(buckets[L]) == 0:
        buckets[L] = []
      buckets[L].append(idx)

      if len(buckets[L]) == self.batch_size:
        batch = list(buckets[L])
        yield batch
        yielded += 1
        buckets[L] = []

    batch = []
    leftover = [idx for bucket in buckets for idx in bucket]

    for idx in leftover:
      batch.append(idx)
      if len(batch) == self.batch_size:
        yielded += 1
        yield batch
        batch = []

    if len(batch) > 0 and not self.drop_last:
      yielded += 1
      yield batch


def dict_to(x, device='cuda'):
  return {k: x[k].to(device) for k in x}


def to_device(x, device='cuda'):
  return tuple(dict_to(e, device) for e in x)


class DeviceDataLoader:

  def __init__(self, dataloader, device='cuda'):
    self.dataloader = dataloader
    self.device = device

  def __len__(self):
    return len(self.dataloader)

  def __iter__(self):
    for batch in self.dataloader:
      yield tuple(dict_to(x, self.device) for x in batch)


class SinusoidalPosEmb(nn.Module):

  def __init__(self, dim=16, M=10000):
    super().__init__()
    self.dim = dim
    self.M = M

  def forward(self, x):
    device = x.device
    half_dim = self.dim // 2
    emb = math.log(self.M) / half_dim
    emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
    emb = x[..., None] * emb[None, ...]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb


class RNA_Model(nn.Module):

  def __init__(self, dim=192, depth=12, head_size=32, **kwargs):
    super().__init__()
    self.emb = nn.Embedding(4, dim)
    self.pos_enc = SinusoidalPosEmb(dim)
    self.transformer = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=dim,
                                   nhead=dim // head_size,
                                   dim_feedforward=4 * dim,
                                   dropout=0.1,
                                   activation=nn.GELU(),
                                   batch_first=True,
                                   norm_first=True), depth)
    self.proj_out = nn.Linear(dim, 2)

  def forward(self, x0):
    mask = x0['mask']
    Lmax = mask.sum(-1).max()
    mask = mask[:, :Lmax]
    x = x0['seq'][:, :Lmax]

    pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
    pos = self.pos_enc(pos)
    x = self.emb(x)
    x = x + pos

    x = self.transformer(x, src_key_padding_mask=~mask)
    x = self.proj_out(x)

    return x


def loss(pred, target):
  p = pred[target['mask'][:, :pred.shape[1]]]
  y = target['react'][target['mask']].clip(0, 1)
  loss = F.l1_loss(p, y, reduction='none')
  loss = loss[~torch.isnan(loss)].mean()

  return loss


class MAE(Metric):

  def __init__(self):
    self.reset()

  def reset(self):
    self.x, self.y = [], []

  def accumulate(self, learn):
    x = learn.pred[learn.y['mask'][:, :learn.pred.shape[1]]]
    y = learn.y['react'][learn.y['mask']].clip(0, 1)
    self.x.append(x)
    self.y.append(y)

  @property
  def value(self):
    x, y = torch.cat(self.x, 0), torch.cat(self.y, 0)
    loss = F.l1_loss(x, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    return loss


if __name__ == '__main__':
  seed_everything(SEED)
  os.makedirs(OUT, exist_ok=True)
  # ensure data is stored as float32 not float64
  df = pd.read_parquet(os.path.join(PATH, 'train_data.parquet'))

  for fold in [0]:  # running multiple folds at kaggle may cause OOM
    ds_train = RNA_Dataset(df, mode='train', fold=fold, nfolds=nfolds)
    ds_train_len = RNA_Dataset(df,
                               mode='train',
                               fold=fold,
                               nfolds=nfolds,
                               mask_only=True)
    sampler_train = torch.utils.data.RandomSampler(ds_train_len)
    len_sampler_train = LenMatchBatchSampler(sampler_train,
                                             batch_size=bs,
                                             drop_last=True)
    dl_train = DeviceDataLoader(
        torch.utils.data.DataLoader(ds_train,
                                    batch_sampler=len_sampler_train,
                                    num_workers=num_workers,
                                    persistent_workers=True), device)

    ds_val = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds)
    ds_val_len = RNA_Dataset(df,
                             mode='eval',
                             fold=fold,
                             nfolds=nfolds,
                             mask_only=True)
    sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
    len_sampler_val = LenMatchBatchSampler(sampler_val,
                                           batch_size=bs,
                                           drop_last=False)
    dl_val = DeviceDataLoader(
        torch.utils.data.DataLoader(ds_val,
                                    batch_sampler=len_sampler_val,
                                    num_workers=num_workers), device)
    gc.collect()

    data = DataLoaders(dl_train, dl_val)
    model = RNA_Model()
    model = model.to(device)
    learn = Learner(
        data,
        model,
        loss_func=loss,
        cbs=[GradientClip(3.0)],
        metrics=[MAE()],
    )
    #fp16 doesn't help at P100 but gives x1.6-1.8 speedup at modern hardware

    learn.fit_one_cycle(32, lr_max=5e-4, wd=0.05, pct_start=0.02)
    torch.save(
        learn.model.state_dict(),
        os.path.join(OUT, f'{fname}_{fold}.pth'),
    )
    gc.collect()
