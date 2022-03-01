import torch 
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import itertools
from math import log, exp
import numpy as np
from sys import exit as e

from model import Glow
from loss import CustomLoss
import util as ut
from args import get_args

from torchvision import io

class DatasetMoons:
  """ two half-moons """
  def sample(self, n):
    moons, target = datasets.make_moons(n_samples=n, noise=0.05)
    moons = moons.astype(np.float32)
    return torch.from_numpy(moons), torch.from_numpy(target)

  def sample_gauss(self, n):
    X,y = make_classification(n_samples=n, n_features=2, n_informative=2, \
      n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2,class_sep=2,\
        flip_y=0,weights=[0.5,0.5], random_state=17)
    return torch.from_numpy(X), torch.from_numpy(y)
  
  def sample_circles(self, n):
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    print(noisy_circles.shape)

  def sample_iris(self):
    iris = load_iris()
    X = iris.data
    y = iris.target
    return torch.from_numpy(X), torch.from_numpy(y)

def get_dataset(n_samples):
  moon, labels = make_moons(n_samples, noise=0.05)
  return moon.astype(np.float32), labels

if __name__ == "__main__":
  opt = get_args()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)

  ctype = opt.config
  config = ut.get_config("./config.yaml")[ctype]

  epochs = int(config['epochs'])
  lr = float(config['lr']) #1e-3
  lr2 = float(config['lr2']) #1e-4
  bsize = int(config['bsize']) #256
  n_chan = int(config['n_chan']) #2
  n_class = int(config['n_class']) #2
  n_flow = int(config['n_flow']) #8
  torch.manual_seed(0)
  d = DatasetMoons()
  writer = SummaryWriter()

  # epochs = 1000
  # n_class = 2
  model = Glow(n_chan, n_flow)
  print("number of params: ", sum(p.numel() for p in model.parameters()))
  # params = []
  # for block in model.flows:
  #     params.extend(list(block.prior.parameters()))
  # e()

  optimizer =  optim.Adam(model.parameters(), lr=lr)
  optimizer2 =  optim.Adam(model.prior.parameters(), lr=lr2)
  # optimizer3 =  optim.Adam(model.prior.parameters(), lr=lr)
  criterion = CustomLoss(n_class)
  best_loss = 1e5

  z_rec = torch.FloatTensor(256, 2).normal_(0, 1)

  model.train()
  for k in range(epochs):
    x, target = get_dataset(bsize)
    x = torch.from_numpy(x)
    target = torch.from_numpy(target)

    # z, logdet, prior_logprob = model(x)
    # logdet = logdet.mean()
    # logprob = prob + logdet
    # loss = -torch.mean(logprob) # NLL

    z, logdet, mean, log_sd, prob = model(x)
    log_prob, mus_per_class, log_sds_per_class, prob_x, log_p_total = criterion(z, mean, log_sd, target, logdet)
    # sim_loss, diff_loss = criterion.b_loss(z, target, mus_per_class, log_sds_per_class)

    loss = -log_prob
    # loss2 = -sim_loss
    # loss3 = diff_loss
    # loss = loss + loss2
    loss2 = torch.tensor(1., requires_grad=True)
    loss3 = torch.tensor(1., requires_grad=True)

    model.zero_grad()
    loss.backward(retain_graph=False)
    avg = ut.compute_avg_grad(model)
    # loss2.backward(retain_graph=True)
    grad2 = ut.compute_avg_grad(model) - avg
    # loss3.backward(retain_graph=False)
    grad3 = ut.compute_avg_grad(model) - avg - grad2
    
    optimizer.step()
    # optimizer2.step()
    # optimizer3.step()

    # nn.utils.clip_grad_norm(model.parameters(), 0.5)

    if loss.item() < best_loss:
      torch.save(model.state_dict(), f"./models/best_model.pt")
      best_loss = loss.item()

    if k % 100 == 0:
      # print(mus_per_class, log_sds_per_class)
      # ut.plot_grad_flow(model.named_parameters(), f"./plots/grad/{k}_high.png")
      ut.plot(z.detach(), target.detach(), k)
      z_rec = torch.normal(mean, log_sd.exp())
      ut.plot(model.reverse(z_rec).detach(), target, f"recon_{k}.png")
      ut.plot_3d(z.detach(), prob.detach(), k, target)
      print(f"NLL: {round(loss.item(), 4)} BLL: {round(loss2.item(), 4), round(loss3.item(), 4)} \
        lr: {optimizer.param_groups[0]['lr']} NLL gradients: {round(avg, 4)} \
          BLL gradients: {round(grad2, 4), round(grad3, 4)} epoch: {k} log_p_total: {log_p_total.exp().data}")
      # print(loss.item(), loss2.item(), loss3.item(), avg, grad2, grad3)
  
  log_prob, mus_per_class, log_sds_per_class, prob_x, log_p_total = criterion(z, mean, log_sd, target, logdet)
  print(f"Mean, log_sd {mus_per_class, log_sds_per_class}")
  print(f"Best loss at {best_loss}")
  