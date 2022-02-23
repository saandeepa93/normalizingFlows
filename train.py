import torch 
from torch import nn, optim
from torch.nn.init import xavier_normal_, zeros_, normal_, uniform_
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification, load_iris
from sklearn import datasets
import itertools
import numpy as np
import time
from math import log, pi, sqrt, exp
from sys import exit as e

# from model import Glow
from model import Invertible1x1Conv, ActNorm, AffineHalfFlow, NormalizingFlowModel 
from loss import CustomLoss
from args import get_args
import util as ut

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

  def sample_iris(self):
    iris = load_iris()
    X = iris.data
    y = iris.target
    return torch.from_numpy(X), torch.from_numpy(y)


def manual_grad(z, mu, log_sd):
  term1 = (z - mu)/torch.exp(2 * log_sd)
  return term1

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
  torch.manual_seed(0)
  d = DatasetMoons()
  writer = SummaryWriter()
  
  # Glow paper
  flows = [Invertible1x1Conv(dim=2) for i in range(config['n_flow'])]
  norms = [ActNorm(dim=2) for _ in flows]
  couplings = [AffineHalfFlow(dim=2, parity=i%2, nh=config['nh']) for i in range(len(flows))]
  flows = list(itertools.chain(*zip(norms, flows, couplings))) # append a coupling layer after each 1x1
  model = NormalizingFlowModel(flows, 2)

  # model = nn.DataParallel(model)
  model = model.to(device)
  # model.apply(ut.weights_init)

  # Define optimizer and loss
  optimizer = optim.Adam(model.parameters(), lr=lr)
  optimizer2 = optim.Adam(model.parameters(), lr=lr2)
  optimizer3 = optim.Adam(model.parameters(), lr=lr2)
  scheduler = MultiStepLR(optimizer, milestones=config['lr_milestone'], gamma=config['lr_gamma'])
  scheduler2 = MultiStepLR(optimizer2, milestones=config['lr2_milestone'], gamma=config['l2_gamma'])

  bhatta_loss = CustomLoss()
  best_loss = 1e5

  model.train()
  for k in range(epochs):
    x, target = d.sample(bsize)
    x = x.to(device)

    x = (x - x.min())/x.max()
    # Forward propogation
    zs, logdet, mean, log_sd= model(x)
    print(logdet.size(), mean.size(), log_sd.size())
    e()
    # NLL
    logprob, mus_per_class, log_sds_per_class, prior_prob = bhatta_loss(zs[-1], mean, log_sd, target, logdet, device)
    start_time = time.time()
    # BLL
    # bloss = bhatta_loss.b_loss(zs[-1], target, mus_per_class, log_sds_per_class, device, model)
    # bc_sim, bc_diff = bhatta_loss.b_loss(zs[-1], target, mus_per_class, log_sds_per_class, device, model)
    bloss = torch.tensor(1., requires_grad=True)
    
    loss1 = -torch.mean(logprob)
    loss2 = -bloss
    # loss3 = bc_diff
    # Gradient descent and optimization
    optimizer.zero_grad()
    # optimizer2.zero_grad()
    # optimizer3.zero_grad()

    # loss1.backward()
    loss1.backward(retain_graph=True)
    avg = ut.compute_avg_grad(model)
    loss2.backward(retain_graph=True)
    b_grad = ut.compute_avg_grad(model) - avg
    # loss3.backward(retain_graph=False)
    # b_grad = ut.compute_avg_grad(model) - avg

    optimizer.step()
    # optimizer2.step()
    # optimizer3.step()

    # Logging
    writer.add_scalar("NLLLoss/Train", loss1.item(), k)
    writer.add_scalar("BLoss/Train", loss2.item(), k)
    # writer.add_scalar("BLoss/Train", loss3.item(), k)
    writer.add_scalar("avg_grad/Train", avg, k)
    writer.add_scalar("b_grad/Train", b_grad, k)
    
    if loss1.item() < best_loss:
      torch.save(model.state_dict(), f"./models/best_model.pt")
      best_loss = loss1.item()

    # Generate plots
    if k % 100 == 0:
      z = zs[-1]
      print(f"NLL: {loss1.item()} BLL: {loss2.item()} lr: {optimizer.param_groups[0]['lr']}\
        NLL gradients: {round(avg, 4)} BLL gradients: {round(b_grad, 4)} epoch: {k}")
      ut.plot(z.detach(), target, f"{k}")
      # ut.plot_3d(z.detach(), prior_prob.detach(), k, target)
      x_rec, mean, log_sd = model.sample(bsize)
      ut.plot(x_rec.detach(), target, f"recon_{k}")
      # plot_grad_flow(model.named_parameters(), f"./plots/grad/{k}_high.png")
    scheduler.step()
    # scheduler2.step()

  zs, logdet, mean, log_sd= model(x)
  logprob, mus_per_class, log_sds_per_class, prior_prob = bhatta_loss(zs[-1], mean, log_sd, target, logdet, device)

  print(mus_per_class)
  print(log_sds_per_class)

  print(f"Best loss at {best_loss}")
  writer.flush()
  writer.close()
  