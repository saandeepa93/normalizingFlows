import torch 
from torch import nn, optim
from torch.nn.init import xavier_normal_, zeros_
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification, load_iris
from sklearn import datasets
import itertools
import numpy as np
from math import log, pi
import pandas as pd
from sys import exit as e
import plotly.express as px

# from model import Glow
from model import Invertible1x1Conv, ActNorm, AffineHalfFlow, NormalizingFlowModel 
from loss import CustomLoss
import util as ut

nan_fn = lambda x: [torch.sum(torch.isnan(j)) for j in x]
max_fn = lambda x: torch.max(x)
min_fn = lambda x: torch.min(x)


def gaussian_log_p(x, mean, log_sd):
  return -0.5 * log(2 * pi) - log_sd - 0.5 * ((x - mean) ** 2 / torch.exp(2 * log_sd))


def weights_init(m):
  if isinstance(m, nn.Linear):
    xavier_normal_(m.weight.data)
    zeros_(m.bias.data)


def plot_3d(x, prob, k, target):
  target = [str(i) for i in target.detach().numpy()]
  df = pd.DataFrame(x, columns=["x0", "x1"])
  df_2 = pd.DataFrame(prob, columns=["pdf"])
  df_color = pd.DataFrame(target, columns=["color"])
  df = df.join(df_2)
  df = df.join(df_color)
  fig = px.scatter_3d(df, x='x0', y='x1', z='pdf', color='color', title=f"multivariate")
  fig.update_traces(marker=dict(size=6))
  fig.write_html(f"./plots/sample/html/{k}.html")

def plot_mult(x, prob, k):
  fig, ax = plt.subplots(1, figsize=(8, 4))
  ax = plt.axes(projection='3d')
  ax.scatter(x[:, 0], x[:, 1], prob)
  plt.show()

def plot_univariate(arr, ep, mu, sigma):
  y = gaussian_log_p(arr, mu, torch.log(sigma))

  x0 = arr[:, 0]
  y0 = y[:, 0]

  plt.scatter(x0, y0)
  plt.xlabel("x0")
  plt.ylabel("f(x0)")
  plt.savefig(f"./plots/sample/{ep}.png")
  plt.close()


def plot_grad_flow(named_parameters, name):
  ave_grads = []
  layers = []
  for n, p in named_parameters:
    if(p.requires_grad) and ("bias" not in n):
        layers.append(n.split(".")[0] + "_" + n.split(".")[1])
        ave_grads.append(p.grad.abs().mean())
  plt.plot(ave_grads, alpha=0.3, color="b")
  plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
  plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
  plt.xlim(xmin=0, xmax=len(ave_grads))
  plt.xlabel("Layers")
  plt.ylabel("average gradient")
  plt.title("Gradient flow")
  # plt.grid(True)
  plt.savefig(name)
  plt.close()

def plot(arr, labels, ep):
  if labels is None:
    plt.scatter(arr[:, 0], arr[:, 1])
  else:
    colors = ['r', 'g']
    for i in range(len(arr)):
      plt.scatter(arr[i, 0], arr[i, 1], color = colors[labels[i]])
  plt.savefig(f"./plots/sample/{ep}.png")
  plt.close()

def get_dataset(n_samples):
  moon, labels = make_moons(n_samples, noise=0.05)
  return moon.astype(np.float32), labels


def unbound(y):
  y = (2 * y - 1) * torch.tensor(0.9)
  y = (y + 1) / 2
  print(y)
  y = y.log() - (1. - y).log()
  return y


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




if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  epochs = 2001
  b_size = 256
  lr = 1e-3
  lr2 = 1e-3
  
  d = DatasetMoons()
  
  # x, y = d.sample_gauss(1000)
  # x, target = d.sample_iris()
  # x = x[:, :2].type(torch.float32)

  # Glow paper
  flows = [Invertible1x1Conv(dim=2) for i in range(3)]
  norms = [ActNorm(dim=2) for _ in flows]
  couplings = [AffineHalfFlow(dim=2, parity=i%2, nh=128) for i in range(len(flows))]
  flows = list(itertools.chain(*zip(norms, flows, couplings))) # append a coupling layer after each 1x1
  model = NormalizingFlowModel(flows, 2)
  # model = nn.DataParallel(model)
  # model = model.to(device)

  # Define optimizer and loss
  optimizer = optim.Adam(model.parameters(), lr=lr)
  # optim_zeroNN = optim.Adam(model.prior.parameters(), lr=lr2)

  bhatta_loss = CustomLoss()
  best_loss = 1e5

  model.train()
  for k in range(epochs):
    x, target = d.sample(256)

    x = x.to(device)

    # Forward propogation
    zs, logdet, mean, log_sd= model(x)

    # Likelihood maximization
    logprob, bloss = bhatta_loss(zs[-1], mean, log_sd, target, logdet, device)

    # loss2 = -torch.logaddexp(logprob, log_bloss)
    loss1 = -torch.mean(logprob)
    loss2 = bloss


    # Gradient descent and optimization
    optimizer.zero_grad()
    # loss1.backward()
    loss1.backward(retain_graph=True)
    loss2.backward()
    optimizer.step()
    # optim_zeroNN.step()

    if loss1.item() < best_loss:
      torch.save(model.state_dict(), f"./models/best_model.pt")
      best_loss = loss1.item()


    # Generate plots
    if k % 100 == 0:
      # for name, param in model.named_parameters():
      #   print(name, param.grad.abs().mean())
      z = zs[-1]
      print(loss1.item(), loss2.item())
      plot(z.detach(), target, f"{k}")
      # plot_3d(z.detach(), prior_logprob.detach(), k, target)
      x_rec, mean, log_sd = model.sample(256)
      plot(x_rec.detach(), None, f"recon_{k}")
      # plot_grad_flow(model.named_parameters(), f"./plots/grad/{k}_high.png")

    
    
  print(f"Best loss at {best_loss}")
  