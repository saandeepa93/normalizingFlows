import torch 
from torch.nn.init import xavier_normal_, zeros_
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from math import pi, log


def plot_3d(x, prob, k):
  df = pd.DataFrame(x, columns=["x0", "x1"])
  df_2 = pd.DataFrame(prob, columns=["pdf"])
  df = df.join(df_2)
  fig = px.scatter_3d(df, x='x0', y='x1', z='pdf', title=f"multivariate")
  fig.update_traces(marker=dict(size=6))
  fig.write_html(f"./plots/sample/html/{k}.html")



def gaussian_log_p(x, mean, log_sd):
  return -0.5 * log(2 * pi) - log_sd - 0.5 * ((x - mean) ** 2 / torch.exp(2 * log_sd))


def weights_init(m):
  if isinstance(m, nn.Linear):
    xavier_normal_(m.weight.data)
    zeros_(m.bias.data)


def plot_3d(x, prob, k):
  df = pd.DataFrame(x, columns=["x0", "x1"])
  df_2 = pd.DataFrame(prob, columns=["pdf"])
  df = df.join(df_2)
  fig = px.scatter_3d(df, x='x0', y='x1', z='pdf', title=f"multivariate")
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
    if(p.requires_grad) and ("bias" not in n) and (("prior" in n) or ("affine" in n)):
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