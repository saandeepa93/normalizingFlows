import torch 
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from torch import nn, optim

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import itertools
import numpy as np
from sys import exit as e

from model import Glow


from torchvision import io



def plot_grad_flow(named_parameters, name):
  ave_grads = []
  layers = []
  for n, p in named_parameters:
    if(p.requires_grad) and ("bias" not in n):
        layers.append(n)
        ave_grads.append(p.grad.abs().mean())
  plt.plot(ave_grads, alpha=0.3, color="b")
  plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
  plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
  plt.xlim(xmin=0, xmax=len(ave_grads))
  plt.xlabel("Layers")
  plt.ylabel("average gradient")
  plt.title("Gradient flow")
  plt.grid(True)
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
  # print(moon.shape, labels.shape)
  # plot(moon, labels)


def test():
  model_path = "./models/best_model.pt"
  prior = TransformedDistribution(Uniform(torch.zeros(2), torch.ones(2)), SigmoidTransform().inv)

  # flows = [Invertible1x1Conv(dim=2) for i in range(3)]
  # norms = [ActNorm(dim=2) for _ in flows]
  # couplings = [AffineHalfFlow(dim=2, parity=i%2, nh=32) for i in range(len(flows))]
  # flows = list(itertools.chain(*zip(norms, flows, couplings)))
  # model = NormalizingFlowModel(prior, flows)

  model.load_state_dict(torch.load(model_path))
  model.eval()
  with torch.no_grad():
    x, _ = get_dataset(128)
    zs = model.sample(128*8)
    z = zs[-1]
    z = z.detach().numpy()
    # plt.scatter(x[:,0], x[:,1], c='b', s=5, alpha=0.5)
  plt.scatter(z[:,0], z[:,1], c='r', s=5, alpha=0.5)
  plt.legend(['data', 'z->x'])
  plt.title('z -> x')
  plt.show()

def gaussian_sample(eps, mean, log_sd):
  return mean + torch.exp(log_sd) * eps


if __name__ == "__main__":
  epochs = 1000
  model = Glow(2, 8)
  print("number of params: ", sum(p.numel() for p in model.parameters()))

  optimizer =  optim.Adam(model.parameters(), lr=1e-3)
  best_loss = 1e5

  z_rec = torch.FloatTensor(256, 2).normal_(0, 1)


  model.train()
  for k in range(epochs):
    x, target = get_dataset(256)
    x = torch.from_numpy(x)
    target = torch.from_numpy(target)

    z, logdet, prior_logprob = model(x)

    logdet = logdet.mean()
    logprob = prior_logprob + logdet
    loss = -torch.mean(logprob) # NLL

    model.zero_grad()
    loss.backward()
    optimizer.step()

    # nn.utils.clip_grad_norm(model.parameters(), 0.5)

    if loss.item() < best_loss:
      torch.save(model.state_dict(), f"./models/best_model.pt")
      best_loss = loss.item()

    if k % 100 == 0:
      plot_grad_flow(model.named_parameters(), f"./plots/grad/{k}_high.png")
      plot(z.detach(), target.detach(), k)
      plot(model.reverse(z_rec).detach(), None, f"recon_{k}.png")
      print(loss.item())
  
  print(f"Best loss at {best_loss}")
  