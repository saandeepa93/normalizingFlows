import numpy as np
from math import log, pi

import torch
import torch.nn.functional as F
from torch import nn

from sys import exit as e



def gaussian_log_p(x, mean, log_sd):
  return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
  return mean + torch.exp(log_sd) * eps


logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
  def __init__(self, in_channel, logdet=True):
    super().__init__()

    self.loc = nn.Parameter(torch.zeros(1, in_channel))
    self.scale = nn.Parameter(torch.ones(1, in_channel))

    self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
    self.logdet = logdet

  def initialize(self, input):
    with torch.no_grad():
        flatten = input.permute(1, 0).contiguous()
        mean = (
            flatten.mean(1)
            .unsqueeze(1)
            .permute(1, 0)
        )
        std = (
            flatten.std(1)
            .unsqueeze(1)
            .permute(1, 0)
        )
        self.loc.data.copy_(-mean)
        self.scale.data.copy_(1 / (std + 1e-6))

  def forward(self, input):
    # _, _, height, width = input.shape

    if self.initialized.item() == 0:
        self.initialize(input)
        self.initialized.fill_(1)

    log_abs = logabs(self.scale)

    logdet = torch.sum(log_abs)

    return self.scale * (input + self.loc), logdet

  def reverse(self, output):
    return output / self.scale - self.loc


class Invertible1x1Conv(nn.Module):
  """ 
  As introduced in Glow paper.
  """
  
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
    P, L, U = torch.lu_unpack(*Q.lu())
    self.P = P # remains fixed during optimization
    self.L = nn.Parameter(L) # lower triangular portion
    self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
    self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

  def _assemble_W(self):
    """ assemble W from its pieces (P, L, U, S) """
    L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim))
    U = torch.triu(self.U, diagonal=1)
    W = self.P @ L @ (U + torch.diag(self.S))
    return W

  def forward(self, x):
    W = self._assemble_W()
    z = x @ W
    log_det = torch.sum(torch.log(torch.abs(self.S)))
    return z, log_det

  def reverse(self, z):
    W = self._assemble_W()
    W_inv = torch.inverse(W)
    x = z @ W_inv
    log_det = -torch.sum(torch.log(torch.abs(self.S)))
    return x, log_det


class ZeroNN(nn.Module):
  def __init__(self, in_chan, out_chan):
    super().__init__()

    self.linear = nn.Linear(in_chan, out_chan)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    self.scale = nn.Parameter(torch.zeros(1, out_chan))
  
  def forward(self, x):
    out = self.linear(x)
    out = out * torch.exp(self.scale * 3)
    return out

class AffineCoupling(nn.Module):
  def __init__(self, in_channel, parity, filter_size=32):
    super().__init__()

    self.parity = parity
    self.net = nn.Sequential(
      nn.Linear(in_channel//2, filter_size),
      nn.LeakyReLU(),
      nn.Linear(filter_size, filter_size),
      nn.LeakyReLU(),
      ZeroNN(filter_size, in_channel)
    )

    self.net[0].weight.data.normal_(0, 0.05)
    self.net[0].bias.data.zero_()

    self.net[2].weight.data.normal_(0, 0.05)
    self.net[2].bias.data.zero_()

  
  def forward(self, input):
    in_a, in_b = input.chunk(2, 1)
    if self.parity:
      in_a, in_b = in_b, in_a
    log_s, t = self.net(in_a).chunk(2, 1)
    s = torch.sigmoid(log_s + 2)
    out_b = (in_b + t) * s
    logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
    if self.parity:
      in_a, out_b = out_b, in_a
    return torch.cat([in_a, out_b], 1), logdet
  
  def reverse(self, output):
    out_a, out_b = output.chunk(2, 1)
    if self.parity:
      out_a, out_b = out_b, out_a
    log_s, t = self.net(out_a).chunk(2, 1)
    s = torch.sigmoid(log_s + 2)
    in_b = out_b / s - t
    if self.parity:
      out_a, in_b = in_b, out_a
    return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
  def __init__(self, in_channel, parity):
    super().__init__()

    self.actnorm = ActNorm(in_channel)
    self.inconvlu = Invertible1x1Conv(in_channel)
    self.affine = AffineCoupling(in_channel, parity)
  
  def forward(self, input):
    out, logdet = self.actnorm(input)
    out, det1 = self.inconvlu(out)
    out, det2 = self.affine(out)

    logdet = logdet + det1 + det2

    return out, logdet

  def reverse(self, output):
    input = self.affine.reverse(output)
    input, _ = self.inconvlu.reverse(input)
    input = self.actnorm.reverse(input)
    return input


class Glow(nn.Module):
  def __init__(self, in_channel, n_flows):
    super().__init__()

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      parity = int(i%2)
      self.flows.append(Flow(in_channel, parity))
    self.prior = ZeroNN(in_channel, in_channel*2)

  def forward(self, input):
    b_size = input.size(0)
    out = input 
    logdet = 0
  
    for flow in self.flows:
      out, det = flow(out)
      logdet += det
  
  
    zero = torch.zeros_like(out)
    mean, log_sd = self.prior(zero).chunk(2, 1)
    log_p = gaussian_log_p(out, mean, log_sd)
    log_p = log_p.view(b_size, -1).sum(1)

    return out, logdet, log_p


  def reverse(self, output, eps=None):
    # input = eps
    zero = torch.zeros_like(output)
    mean, log_sd = self.prior(zero).chunk(2, 1)
    z = gaussian_sample(output, mean, log_sd)
    input = z

    for flow in self.flows[::-1]:
        input = flow.reverse(input)
    return input
