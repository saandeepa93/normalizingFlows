import numpy as np
from math import log, pi
from sys import exit as e

import torch
import torch.nn.functional as F
from torch import nn



nan_fn = lambda x: [torch.sum(torch.isnan(j)) for j in x]
max_fn = lambda x: torch.max(x)
min_fn = lambda x: torch.min(x)

def gaussian_log_p(x, mean, log_sd):
  return -0.5 * log(2 * pi) - log_sd - 0.5 * ((x - mean) ** 2 / torch.exp(log_sd))


def gaussian_sample(eps, mean, log_sd):
  return mean + torch.exp(log_sd) * eps


logabs = lambda x: torch.log(torch.abs(x))

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

  def backward(self, z):
    W = self._assemble_W()
    W_inv = torch.inverse(W)
    x = z @ W_inv
    log_det = -torch.sum(torch.log(torch.abs(self.S)))
    return x, log_det


class AffineConstantFlow(nn.Module):
    """ 
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """
    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None
        
    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class ActNorm(AffineConstantFlow):
  """
  Really an AffineConstantFlow but with a data-dependent initialization,
  where on the very first batch we clever initialize the s,t so that the output
  is unit gaussian. As described in Glow paper.
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.data_dep_init_done = False
  
  def forward(self, x):
    # first batch is used for init
    if not self.data_dep_init_done:
      assert self.s is not None and self.t is not None # for now
      self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
      self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
      self.data_dep_init_done = True
    return super().forward(x)


class ZeroNN(nn.Module):
  def __init__(self, nin, nout):
    super().__init__()

    self.linear = nn.Linear(nin, nout)

    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

  def forward(self, input):
    out = self.linear(input)
    return out


class MLP(nn.Module):
  """ a simple 4-layer MLP """

  def __init__(self, nin, nout, nh):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(nin, nh),
      nn.LeakyReLU(0.2),
      nn.Linear(nh, nh),
      nn.LeakyReLU(0.2),
      nn.Linear(nh, nh),
      nn.LeakyReLU(0.2),
      ZeroNN(nh, nout)
      # nn.Linear(nh, nout),
    )
  def forward(self, x):
    return self.net(x)

class AffineHalfFlow(nn.Module):
  """
  As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the 
  dimensions in x are linearly scaled/transfromed as a function of the other half.
  Which half is which is determined by the parity bit.
  - RealNVP both scales and shifts (default)
  - NICE only shifts
  """
  def __init__(self, dim, parity, net_class=MLP, nh=24, scale=True, shift=True):
    super().__init__()
    self.dim = dim
    self.parity = parity
    self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
    self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
    if scale:
      self.s_cond = net_class(self.dim // 2, self.dim // 2, nh)
    if shift:
      self.t_cond = net_class(self.dim // 2, self.dim // 2, nh)
      
  def forward(self, x):
    x0, x1 = x[:,::2], x[:,1::2]
    if self.parity:
      x0, x1 = x1, x0
    s = self.s_cond(x0)
    t = self.t_cond(x0)
    z0 = x0 # untouched half
    z1 = torch.exp(s) * x1 + t # transform this half as a function of the other
    if self.parity:
      z0, z1 = z1, z0
    z = torch.cat([z0, z1], dim=1)
    log_det = torch.sum(s, dim=1)
    return z, log_det
  
  def backward(self, z):
    z0, z1 = z[:,::2], z[:,1::2]
    if self.parity:
      z0, z1 = z1, z0
    s = self.s_cond(z0)
    t = self.t_cond(z0)
    x0 = z0 # this was the same
    x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
    if self.parity:
      x0, x1 = x1, x0
    x = torch.cat([x0, x1], dim=1)
    log_det = torch.sum(-s, dim=1)
    return x, log_det

class NormalizingFlow(nn.Module):
  """ A sequence of Normalizing Flows is a Normalizing Flow """

  def __init__(self, flows):
    super().__init__()
    self.flows = nn.ModuleList(flows)

  def forward(self, x):
    m, _ = x.shape
    log_det = torch.zeros(m)
    zs = [x]
    for flow in self.flows:
      x, ld = flow.forward(x)
      log_det += ld
      zs.append(x)
    
    return zs, log_det

  def backward(self, z):
    m, _ = z.shape
    log_det = torch.zeros(m)
    xs = [z]
    for flow in self.flows[::-1]:
      z, ld = flow.backward(z)
      log_det += ld
      xs.append(z)
    return xs, log_det

class NormalizingFlowModel(nn.Module):
  """ A Normalizing Flow Model is a (prior, flow) pair """
  
  def __init__(self, flows, nin, prior=None):
    super().__init__()
    # self.prior = prior
    self.prior = ZeroNN(nin, nin*2)
    self.flow = NormalizingFlow(flows)
  
  def forward(self, x):
    zs, log_det = self.flow.forward(x)
    mean, log_sd = self.prior(zs[-1]).chunk(2, 1)
    return zs, log_det, mean, log_sd

  def backward(self, z):
    xs, log_det = self.flow.backward(z)
    return xs, log_det

  def sample(self, num_samples):
    z_rec = torch.FloatTensor(num_samples, 2).normal_(0, 1)
    mean, log_sd = self.prior(z_rec).chunk(2, 1)
    log_sd = log_sd.mean(0)
    mean = mean.mean(0)
    z = gaussian_sample(z_rec, mean, log_sd)
    # z = self.prior.sample((num_samples,))
    xs, _ = self.flow.backward(z)
    return xs[-1], mean, log_sd
