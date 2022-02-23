import torch 
from torch import nn
from torch.autograd import Variable

from math import pi, log
from sys import exit as e

import util as ut


class CustomLoss(nn.Module):
  def __init__(self):
    super().__init__()
  
  def gaussian_log_p(self, x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * ((x - mean) ** 2 / torch.exp(2 * log_sd))  
  

  def b_loss(self, z, target, mus_per_class, log_sds_per_class, device, model):
    # Initial the (b*k) sizes to hold log_ps and targets
    
    bhatta_loss_total = []
    # bhatta_loss_log = []
    bc_sim_total = []
    bc_diff_total = []

    log_p_lst = torch.zeros((z.size(0), 2), device = device)
    targets = torch.zeros((z.size(0), 2), device = device)
    for j in range(2):
      targets[target==j, j] = 1
      log_p_lst[:, j] = self.gaussian_log_p(z, mus_per_class[j], log_sds_per_class[j]).sum(1)

    for j in range(2):
      p = log_p_lst[:, j]
      t = targets[:, j]
      t_1 = 1. - t
      pwise = (0.5 * (p.unsqueeze(1) + p))
      # pwise = torch.sqrt(p.exp().unsqueeze(1) @ p.exp().unsqueeze(0))

      #Similarity feature coefficients i am poopy
      sim_mask = (t.unsqueeze(1) @ t.unsqueeze(1).T).tril(-1)
      sim_cnt = (sim_mask == 1.).sum()
      bc_sim = ((pwise * sim_mask).sum())/sim_cnt
      # bc_sim = ((torch.exp(pwise) * sim_mask).sum())/sim_cnt


      #Dissimilar feature coefficients
      diff_mask = torch.zeros((z.size(0), z.size(0)))
      for k in range(z.size(0)):
        if t[k].item() == 1.:
          diff_mask[k] = t_1
        else:
          diff_mask[k] = t
      diff_mask = (diff_mask.tril(-1))
      diff_cnt = (diff_mask == 1.).sum()
      bc_diff = ((pwise * diff_mask).sum())/diff_cnt
      # bc_diff = ((torch.exp(pwise) * diff_mask).sum())/diff_cnt

      bc_sim_total.append(bc_sim)
      bc_diff_total.append(bc_diff)

    bc_sim_total = torch.stack(bc_sim_total, dim=0).mean()/2
    bc_diff_total = torch.stack(bc_diff_total, dim=0).mean()/2
    return bc_sim_total, bc_diff_total

  def forward(self, z, mean, log_sd, target, logdet, device):
    # NLL
    log_p_total = []
    logdet_total = []
    prior_prob = []
    # NLL

    # contrastive b-loss
    cls_len = []
    mus_per_class, log_sds_per_class = [], []
    mus_per_class_lst, log_sds_per_class_lst = [], []
    # contrastive b-loss

    for cls in [0, 1]:
      ind = ((target == cls).nonzero(as_tuple=True)[0])
      logdet_total.append(logdet[ind].mean())
      z_cls = z[ind]
      mu_cls = mean[ind].mean(0)
      log_sd_cls = log_sd[ind].mean(0)

      mus_per_class.append(mu_cls)
      log_sds_per_class.append(log_sd_cls)

      prior_prob.append(self.gaussian_log_p(z_cls, mu_cls, log_sd_cls).view(z_cls.size(0), -1).sum(1))
      log_p_total.append(self.gaussian_log_p(z_cls, mu_cls, log_sd_cls).view(z_cls.size(0), -1).sum(1).mean())
    
    log_p_total = torch.stack(log_p_total, dim = 0)
    logdet_total = torch.stack(logdet_total, dim = 0)
    prior_logprob = (log_p_total + logdet_total).mean()

    return prior_logprob, mus_per_class, log_sds_per_class, torch.cat(prior_prob)

