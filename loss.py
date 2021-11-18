import torch 
from torch import nn
from torch.autograd import Variable

from math import pi, log
from sys import exit as e


class CustomLoss(nn.Module):
  def __init__(self):
    super().__init__()
    # self.labels = [0, 1]
  
  def gaussian_log_p(self, x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * ((x - mean) ** 2 / torch.exp(2 * log_sd))  
  
  def forward(self, z, mean, log_sd, target, logdet, device):
    # NLL
    log_p_total = []
    logdet_total = []
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

      log_p_total.append(self.gaussian_log_p(z_cls, mu_cls, log_sd_cls).view(z_cls.size(0), -1).sum(1).mean())
    
    log_p_total = torch.stack(log_p_total, dim = 0)
    logdet_total = torch.stack(logdet_total, dim = 0)
    prior_logprob = (log_p_total + logdet_total).mean()


    # Initial the (b*k) sizes to hold log_ps and targets
    log_p_lst = torch.zeros((256, 2), device = device)
    # log_p_lst = []
    targets = torch.zeros((256, 2), device = device)
    for j in range(2):
      targets[torch.where(target==j)[0], j] = 1
      for i in range(256):
        log_p_lst[i][j] = self.gaussian_log_p(z[i], mus_per_class[j], log_sds_per_class[j]).sum()
        # log_p_lst.append(self.gaussian_log_p(z[i], mus_per_class[j], log_sds_per_class[j]).sum())
    
    # log_p_lst = torch.tensor(log_p_lst).view(2, 256).T
    # log_p_lst.requires_grad = True

    bhatta_loss = 0
    # bhatta_loss = torch.tensor(2., requires_grad=True)
    for j in range(2):
      p = log_p_lst[:, j]
      t = targets[:, j]
      t_1 = 1. - t
      pwise = (0.5 * (p.unsqueeze(1) + p)).squeeze()

      #Similarity feature coefficients
      sim_mask = (t.unsqueeze(1) @ t.unsqueeze(1).T).tril(-1)
      sim_cnt = (sim_mask == 1.).sum()
      bc_sim = ((torch.exp(pwise) * sim_mask).sum())/sim_cnt

      #Dissimilar feature coefficients
      diff_mask = torch.zeros((256, 256))
      for k in range(256):
        if t[k].item() == 1.:
          diff_mask[k] = t_1
        else:
          diff_mask[k] = t
      diff_mask = (diff_mask.tril(-1))
      diff_cnt = (diff_mask == 1.).sum()
      bc_diff = ((torch.exp(pwise) * diff_mask).sum())/diff_cnt

      #Calculate final bhatta loss  
      bhatta_loss = bhatta_loss + (1. - bc_sim) + bc_diff
    bhatta_loss /= 2

    return prior_logprob, bhatta_loss

