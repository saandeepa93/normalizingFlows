import torch 
from torch import nn
from torch.autograd import Variable

from math import pi, log
from sys import exit as e


def compute_avg_grad(model):
  avg_grad = 0
  cnt = 0
  for name, param in model.named_parameters():
    if param.grad is None:
      return None
    avg_grad += param.grad.abs().mean().item()
    cnt += 1
  avg_grad /= cnt
  return avg_grad

class CustomLoss(nn.Module):
  def __init__(self):
    super().__init__()
  
  def gaussian_log_p(self, x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * ((x - mean) ** 2 / torch.exp(2 * log_sd))  
  

  def b_loss(self, z, target, mus_per_class, log_sds_per_class, device, model):
    # Initial the (b*k) sizes to hold log_ps and targets
    
    # bhatta_loss_total = []
    bhatta_loss_log = []

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

      #Similarity feature coefficients
      sim_mask = (t.unsqueeze(1) @ t.unsqueeze(1).T).tril(-1)
      sim_cnt = (sim_mask == 1.).sum()
      # bc_sim = ((torch.exp(pwise) * sim_mask).sum())/sim_cnt

      bc_sim_test = pwise * sim_mask
      bc_sim_test = bc_sim_test[bc_sim_test != 0.]
      bc_sim_sum = torch.logsumexp(bc_sim_test, dim=0)
      bc_sim_new = bc_sim_sum - log(sim_cnt)

      #Dissimilar feature coefficients
      diff_mask = torch.zeros((z.size(0), z.size(0)))
      for k in range(z.size(0)):
        if t[k].item() == 1.:
          diff_mask[k] = t_1
        else:
          diff_mask[k] = t
      diff_mask = (diff_mask.tril(-1))
      diff_cnt = (diff_mask == 1.).sum()
      # bc_diff = ((torch.exp(pwise) * diff_mask).sum())/diff_cnt

      bc_diff_res = pwise * diff_mask
      bc_diff_res = bc_diff_res[bc_diff_res != 0.]
      bc_diff_sum = torch.logsumexp(bc_diff_res, dim=0)
      bc_diff_new = bc_diff_sum - log(diff_cnt)
      
      #Calculate final bhatta loss  
      # print(bc_sim, bc_diff)
      # bhatta_loss_total = bhatta_loss_total + (1. - bc_sim) + bc_diff
      # bhatta_loss_total.append((1. - bc_sim) + bc_diff)


      # print(bc_sim_new.exp().item(), bc_diff_new.exp().item())
      bhatta_loss_log.append(torch.logaddexp(-bc_sim_new, bc_diff_new))


    # print(compute_avg_grad(model))
    # print("="*30)

    # bhatta_loss_total = torch.stack(bhatta_loss_total).sum()
    # bhatta_loss_total = bhatta_loss_total/2

    bhatta_loss_log = torch.logsumexp(torch.stack(bhatta_loss_log), 0)
    bhatta_loss_log = bhatta_loss_log - log(2)

    # return bhatta_loss_total
    return bhatta_loss_log

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

    return prior_logprob, mus_per_class, log_sds_per_class, log_p_total

