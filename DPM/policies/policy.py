import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import math

from policies.utils import *
from policies.networks import *

import numpy as np

class CNN_Policy(nn.Module):
    def __init__(self, num_inputs, num_action, args, device=None):
        super(CNN_Policy, self).__init__()
        if 'sarl' in args.model_name:
            n = 2
            self.sarl_net = SARL_net(num_inputs, num_action).to(device)
        else:
            n = 1
        self.net = CNN(num_inputs, num_action, n).to(device)


    def train_net(self,x, y, last_w, setw, y_cont, args, device):
        x, y, last_w, y_cont = get_tensor(x, y, last_w, y_cont, device)
        self.net.optimizer.zero_grad()
        if 'sarl' in args.model_name :
            pred, _ = self.sarl_net(x)
            prob, _ = self.net(x, last_w,  pred.detach().argmax(dim=1))
        else:
            prob, _ = self.net(x, last_w)
        pv_vector, baseline, _ = cal_pv(y, prob)
        loss = -torch.mean(torch.log(pv_vector))
        loss.backward()
        self.net.optimizer.step()
        setw(prob[:, 1:].detach().cpu().numpy())
        portfolio_value = torch.prod(pv_vector)
        return portfolio_value.detach().cpu().numpy(), loss.detach().cpu().numpy(), _, _, _, _, _


class Conv4_Policy(nn.Module):
    def __init__(self, num_inputs, action_n, lr=0.001, n_episode_batch=5, nri_d=32, cnn_d=10, shuffle=1, shift=True, nri_lr=0.0005, cnn_d2=15, args=None, device=None):
        super(Conv4_Policy, self).__init__()
        if 'sarl' in args.model_name:
            n = 2
            self.sarl_net = SARL_net_conv4(num_inputs, action_n).to(device)
        else:
            n = 1

        if args.stocks == 0:
            self.net=CNN_conv4_stock(num_inputs, action_n, lr, n_episode_batch, nri_d, cnn_d, shuffle, False, nri_lr, cnn_d2, n).to(device)
        else:
            if args.model_name == 'ours':
                self.net=CNN_conv4(num_inputs, action_n, lr, n_episode_batch, nri_d, cnn_d, shuffle, True, nri_lr, cnn_d2, n).to(device)
            else:
                self.net=CNN_conv4(num_inputs, action_n, lr, n_episode_batch, nri_d, cnn_d, shuffle, False, nri_lr, cnn_d2, n).to(device)
        self.nri_net = NRI_net(num_inputs, action_n, lr, n_episode_batch, nri_d, cnn_d, shuffle, shift, nri_lr, cnn_d2).to(device)
        

    def train_net(self, x, y, last_w, setw, y_cont, device, args=None):
        x, y, last_w, y_cont = get_tensor(x, y, last_w, y_cont, device)
        L2= torch.tensor(0)
        nll= torch.tensor(0)
        kl= torch.tensor(0)

        self.net.optimizer.zero_grad()
        if 'sarl' in args.model_name :
            pred, _ = self.sarl_net(x)
            prob, _ = self.net(x, last_w,  pred.detach().argmax(dim=1))
        else:
            prob, re = self.net(x, last_w)
        

        pv_vector, baseline, _ = cal_pv(y, prob)
        c_profit = torch.bmm(prob[:,1:].unsqueeze(1), 0.2*(y_cont[:,0,:,:]-1)).squeeze(1).sum(axis=1)
        
        if args.L2_w > 0:
            L2, nll, kl = self.nri_net.nriLoss(re, x)

        if args.L1_baseline:
            L1 = -torch.mean(torch.log(pv_vector) - torch.log(baseline))
        else:
            L1 = -torch.mean(torch.log(pv_vector))

        L3 =  - (c_profit).mean()

        if args.L1_w:
            L1_w = args.L1_w
        else:
            L1_w = 2

        if args.L3_w_const:
            L3_w = args.L3_w
        else:
            L3_w = args.L3_w*torch.exp(L1.detach())
        loss = L1_w * L1 + L3_w * L3 + args.L2_w * L2

        loss.backward()
        self.net.optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
        setw(prob[:, 1:].detach().cpu().numpy())

        portfolio_value = torch.prod(pv_vector)
        return portfolio_value.detach().cpu().numpy(), L1.detach().cpu().numpy(), L2.detach().cpu().numpy(), L3.detach().cpu().numpy(), nll, kl, L3_w.detach().cpu().numpy()

