import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import math

from policies.NRI.utils import *
from policies.NRI.modules import *

from policies.utils import *

import numpy as np

class CNN(nn.Module):
    def __init__(self, num_inputs, num_action, n):
        super(CNN, self).__init__()

        self.window_size = 31
        self.num_inputs = num_inputs
        self.num_action = num_action
        D1 = 3
        D2 = 10
        self.conv1 = nn.Conv2d(self.num_inputs, D1, kernel_size=(1, 2))
        self.conv2 = nn.Conv2d(D1, D2, kernel_size=(1, self.window_size-1))


        self.conv3 = nn.Conv2d(D2+n, 1, kernel_size=(1, 1))
        self.bias = nn.Parameter(torch.zeros(1))
        
        self.optimizer = optim.AdamW([
            dict(params=[*self.conv1.parameters(),
                         self.bias]),
            dict(params=self.conv2.parameters(),
                 weight_decay=5e-9), # L2 reg
           dict(params=self.conv3.parameters(),
                 weight_decay=5e-8), # L2 reg
        ], lr=0.00028)


    def forward(self, x, last_action, pred=None):
        B = x.shape[0]
        x = x / x[:, 0, None, :, -1, None] 
        x = self.conv1(x)  
        x = torch.relu(x)
        x = self.conv2(x)  
        x = torch.relu(x)
        re = x
        prev_w = last_action.view(B, 1, self.num_action-1, 1) 
        if pred is not None:
            pred = pred.view(B, 1, self.num_action-1, 1)
            x = torch.cat([x, prev_w, pred], 1) 
        else:
            x = torch.cat([x, prev_w], 1) 
        x = self.conv3(x) 
        x = torch.cat([
            self.bias.repeat(B, 1),  # (B, 1)
            x[:, 0, :, 0]  # (B, C)
        ], 1)  # (B, 1+C)
        x = torch.softmax(x, -1)  # (B, 1+C)
        return x.squeeze(0), re

class SARL_net(nn.Module):
    def __init__(self, num_inputs, num_action):
        super(SARL_net, self).__init__()
        self.num_inputs = num_inputs
        self.window_size = 31
        self.num_action = num_action
        
        D1 = 3
        D2 = 10

        self.conv1 = nn.Conv2d(self.num_inputs, D1, kernel_size=(1, 2))
        self.conv2 = nn.Conv2d(D1, D2, kernel_size=(1, self.window_size-1))


        self.conv3 = nn.Conv2d(D2, 2, kernel_size=(1, 1))

        self.optimizer = optim.AdamW([
            dict(params=self.conv1.parameters(),
                                    ),
            dict(params=self.conv2.parameters(),
                 weight_decay=5e-9), # L2 reg
           dict(params=self.conv3.parameters(),
                 weight_decay=5e-8), # L2 reg
                ], lr=0.00028)

    def forward(self, x):
        B = x.shape[0]

        x = x / x[:, 0, None, :, -1, None] 

        x = self.conv1(x)  
        x = torch.relu(x)

        x = self.conv2(x) 

        x = torch.relu(x)
        re = x
        x = self.conv3(x)  
        x = x[:,:,:,0]
        x = torch.softmax(x, 1)  
        return x, re

    def sarl_train_net(self, x, y, last_w, y_cont, device):
        x, y, last_w, y_cont = get_tensor(x, y, last_w, y_cont, device)
        pred , _ = self(x)
        targets = (y<=1).float()
        
        self.optimizer.zero_grad()
        loss=0
        loss_fn=nn.BCELoss()
        for i in range(pred.shape[2]):
            loss += loss_fn(pred[:,0,i],targets[:,0,i])
        loss.backward()
        self.optimizer.step()
        return loss.item()

class SARL_net_conv4(nn.Module):
    def __init__(self, num_inputs, num_action):
        super(SARL_net_conv4, self).__init__()
        self.num_inputs = num_inputs
        self.window_size = 31
        self.num_action = num_action
        self.hidden_dim = 64

        self.hidden_dim1 = 32
        D1 = 50
        D2 = 25

        self.conv1 = nn.Conv2d(self.num_inputs, D1, kernel_size=(1, 2))
        self.conv2 = nn.Conv2d(D1, D2, kernel_size=(1, self.window_size-1))
        self.conv2_5 = nn.Conv2d(1, self.num_action-1 , kernel_size=(self.num_action-1, 1))
        self.conv2_5_ = nn.Conv2d(1, self.num_action-1 , kernel_size=(self.num_action-1, 1))
        self.conv3 = nn.Conv2d(D2, 15, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(15, 2, kernel_size=(1, 1))

        self.bias = nn.Parameter(torch.zeros(1))
        self.leakyrelu = nn.LeakyReLU()
        self.optimizer = optim.AdamW([
                dict(params=[*self.conv1.parameters(),
                                self.bias,
                            ]),
                dict(params=self.conv2.parameters(),
                    weight_decay=5e-9), # L2 reg
                dict(params=self.conv3.parameters(),
                    weight_decay=5e-8), # L2 reg
                dict(params=self.conv2_5.parameters(),
                    weight_decay=5e-9),
                dict(params=self.conv4.parameters(),
                    weight_decay=5e-8),
            ], lr=0.00015)


    def forward(self, x):
        B = x.shape[0]
        x = x / x[:, 0, None, :, -1, None] # normalize???

        x = self.conv1(x)  # (B, 3, C, W-1)
        x = torch.relu(x)

        x = self.conv2(x)  # (B, 10, C, 1)
        x = torch.relu(x)
        re = x
        x = self.conv3(x)  # (B, 1, C, 1)
        x = self.conv4(x)
        x = x[:,:,:,0]
        x = torch.softmax(x, 1)  # (B, 1+C)

        return x, re

    def sarl_train_net(self,x, y, last_w,  y_cont, device):
        x, y, last_w, y_cont = get_tensor(x, y, last_w, y_cont, device)
        pred , _ = self(x)
        targets = (y<=1).float()
        

        self.optimizer.zero_grad()
        loss=0
        loss_fn=nn.BCELoss()
        for i in range(pred.shape[2]):
            loss += loss_fn(pred[:,0,i],targets[:,0,i])
        loss.backward()
        self.optimizer.step()
        return loss.item()

class SARL_net_conv4_stock(nn.Module):
    def __init__(self, num_inputs, num_action):
        super(SARL_net_conv4_stock, self).__init__()
        self.num_inputs = num_inputs
        self.window_size = 31
        self.num_action = num_action
        self.hidden_dim = 64

        self.hidden_dim1 = 32
        D1 = 50
        D2 = 25

        self.conv1 = nn.Conv2d(self.num_inputs, D1, kernel_size=(1, 2))
        self.conv2 = nn.Conv2d(D1, D2, kernel_size=(1, self.window_size-1))
        self.conv2_5 = nn.Conv2d(1, self.num_action-1 , kernel_size=(self.num_action-1, 1))
        self.conv2_5_ = nn.Conv2d(1, self.num_action-1 , kernel_size=(self.num_action-1, 1))
        self.conv3 = nn.Conv2d(D2, 15, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(15, 2, kernel_size=(1, 1))

        self.bias = nn.Parameter(torch.zeros(1))
        self.leakyrelu = nn.LeakyReLU()
        self.optimizer = optim.AdamW([
                dict(params=[*self.conv1.parameters(),
                                self.bias,
                            ]),
                dict(params=self.conv2.parameters(),
                    weight_decay=5e-9), # L2 reg
                dict(params=self.conv3.parameters(),
                    weight_decay=5e-8), # L2 reg
                dict(params=self.conv2_5.parameters(),
                    weight_decay=5e-9),
                dict(params=self.conv4.parameters(),
                    weight_decay=5e-8),
            ], lr=0.00015)


    def forward(self, x):
        B = x.shape[0]
        x = x / x[:, 0, None, :, -1, None] # normalize???

        x = self.conv1(x)  # (B, 3, C, W-1)
        x = self.leakyrelu(x)

        x = self.conv2(x)  # (B, 10, C, 1)
        

        x = x.transpose(1,3)
        x = self.conv2_5(x)
        x = x.permute(0,3,1,2)

        re = x
        x = self.leakyrelu(x)

        x = self.conv3(x)  # (B, 1, C, 1)
        x = self.conv4(x)
        x = x[:,:,:,0]
        x = torch.softmax(x, 1)  # (B, 1+C)
        return x, re

    def sarl_train_net(self,x, y, last_w,  y_cont, device):
        x, y, last_w, y_cont = get_tensor(x, y, last_w, y_cont, device)
        pred , _ = self(x)
        targets = (y<=1).float()
        
        self.optimizer.zero_grad()
        loss=0
        loss_fn=nn.BCELoss()
        for i in range(pred.shape[2]):
            loss += loss_fn(pred[:,0,i],targets[:,0,i])
        loss.backward()
        self.optimizer.step()
        return loss.item()

class CNN_conv4(nn.Module):
    def __init__(self, num_inputs, action_n, lr=0.001, n_episode_batch=5, nri_d=32, cnn_d=10, shuffle=1, shift=False, nri_lr=0.0005, cnn_d2=15, n=1):
        super(CNN_conv4, self).__init__()
        self.num_inputs = num_inputs
        self.window_size = 31
        self.num_action = action_n
        self.shuffle = shuffle
        self.pre = shift

        D1 = cnn_d
        D2 = cnn_d2 #15

        self.conv1 = nn.Conv2d(self.num_inputs, D1, kernel_size=(1, 2))
        self.conv2 = nn.Conv2d(D1, D2, kernel_size=(1, self.window_size-1))
        self.conv2_5 = nn.Conv2d(1, self.num_action-1 , kernel_size=(self.num_action-1, 1))

        self.conv3 = nn.Conv2d(D2+n, 15, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(15, 1, kernel_size=(1, 1))

        self.bias = nn.Parameter(torch.zeros(1))
        self.leakyrelu = nn.LeakyReLU()

        self.optimizer = optim.AdamW([
            dict(params=[*self.conv1.parameters(),
                            self.bias,
                         ]),
            dict(params=self.conv2.parameters(),
                 weight_decay=5e-9), # L2 reg
            dict(params=self.conv3.parameters(),
                 weight_decay=5e-8), # L2 reg
            dict(params=self.conv2_5.parameters(),
                 weight_decay=5e-9),
            dict(params=self.conv4.parameters(),
                 weight_decay=5e-8),
        ], lr=lr)
    def forward(self, x, last_action, pred=None):
        B = x.shape[0]
        if self.pre:
            x = (x - x[:, 0, None, :, -1, None])/ x[:, 0, None, :, -1, None] 
        else:
            x = (x )/ x[:, 0, None, :, -1, None] 


        x = self.conv1(x)  
        x = torch.relu(x)
        x = self.conv2(x) 
        re = x
        re[re != re] = 0
        x = torch.relu(x)
        
        prev_w = last_action.view(B, 1, self.num_action-1, 1) 
        if pred is not None:
            pred = pred.view(B, 1, self.num_action-1, 1)
            x = torch.cat([x, prev_w, pred], 1) 
        else:
            x = torch.cat([x, prev_w], 1)  
        x = self.conv3(x)
        x = self.conv4(x)  
        x = torch.cat([
            self.bias.repeat(B, 1),  # (B, 1)
            x[:, 0, :, 0]  # (B, C)
        ], 1)  # (B, 1+C)
    
        x[x != x] = 0
        x = torch.softmax(x, -1)  # (B, 1+C)
        return x, re

class CNN_conv4_stock(nn.Module):
    def __init__(self, num_inputs, action_n, lr=0.001, n_episode_batch=5, nri_d=32, cnn_d=10, shuffle=1, shift=False, nri_lr=0.0005, cnn_d2=15, n=1):
        super(CNN_conv4_stock, self).__init__()
        self.num_inputs = num_inputs
        self.window_size = 31
        self.num_action = action_n
        self.shuffle = shuffle
        self.pre = shift

        D1 = cnn_d
        D2 = cnn_d2 #15

        self.conv1 = nn.Conv2d(self.num_inputs, D1, kernel_size=(1, 2))
        self.conv2 = nn.Conv2d(D1, D2, kernel_size=(1, self.window_size-1))
        self.conv2_5 = nn.Conv2d(1, self.num_action-1 , kernel_size=(self.num_action-1, 1))

        self.conv3 = nn.Conv2d(D2+n, 15, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(15, self.num_action, kernel_size=(self.num_action-1, 1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.leakyrelu = nn.LeakyReLU()

        self.optimizer = optim.AdamW([
            dict(params=[*self.conv1.parameters(),
                            self.bias,
                         ]),
            dict(params=self.conv2.parameters(),
                 weight_decay=5e-9), # L2 reg
            dict(params=self.conv3.parameters(),
                 weight_decay=5e-8), # L2 reg
            dict(params=self.conv2_5.parameters(),
                 weight_decay=5e-9),
            dict(params=self.conv4.parameters(),
                 weight_decay=5e-8),
        ], lr=lr)
    def forward(self, x, last_action, pred=None):
        B = x.shape[0]
        if self.pre:
            x = (x - x[:, 0, None, :, -1, None])/ x[:, 0, None, :, -1, None] 
        else:
            x = (x )/ x[:, 0, None, :, -1, None] 


        x = self.conv1(x)  
        x = self.leakyrelu(x)
        
        x = self.conv2(x)  
        x = x.transpose(1,3)
        x = self.conv2_5(x)
        x = x.permute(0,3,1,2)
        re = x
        re[re != re] = 0
        x = self.leakyrelu(x)

        prev_w = last_action.view(B, 1, self.num_action-1, 1) 
        if pred is not None:
            pred = pred.view(B, 1, self.num_action-1, 1)
            x = torch.cat([x, prev_w, pred], 1) 
        else:
            x = torch.cat([x, prev_w], 1)  
        
        x = self.conv3(x)
        x = self.conv4(x)  

        x = x[:, :, 0, 0]
        x[x != x] = 0
        x = torch.softmax(x, -1)  
        return x, re

class NRI_net(nn.Module):
    def __init__(self, num_inputs, action_n, lr=0.001, n_episode_batch=5, nri_d=32, cnn_d=10, shuffle=1, shift=False, nri_lr=0.0005, cnn_d2=15):
        super(NRI_net, self).__init__()
        self.num_inputs = num_inputs
        self.window_size = 31
        self.num_action = action_n
        self.shuffle = shuffle

        self.encoder = MLPEncoder(self.window_size * self.num_inputs, nri_d, 2, 0, False)

        self.decoder = MLPDecoder(n_in_node=self.num_inputs,edge_types=2,
                                    msg_hid=nri_d,
                                    msg_out=nri_d,
                                    n_hid=nri_d,
                                    do_prob=0,
                                    skip_first=True)



        self.nri_optimizer = optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                       weight_decay=5e-8,lr=nri_lr)
        self.nri_scheduler = optim.lr_scheduler.StepLR(self.nri_optimizer, step_size=200,
                                        gamma=0.5)


        self.batch_size = int(n_episode_batch/self.shuffle)

        self.off_diag = np.ones([self.batch_size, self.batch_size]) - np.eye(self.batch_size)


        self.rel_rec = np.array(encode_onehot(np.where(self.off_diag)[0]), dtype=np.float32)
        self.rel_send = np.array(encode_onehot(np.where(self.off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(self.rel_rec)
        self.rel_send = torch.FloatTensor(self.rel_send)


        self.triu_indices = get_triu_offdiag_indices(self.batch_size)
        self.tril_indices = get_tril_offdiag_indices(self.batch_size)
        
    def train_nri(self, data):
        self.encoder.train()
        self.decoder.train()
        self.nri_optimizer.zero_grad()

        logits = self.encoder(data, self.rel_rec.to(data.device), self.rel_send.to(data.device))
        edges = gumbel_softmax(logits, tau=0.5, hard=False)
        prob = my_softmax(logits, -1)

        output = self.decoder(data, edges, self.rel_rec.to(data.device), self.rel_send.to(data.device), 1)

        target = data[:, :, 1:, :]

        loss_nll = nll_gaussian(output, target, 5e-5)

        loss_kl = kl_categorical_uniform(prob, self.batch_size, 2)

        loss = loss_nll + loss_kl
        loss.backward()
        self.nri_optimizer.step()
        self.nri_scheduler.step()
        return loss_nll.item(), loss_kl.item() 

    def nriLoss(self, emb, x, temperature=0.05):
        B = emb.shape[0]
        C = emb.shape[2]

        x = x.transpose(1,2).transpose(2,3).transpose(0,1)
        emb = emb.transpose(1,2).transpose(0,1).squeeze(-1)
        emb = F.normalize(emb)
        if self.shuffle > 1:
            perm = torch.randperm(B*C)
            x = x.reshape(B*C, self.window_size, -1)
            emb = emb.reshape(B*C, -1)
            B=int(B/self.shuffle)
            C=C*self.shuffle
            x = x[perm,:,:].reshape(C,B,self.window_size, -1)
            emb = emb[perm,:].reshape(C,B,-1)

        with torch.no_grad():
            logits = self.encoder(x, self.rel_rec, self.rel_send)
            edges = gumbel_softmax(logits, tau=0.5, hard=True)

        contrastiveGraph1 = edges[:,:,0].reshape(C,B, -1)
        contrastiveGraph2 = edges[:,:,1].reshape(C,B, -1)
        
        similarity = torch.matmul(emb, emb.transpose(1,2))
        max1 = torch.max(contrastiveGraph1, dim=2, keepdim=True)[0]
        max2 = torch.max(contrastiveGraph2, dim=2, keepdim=True)[0]

        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((B, B)) - np.eye(B)),
            [B, B])
        similarity = similarity.reshape(C, -1)
        similarity = similarity[:, off_diag_idx].reshape(C,B, -1)
        sim_max = torch.max(similarity, dim=2, keepdim=True)[0].detach()
        similarity = similarity-sim_max
        
        positives1 = (contrastiveGraph1.ge(max1)/contrastiveGraph1.ge(max1).sum(-1).unsqueeze(2))*similarity #positive torch.Size([10, 300, 1])
        positives2 = (contrastiveGraph2.ge(max2)/contrastiveGraph2.ge(max2).sum(-1).unsqueeze(2))*similarity #positive torch.Size([10, 300, 1])


        nominator1 = torch.exp(positives1 / temperature).sum(-1)
        nominator2 = torch.exp(positives2 / temperature).sum(-1)


        denominator = torch.exp(similarity / temperature).sum(-1) #


        contrastive = -torch.log(nominator1 / denominator) -torch.log(nominator2 / denominator)

        loss = torch.sum(contrastive)

        nll, kl = self.train_nri(x)
        return loss / (B*C*2), nll, kl

