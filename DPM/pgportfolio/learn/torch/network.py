import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def ContrastiveLoss(emb, y, temperature=0.5):
    y = y[:, 0, :]
    B = y.shape[0]
    C = y.shape[1]

    emb = emb.transpose(1,2)
    emb = emb.reshape(B*C, -1)
    y = y.reshape(B*C, -1)
    y = y/y.sum(1).view(B*C,1)

    similarity = F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=2)

    def each_loss(i, j):
        numerator = torch.exp(similarity[i, j] / temperature)

        mask = torch.ones((B*C, )).to(emb.device).scatter_(0, torch.tensor([i]).to(emb.device), 0.0)
        denominator = torch.sum(mask*torch.exp(similarity[i, :]/temperature))

        return -torch.log(numerator / denominator).squeeze(0)

    loss = 0

    for k in range(0, B*C):
        y_k = torch.ones((B*C,y.shape[1])).to(emb.device)*y[k,:]
        criterion = nn.KLDivLoss(reduction='none')
        #print(criterion(y , y_k).sum(1).squeeze().data.shape)

        _, neighbor = torch.topk(criterion(y , y_k).sum(1).squeeze(), 2, dim=-1, largest=False, sorted=True)

        #print(neighbor.shape)
        #print(neighbor[1])

        loss += each_loss(k, neighbor[1])
    return loss


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self,q, k, v, d_k, mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if mask is not None:
                mask = mask.unsqueeze(1)
                scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output

    def forward(self, x, mask=None):

        bs = x.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(x).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(x).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(x).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps)

        if torch.isnan(norm).any():
            print('1', self.alpha)
            print('2', x.mean(dim=-1, keepdim=True))
            print('3', x.std(dim=-1, keepdim=True))
            print('4', (x - x.mean(dim=-1, keepdim=True)))
            print('5', x)
        assert not torch.isnan(norm).any()

        return norm

def tflearn_default_conv2d_init_(conv2d, factor=1.0):
    shape = [*conv2d.kernel_size, conv2d.in_channels, conv2d.out_channels]

    input_size = 1.0
    for dim in shape[:-1]:
        input_size *= float(dim)

    max_val = math.sqrt(3 / input_size) * factor
    conv2d.weight.data.uniform_(-max_val, max_val)
    conv2d.bias.data.zero_()

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.F = config["input"]["feature_number"]
        self.C = config["input"]["coin_number"]
        self.W = config["input"]["window_size"]

        D1 = 3
        D2 = 10


        self.conv1 = nn.Conv2d(self.F, D1, kernel_size=(1, 2))
        self.conv2 = nn.Conv2d(D1, D2, kernel_size=(1, self.W-1))

        #self.head1 = MultiHeadAttention(4, self.hidden_dim1)

        self.conv3 = nn.Conv2d(D2+1, 1, kernel_size=(1, 1))
        self.bias = nn.Parameter(torch.zeros(1))


        self.recorded = {}
        self.recording = True

        #self.reset_parameters()

    #def reset_parameters(self):
    #    for m in self.modules():
    #        if isinstance(m, nn.Conv2d):
    #            tflearn_default_conv2d_init_(m)
    #    self.bias.data.zero_()

    def forward(self, *a, **ka):
        self.recorded.clear()
        if self.recording:
            for name, output in self.nodeiter(*a, **ka):
                self.recorded[name] = output.detach()
        else:
            for _, output in self.nodeiter(*a, **ka):
                pass
        return output

    def nodeiter(self, x, prev_w):
        # x: (B, F, C, W)
        # prev_w: (B, C)

        assert isinstance(x, torch.Tensor)
        assert isinstance(prev_w, torch.Tensor)
        assert x.shape[1:] == (self.F, self.C, self.W)
        assert prev_w.shape[1:] == (self.C,)
        assert prev_w.shape[:1] == x.shape[:1]

        B = x.shape[0]

        # x.shape == (B, F, C, W)
        # x[:, 0, :, -1].shape == (B, C)
        # x[:, 0, None, :, -1, None].shape == (B, 1, C, 1)
        x = x / x[:, 0, None, :, -1, None] # normalize???

        x = self.conv1(x)  # (B, 3, C, W-1)
        x = torch.relu(x)
        yield 'ConvLayer', x
        #print('1.', x.sahpe)

        x = self.conv2(x)  # (B, 10, C, 1)
        x = torch.relu(x)
        yield 'EIIE_Dense', x
        #print('2.', x.sahpe)

        prev_w = prev_w.view(B, 1, self.C, 1)  # (B, 1, C, 1)
        x = torch.cat([x, prev_w], 1)  # (B, 11, C, 1)
        x = self.conv3(x)  # (B, 1, C, 1)
        yield 'EIIE_Output_WithW', x
        #print('3.', x.sahpe)

        x = torch.cat([
            self.bias.repeat(B, 1),  # (B, 1)
            x[:, 0, :, 0]  # (B, C)
        ], 1)  # (B, 1+C)
        yield 'voting', x

        x = torch.softmax(x, -1)  # (B, 1+C)
        yield 'softmax_layer', x
