import torch

def cal_pv(y, prob):
    ones = torch.ones(y.shape[0], 1).to(y.device)
    future_price = torch.clamp(torch.cat([ones, y[:, 0, :]], 1), min=0, max=1.5)

    pure_pc = future_price * prob
    pure_pc = pure_pc / pure_pc.sum(-1, keepdim=True)

    w_t = pure_pc[:y.shape[0] - 1]
    w_t1 = prob[1:y.shape[0]]
    mu = 1 - torch.sum(torch.abs(w_t1[:, 1:] - w_t[:, 1:]), 1) * 0.0025

    ones = torch.ones(1).to(y.device)

    pv_vector = torch.sum(prob * future_price, 1) * torch.cat([ones, mu], 0)
    baseline = torch.sum((torch.ones_like(prob)/prob.shape[1]) * future_price, 1)
    return pv_vector, baseline, 1-mu

def get_tensor(x, y, last_w, y_cont, device):
    x = x.to(device).float()
    y = y.to(device).float()
    last_w = last_w.to(device).float()
    y_cont = y_cont.to(device).float()
    return x, y, last_w, y_cont
