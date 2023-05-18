from __future__ import absolute_import, print_function, division
import numpy as np
from pgportfolio.constants import *
from .network import Network, ContrastiveLoss

import torch
import torch.nn as nn
import torch.optim as optim

import os
import math
import inspect


class NNAgent:
    def __init__(self, config, restore_dir=None, device="cpu"):
        self.config = config
        self.device = device

        # Model
        self.network = Network(config).to(device)

        # Optimizer
        #self.optimizer = optim.AdamW([
        #    dict(params=[*self.network.conv1.parameters(),
        #                 self.network.bias]),
        #    dict(params=self.network.conv2.parameters(),
        #         weight_decay=5e-9), # L2 reg
        #    dict(params=self.network.conv3.parameters(),
        #         weight_decay=5e-8), # L2 reg
        #], lr=config['training']['learning_rate'])

        self.optimizer = optim.AdamW(self.network.parameters(), lr=config['training']['learning_rate'])

        # Exponential LR Scheduler
        r = config['training']['decay_rate']
        s = config['training']['decay_steps']
        gamma = math.exp(math.log(r) / s)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)

        if restore_dir is not None and os.path.isfile(restore_dir):
            print('restore model from', restore_dir)
            self.load_model(restore_dir)

        self.num_assets = config['input']['coin_number']

        def _generate_graph_outputs_dict():
            def future_price(input_num, y):
                # shape: [B, C+1]
                ones = torch.ones(input_num, 1).to(device)
                return torch.cat([ones, y[:, 0, :]], 1)

            def future_omega(future_price, output):
                # shape: [B, C+1]
                val = future_price * output
                return val / val.sum(-1, keepdim=True)

            def pure_pc(future_omega, output, commission_ratio, input_num):
                c = commission_ratio  # float
                w_t = future_omega[:input_num-1]  # [?, 12]
                w_t1 = output[1:input_num]
                mu = 1 - torch.sum(torch.abs(w_t1[:, 1:] - w_t[:, 1:]), 1) * c
                return mu

            def pv_vector(output, future_price, pure_pc):
                ones = torch.ones(1).to(device)
                return torch.sum(output * future_price, 1) * torch.cat([ones, pure_pc], 0)

            def log_mean_free(output, future_price):
                return torch.mean(torch.log(torch.sum(output * future_price, 1)))

            def portfolio_value(pv_vector):
                return torch.prod(pv_vector)

            def portfolio_weights(output):
                return output

            def mean(pv_vector):
                return torch.mean(pv_vector)

            def log_mean(pv_vector):
                return torch.mean(torch.log(pv_vector))

            def standard_deviation(pv_vector, mean):
                return torch.sqrt(torch.mean((pv_vector - mean) ** 2))

            def sharp_ratio(mean, standard_deviation):
                return (mean - 1) / standard_deviation

            def pv_future(output, y_cont):
                #print(output.shape)
                #print(y_cont.shape)
                return torch.sum(torch.bmm(output[:, 1:].unsqueeze(1) , 0.25*(y_cont[:, 0,:, :]-1)), -1).squeeze()

            #def contrastive_loss(representation, y_cont):
                #print('representation.shape', representation.shape)
                #print('y_cont.shape', y_cont.shape)
            #    return ContrastiveLoss(representation, y_cont)

            def loss(pv_vector, pv_future):
                #print(pv_future)
                #print(torch.log(pv_vector), pv_future)

                #print('contrastive', contrastive_loss) #6757.7402
                #print('pv_future', pv_future.shape)
                #return -torch.mean(torch.log(pv_vector) + torch.sigmoid(-torch.log(pv_vector).detach())*pv_future) + contrastive_loss
                return -torch.mean(torch.log(pv_vector))

            return locals()

        self._graph_nodes = {}
        self._graph_nodes.update(_generate_graph_outputs_dict())
        self._graph_nodes.update({
            'commission_ratio': lambda: config['trading']['trading_consumption'],
            'output': lambda x, prev_w: self.network(x, prev_w),
            'representation':lambda output: self.network.recorded['ConvLayer']
        })

    def evaluate(self, feed_dict, node_names):
        computed = feed_dict.copy()
        #print(computed.keys())

        def _evaluate(node_name):
            if node_name in computed:
                return computed[node_name]

            if node_name in self._graph_nodes:
                f = self._graph_nodes[node_name]
                input_nodes = inspect.getargspec(f).args
                value = f(*map(_evaluate, input_nodes))
                computed[node_name] = value
                return value

            raise Exception('expect {!r} in feed_dict'.format(node_name))

        return list(map(_evaluate, node_names))

    def train(self, x, y, last_w, setw, y_cont):
        # , log_node=None, log_grad=None, log_out=None
        self.network.train()

        feed_dict = dict(x=self.to_tensor(x),
                         y=self.to_tensor(y),
                         prev_w=self.to_tensor(last_w),
                         input_num=x.shape[0],
                         y_cont=self.to_tensor(y_cont))

        # if callable(log_node):
        #     self.network.recording = True

        loss, output = self.evaluate(feed_dict,
                                     ['loss', 'output'])

        # if callable(log_out):
        #     log_out('loss', loss)

        # if callable(log_node):
        #     for name, tensor in self.network.recorded.items():
        #         log_node(name, tensor)

        self.network.zero_grad()
        loss.backward()

        # if callable(log_grad):
        #     for name, grad in self.network.named_parameters():
        #         log_grad(name, grad)

        self.optimizer.step()
        self.lr_scheduler.step()

        setw(self.from_tensor(output[:, 1:]))

    def evaluate_tensors(self, x, y, last_w, setw, y_cont, tensors):
        """
        :param x:
        :param y:
        :param last_w:
        :param setw: a function, pass the output w to it to fill the PVM
        :param tensors:
        :return:
        """
        tensors = list(tensors)
        tensors.append('output')

        assert not np.any(np.isnan(x))
        assert not np.any(np.isnan(y))
        assert not np.any(np.isnan(last_w)),\
            "the last_w is {}".format(last_w)

        feed_dict = dict(x=self.to_tensor(x),
                         y=self.to_tensor(y),
                         prev_w=self.to_tensor(last_w),
                         input_num=x.shape[0],
                         y_cont=self.to_tensor(y_cont))

        results = self.evaluate(feed_dict, tensors)
        results = [self.from_tensor(result) for result in results]

        setw(results[-1][:, 1:])
        return results[:-1]

    # save the variables path including file name
    def save_model(self, path):
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'network': self.network.state_dict(),
        }, path)

    def load_model(self, path):
        state = torch.load(path)
        self.network.load_state_dict(state['network'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])

    # the history is a 3d matrix, return a asset vector
    def decide_by_history(self, history, last_w):
        assert isinstance(history, np.ndarray),\
            "the history should be a numpy array, not %s" % type(
                history)
        assert not np.any(np.isnan(last_w))
        assert not np.any(np.isnan(history))
        history = history[np.newaxis, :, :, :]

        feed_dict = dict(x=self.to_tensor(history),
                         prev_w=self.to_tensor(last_w[np.newaxis, 1:]),
                         input_num=1)
        node_names = ['output']

        self.network.eval()
        with torch.no_grad():
            output = self.evaluate(feed_dict, node_names)[0]

        output = self.from_tensor(output)
        return np.squeeze(output)

    def to_tensor(self, np_array):
        return torch.from_numpy(np_array).float().to(self.device)

    def from_tensor(self, torch_tensor):
        if not isinstance(torch_tensor, torch.Tensor):
            return torch_tensor
        return torch_tensor.detach().cpu().numpy()

    def recycle(self):
        pass # NO OP
