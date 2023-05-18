import json
import os
import time
import collections
import argparse
import datetime
import pickle

import numpy as np
import torch
import torch.nn as nn

from pgportfolio.marketdata.datamatrices import DataMatrices
from pgportfolio.tools.configprocess import preprocess_config



import logging
logging.basicConfig(level=logging.INFO)

import math
from utils import *

rolling = True

parser = argparse.ArgumentParser(description='PyTorch PM Args')

parser.add_argument('--lr', type=float, default=0.001, metavar='G',
                    help='learning rate (default: 0.0001)')


parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')

parser.add_argument('--seed', type=int, default=7, metavar='N',
                    help='random seed (default: 7)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')

parser.add_argument('--n_episode', type=int, default=6, metavar='N',
                    help='number of episode for each batch (default: 5)')
parser.add_argument('--num_steps', type=int, default=10000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--rolling_steps', type=int, default=20, metavar='N',
                    help='maximum number of steps (default: 1000000)')

parser.add_argument('--no_cuda', action="store_true", default=False,
                    help='run on CUDA (default: False)')
parser.add_argument('--cuda_no', type=int, default=0,
                    help='cuda no (default: 2)')
parser.add_argument('--model_name', type=str, default='ours',
                    help='model name [dpm, sarl, ours, sarl_v2, dpm_v2]')

parser.add_argument('--test_portion', type=float, default=0.08,
                    help='test portion (default: 0.08, 0.1605)')
parser.add_argument('--smoothing_days', type=int, default=5,
                    help='smoothing days (default: 5)')

parser.add_argument('--stocks', type=int, default=1,
                    help='smoothing days (default: 1)')
parser.add_argument('--buffer_biased', type=float, default=5e-5,
                    help='buffer_biased for sampling (default: 5e-5)') #stock 2e-4 #btc 5e-5

parser.add_argument('--nri_d', type=int, default=32,
                    help='nri dimension (default: 32)')
parser.add_argument('--cnn_d', type=int, default=10,
                    help='cnn dimension (default: 10)')
parser.add_argument('--cnn_d2', type=int, default=15,
                    help='cnn2 dimension (default: 10)')
parser.add_argument('--nri_shuffle', type=int, default=1,
                    help='nri batch shuffle, shuffle=1 means not shuffle (default: 1)')
parser.add_argument('--nri_lr', type=float, default=0.0005, metavar='G',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--input_shift', action='store_true', default=False,
                    help='model input minus one (default: False)')


parser.add_argument('--L1_w', type=float, default=2,
                    help='L1 weight (default: 2)')
parser.add_argument('--L1_baseline', action='store_true',
                    help='L1 baseline (default: False)')
parser.add_argument('--L3_w', type=float, default=1,
                    help='smoothing weight (default: 1)')
parser.add_argument('--L2_w', type=float, default=1e-6,
                    help='contrastive weight (default: 1e-6)')
parser.add_argument('--L3_baseline', action='store_true',
                    help='L3 baseline (default: False)')
parser.add_argument('--L3_w_const', action='store_true', default=False,
                    help='L3 weight constant (default: False)')

args = parser.parse_args()

if args.seed != -1:
    torch.manual_seed(args.seed)
args.useCuda = not args.no_cuda

device = torch.device("cuda:{}".format(args.cuda_no) if (torch.cuda.is_available() and args.useCuda) else "cpu")

stock = args.stocks
if stock ==0:
    action_n = 10 # stock 10 btc 11
    args.buffer_biased = 2e-4
    args.test_portion = 0.1605
else:
    action_n = 11 # stock 10 btc 11


def main():
    results_dict = {'eval_rewards': [],
                    'eval_actions': [],
                    'eval_returns': [],
                    'l1_losses': [],
                    'l2_losses': [],
                    'l3_losses': [],
                    'l3_losses_w': [],
                    'nri_nll_losses': [],
                    'nri_kl_losses': []
                    }


    base_dir = (os.getcwd() + '/models_0911_sarl/' + '/RT(stock_layer)-NoVAL-stock_'+ str(stock) + '-seed_' + str(args.seed) + '/' +
                                'lr_' + str(args.lr) + '-steps_' + str(args.num_steps) + '-rolling_' + str(args.rolling_steps) +
                                '-smooth_' + str(args.smoothing_days) + '-nri_d_' + str(args.nri_d) + '-cnn_d_' + str(args.cnn_d) +
                                '-cnn_d2_' + str(args.cnn_d2) +
                                '-nri_batch_shuffle_' + str(args.nri_shuffle) + '-input_shift_' + str(args.input_shift) +
                                '-nri_lr_' + str(args.nri_lr)  + '-l2_w'+ str(args.L2_w) + '-l3_w'+ str(args.L3_w) +
                                '-n_episode' + str(args.n_episode)  + '-model' + args.model_name  +'/' )

    run_number = 0
    while os.path.exists(base_dir + str(run_number)):
        run_number += 1
    base_dir = base_dir + str(run_number)
    os.makedirs(base_dir)

    with open(base_dir + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    with open("pgportfolio/" + "net_config.json") as file:
        config = json.load(file)
    config = preprocess_config(config)

    train_config = config["training"]
    input_config = config["input"]

  

    if (args.model_name == 'dpm') or (args.model_name == 'sarl'):
        from policies.policy import CNN_Policy as Policy
        pi = Policy(3, action_n, args, device)

    elif (args.model_name == 'dpm_v2') or (args.model_name == 'sarl_v2') or (args.model_name == 'ours'):
        from policies.policy import Conv4_Policy as Policy
        pi = Policy(3, action_n, args.lr, args.n_episode*train_config['batch_size'], args.nri_d, args.cnn_d, args.nri_shuffle, args.input_shift, args.nri_lr, args.cnn_d2, args, device)


    matrix = DataMatrices.create_from_config(config, stock, args)


    test_set = matrix.get_test_set()
    training_set = matrix.get_training_set()

    if 'sarl' in args.model_name:
        pi.sarl_net.train()
        for n_epi in range(args.num_steps): 
            batch = matrix.next_batch(args.n_episode)

            x = torch.from_numpy(batch["X"]) 
            y = torch.from_numpy(batch["y"]) 
            last_w = torch.from_numpy(batch["last_w"]) 
            setw = batch["setw"]
            y_cont = torch.from_numpy(batch['y_cont'])
            sarl_loss = pi.sarl_net.sarl_train_net(x, y, last_w=last_w,  y_cont=y_cont, device=device)
        pi.sarl_net.eval()

    pi.net.train()
    for n_epi in range(args.num_steps): 

        batch = matrix.next_batch(args.n_episode)

        x = torch.from_numpy(batch["X"]) 
        y = torch.from_numpy(batch["y"]) 
        last_w = torch.from_numpy(batch["last_w"]) 
        setw = batch["setw"]
        y_cont = torch.from_numpy(batch['y_cont'])
        
        
        pv, L1, L2, L3, nll, kl, L3_w = pi.train_net(x, y, last_w=last_w, setw=setw, y_cont=y_cont, device=device, args=args)

        results_dict['l1_losses'].append((n_epi, L1))
        results_dict['l2_losses'].append((n_epi, L2))
        results_dict['l3_losses'].append((n_epi, L3))
        results_dict['l3_losses_w'].append((n_epi, L3_w))
        results_dict['nri_nll_losses'].append((n_epi, nll))
        results_dict['nri_kl_losses'].append((n_epi, kl))


    last_action = np.zeros(action_n)
    last_action[0] = 1

    last_action = torch.from_numpy(last_action).float().to(device).unsqueeze(0)
    portfolio_value = 1

    steps = 0

    for b, i in enumerate(test_set['X']):
        with torch.no_grad():
            i = torch.from_numpy(i).float().to(device).unsqueeze(0)
            pi.net.eval()
            

            if 'sarl' in args.model_name:
                pred, _ = pi.sarl_net(i.to(device).float())
                prob, _ = pi.net(i, last_action[:,1:],  pred.detach().argmax(dim=1))
            else:
                prob, _ = pi.net(i, last_action[:,1:])

            y = torch.from_numpy(test_set["y"][b]).to(device).float().unsqueeze(0)

            ones = torch.ones(1, 1).to(device)
            future_price = torch.cat([ones, y[:, 0, :]], 1)

            w_t = last_action.squeeze(0).cpu().detach().numpy()  # [?, 12]
            w_t1 = prob.squeeze(0).cpu().detach().numpy()
            mu = calculate_pv_after_commission(w_t1, w_t, 0.0025)


            pv_vector = torch.sum(prob * future_price, 1) * mu
            portfolio_value *= pv_vector.item()


            last_action = prob * future_price / torch.sum(prob * future_price, 1)

            results_dict['eval_rewards'].append((b, portfolio_value))
            results_dict['eval_actions'].append((b, prob.detach().cpu().numpy()))
            results_dict['eval_returns'].append((b, pv_vector.detach().cpu().numpy()))

        if b%30 == 0:
            print('n_epi', b, 'pv', portfolio_value)
            with open(base_dir + '/results', 'wb') as f:
                pickle.dump(results_dict, f)


        # rolling

        if rolling:
            matrix.append_experience(None)
            
            if 'sarl' in args.model_name:
                pi.sarl_net.train()
                for i in range(args.rolling_steps): 
                    batch = matrix.next_batch(args.n_episode)

                    x = torch.from_numpy(batch["X"]) 
                    y = torch.from_numpy(batch["y"]) 
                    last_w = torch.from_numpy(batch["last_w"]) 
                    setw = batch["setw"]
                    y_cont = torch.from_numpy(batch['y_cont'])

                    sarl_loss = pi.sarl_net.sarl_train_net(x, y, last_w=last_w, y_cont=y_cont, device=device)
                pi.sarl_net.eval()

            pi.net.train()
            for i in range(args.rolling_steps): 
                steps += 1
                batch = matrix.next_batch(args.n_episode)

                x = torch.from_numpy(batch["X"]) 
                y = torch.from_numpy(batch["y"]) 
                last_w = torch.from_numpy(batch["last_w"])
                setw = batch["setw"]
                y_cont = torch.from_numpy(batch['y_cont'])
                
                
                pv, L1, L2, L3, nll, kl, L3_w = pi.train_net(x, y, last_w=last_w, setw=setw, y_cont=y_cont, device=device, args=args)

                results_dict['l1_losses'].append((n_epi+steps, L1))
                results_dict['l2_losses'].append((n_epi+steps, L2))
                results_dict['l3_losses'].append((n_epi+steps, L3))
                results_dict['l3_losses_w'].append((n_epi+steps, L3_w))
                results_dict['nri_nll_losses'].append((n_epi+steps, nll))
                results_dict['nri_kl_losses'].append((n_epi+steps, kl))


    print('n_epi', b, 'pv', portfolio_value)

    with open(base_dir + '/results', 'wb') as f:
        pickle.dump(results_dict, f)

if __name__ == '__main__':
    main()