import torch
import math
import numpy as np
import pandas as pd

import os
import json
import time

from based_on_vcnet import Vcnet
from data import get_iter
from based_on_vcnet_evaluation import curve

import argparse

def adjust_learning_rate(optimizer, init_lr, epoch):
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * epoch / num_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = init_lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = init_lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, model_name='', checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, model_name + '_ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)

# criterion
def log(input,epsilon=1e-6,):

    log_n=torch.log(input+epsilon)
  #  print(log_n)
    return log_n

def normalize(input,epsilon=1e-6,):
    min=torch.min(input)
    input_temp=input-min
    normalized_input=input_temp/torch.max(input_temp)
    return normalized_input+epsilon

#g, Q,gamma,delta,psi
def criterion(out, y, alpha=0.5,beta=0,gamma=0.5, epsilon=1e-6):

    '''reweight'''
    reweight=(1/(out[0]+ epsilon))*0.3
   # reweight=1

    '''factual loss'''
    factual_loss=(reweight*((out[1].squeeze() - y.cuda().squeeze())**2)).mean()

    '''treatment loss'''
    treatment_loss= - alpha * torch.log(out[0] + epsilon).mean()

    '''discrepancy loss'''
    gamma_p = log(out[2])
    delta_p = log(out[3])
    psi_p= log(out[4])

    criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=True).cuda()
    discrepancy_loss_temp = criterion(gamma_p, delta_p)
    #print(gamma_p)
    #print(delta_p)
   # print("here",discrepancy_loss_temp)
    # print("here",discrepancy_loss_temp)
    discrepancy_loss_temp += criterion(delta_p, psi_p)
    #print("there",discrepancy_loss_temp)

    #  print("there",discrepancy_loss_temp)
    discrepancy_loss= beta* (1/(discrepancy_loss_temp+epsilon))

    '''imbalance loss'''
    imbalance_loss= gamma * torch.log(out[5] + epsilon).mean()

    '''total loss'''
    total_loss=factual_loss+treatment_loss+discrepancy_loss+imbalance_loss
    # print("factual_loss",factual_loss)
    # print("treatment_loss",treatment_loss)
    # print("discrepancy_loss",discrepancy_loss)

    return total_loss,((out[1].squeeze() - y.cuda().squeeze())**2).mean(),treatment_loss,discrepancy_loss,imbalance_loss

def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    return beta * ((y.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[1].squeeze())**2).mean()


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,7"

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1314)
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    print(use_cuda)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

    device_ids = [0,1,2,3]

    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)
    gpu = use_cuda

    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/ihdp', help='dir of data matrix')
    parser.add_argument('--data_split_dir', type=str, default='dataset/ihdp/eval', help='dir of data split')
    parser.add_argument('--save_dir', type=str, default='logs/ihdp/eval', help='dir to save result')
    # common
    parser.add_argument('--num_dataset', type=int, default=50, help='num of datasets to train')

    # training
    parser.add_argument('--n_epochs', type=int, default=800, help='num of epochs to train')

    # print train info
    parser.add_argument('--verbose', type=int, default=10, help='print train info freq')


    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    # fixed parameter for optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9

    # targeted regularization optimizer
    tr_wd = 1e-3

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    # load
    load_path = args.data_split_dir
    num_dataset = args.num_dataset

    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_matrix = torch.load(args.data_dir + '/data_matrix.pt')
    t_grid_all = torch.load(args.data_dir + '/t_grid.pt')

    Result = {}
    for model_name in [ 'Vcnet_disentangled']:

    #for model_name in ['Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr']:
        Result[model_name]=[]
        if model_name == 'Vcnet_disentangled':
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(100, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model = Vcnet(cfg_density, num_grid, cfg, degree, knots).cuda()
            print(model)
            model._initialize_weights()

        if model_name == 'Vcnet_disentangled':
            init_lr = 0.00005
            alpha = 0.6
            beta=0.6
            gamma=0.1
            Result['Vcnet_disentangled'] = []

        for _ in range(num_dataset):

            cur_save_path = save_path + '/' + str(_)
            if not os.path.exists(cur_save_path):
                os.makedirs(cur_save_path)

            idx_train = torch.load('dataset/ihdp/eval/' + str(_) + '/idx_train.pt')
            idx_test = torch.load('dataset/ihdp/eval/' + str(_) + '/idx_test.pt')

            train_matrix = data_matrix[idx_train, :]
            test_matrix = data_matrix[idx_test, :]
            t_grid = t_grid_all[:, idx_test]

            # train_matrix, test_matrix, t_grid = simu_data1(500, 200)
            train_loader = get_iter(data_matrix[idx_train, :], batch_size=471, shuffle=True)

            # reinitialize model
            model._initialize_weights()
            test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

            # define optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)

            print('model : ', model_name)

            for epoch in range(num_epoch):

                for idx, (inputs, y) in enumerate(train_loader):
                    t = inputs[:, 0].cuda()
                    x = inputs[:, 1:].cuda()

                    optimizer.zero_grad()
                    out = model.forward(t, x)
                    #print("forward ", after_forward - start)
                    loss,factual_loss,treatment_loss,discrepancy_loss,imbalance_loss = criterion(out, y, alpha=alpha, beta=beta,gamma=gamma)
                    loss.backward()
                    optimizer.step()
                    #print("after backward ", after_backward - after_forward)
                    # if epoch == 1:
                    #     input()

                if epoch % verbose == 0:
                    print('current epoch: ', epoch)
                    print('loss: ', loss.data)
                    print('factual_loss: ', factual_loss.data)
                    print('treatment_loss: ', treatment_loss.data)
                    print('discrepancy_loss: ', discrepancy_loss.data)
                    print("imbalance_loss",imbalance_loss.data)

            t_grid_hat, mse = curve(model, test_matrix, t_grid)

            mse = float(mse)
            print('current loss: ', float(loss.data))
            print('current test loss: ', mse)
            # print('-----------------------------------------------------------------')
            # save_checkpoint({
            #     'model': model_name,
            #     'best_test_loss': mse,
            #     'model_state_dict': model.state_dict(),
            # }, model_name=model_name, checkpoint_dir=cur_save_path)
            # print('-----------------------------------------------------------------')

            Result[model_name].append(mse)
            # #
            with open(save_path + '/result_ivc_50_03reweight.json', 'w') as fp:
                json.dump(Result, fp)