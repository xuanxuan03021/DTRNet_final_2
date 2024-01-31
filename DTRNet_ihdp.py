import torch
import math
import numpy as np
import pandas as pd

import os
import json
import time

from model import DBRNet
from data import get_iter
from eval import curve
from pytorchtools import EarlyStopping


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
    return log_n

def normalize(input,epsilon=1e-6,):
    min=torch.min(input)
    input_temp=input-min
    normalized_input=input_temp/torch.max(input_temp)
    return normalized_input+epsilon

def criterion(out, y, alpha=0.5,beta=0,gamma=0.5, epsilon=1e-6):

    '''reweight'''
    reweight=1/(out[0]+ epsilon)

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
    discrepancy_loss_temp += criterion(delta_p, psi_p)
    discrepancy_loss= beta* (1/(discrepancy_loss_temp+epsilon))

    '''imbalance loss'''
    imbalance_loss= gamma * torch.log(out[5] + epsilon).mean()

    '''total loss'''
    total_loss=factual_loss+treatment_loss+discrepancy_loss+imbalance_loss

    return total_loss,((out[1].squeeze() - y.cuda().squeeze())**2).mean(),treatment_loss,discrepancy_loss,imbalance_loss

def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    return beta * ((y.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[1].squeeze())**2).mean()


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,7"

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
    parser.add_argument('--val_dir', type=str, default='dataset/ihdp/tune', help='dir of eval dataset')

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
    val_path = args.val_dir

    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_matrix = torch.load(args.data_dir + '/data_matrix.pt')
    t_grid_all = torch.load(args.data_dir + '/t_grid.pt')
    t_grid_mise_all = torch.load(args.data_dir + '/t_grid_mise.pt')

    Result = {}
    for model_name in [ 'Vcnet_disentangled']:
        Result[model_name]=[]
        Result[model_name + "mise"] = []

        if model_name == 'Vcnet_disentangled':
                cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
                num_grid = 10
                cfg = [(100, 50, 1, 'relu'), (50, 1, 1, 'id')]
                degree = 2
                knots = [0.33, 0.66]
                model = DBRNet(cfg_density, num_grid, cfg, degree, knots).cuda()
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

            idx_train = torch.load(args.data_split_dir  + '/' + str(_) + '/idx_train.pt')
            idx_test = torch.load(args.data_split_dir  + '/' + str(_) + '/idx_test.pt')
            idx_val = torch.load(val_path + '/' + str(0) + '/idx_train.pt')

            train_matrix = data_matrix[idx_train, :]
            test_matrix = data_matrix[idx_test, :]
            train_matrix_val = data_matrix[idx_val, :]

            t_grid = t_grid_all[:, idx_test]
            t_grid_mise=t_grid_mise_all[idx_test, idx_test]
            # train_matrix, test_matrix, t_grid = simu_data1(500, 200)


            train_loader = get_iter(data_matrix[idx_train, :], batch_size=471, shuffle=True)
            test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)
            val_loader = get_iter(data_matrix[idx_val, :], batch_size=471, shuffle=False)

            # reinitialize model
            model._initialize_weights()

            # define optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)

            print('model : ', model_name)
            # to track the training loss as the model trains
            train_losses = []
            # to track the validation loss as the model trains
            valid_losses = []
            # to track the average training loss per epoch as the model trains
            avg_train_losses = []
            # to track the average validation loss per epoch as the model trains
            avg_valid_losses = []

            early_stopping = EarlyStopping(patience=50, verbose=True)
            for epoch in range(num_epoch):

                for idx, (inputs, y) in enumerate(train_loader):
                    t = inputs[:, 0].cuda()
                    x = inputs[:, 1:].cuda()

                    optimizer.zero_grad()
                    out = model.forward(t, x)
                    loss,factual_loss,treatment_loss,discrepancy_loss,imbalance_loss = criterion(out, y, alpha=alpha, beta=beta,gamma=gamma)
                    loss.backward()
                    optimizer.step()

                model.eval()  # prep model for evaluation
                for idx, (inputs, y) in enumerate(val_loader):
                    # forward pass: compute predicted outputs by passing inputs to the model

                    t = inputs[:, 0].cuda()
                    x = inputs[:, 1:].cuda()
                    y = y.cuda()
                    out = model.forward(t, x)
                    loss,factual_loss,treatment_loss,discrepancy_loss,imbalance_loss = criterion(out, y, alpha=alpha, beta=beta,gamma=gamma)

                    # record validation loss
                    # valid_losses.append(loss.item())
                # print training/validation statistics
                # calculate average loss over an epoch
                train_loss = np.average(train_losses)
                valid_loss = np.average(valid_losses)
                avg_train_losses.append(train_loss)
                avg_valid_losses.append(valid_loss)

                epoch_len = len(str(num_epoch))

                print_msg = (f'[{epoch:>{epoch_len}}/{num_epoch:>{epoch_len}}] ' +
                             f'train_loss: {train_loss:.5f} ' +
                             f'valid_loss: {valid_loss:.5f}')

                # print(print_msg)

                # clear lists to track next epoch
                train_losses = []
                valid_losses = []

                # early_stopping needs the validation loss to check if it has decresed,
                # and if it has, it will make a checkpoint of the current model
                early_stopping(valid_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                if epoch % verbose == 0:
                    print('current epoch: ', epoch)
                    print('loss: ', loss.data)
                    print('factual_loss: ', factual_loss.data)
                    print('treatment_loss: ', treatment_loss.data)
                    print('discrepancy_loss: ', discrepancy_loss.data)
                    print("imbalance_loss",imbalance_loss.data)

            t_grid_hat, mse,mise = curve(model, test_matrix, t_grid,t_grid_mise)

            mse = float(mse)
            mise = float(mise)

            print('current loss: ', float(loss.data))
            print('current test loss: ', mse)
            print('current test mise loss: ', mise)

            print('-----------------------------------------------------------------')
            save_checkpoint({
                'model': model_name,
                'best_test_loss': mse,
                'best_test_mise_loss': mise,
                'model_state_dict': model.state_dict(),
            }, model_name=model_name+"_AMSE_MISE_test_1", checkpoint_dir=cur_save_path)
            print('-----------------------------------------------------------------')

            Result[model_name].append(mse)
            Result[model_name + "mise"].append(mise)

    with open(save_path + '/result_ivc_50_AMSE_MISE_test1.json', 'w') as fp:
        json.dump(Result, fp)