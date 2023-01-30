from synthetic_dataset import generate_gaussians
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
from vcnet import Vcnet
from model import iCXAI_model
#from distance import calculate_disc
from OpenXAI.openxai.dataloader import return_loaders
import mdn
from mdn import gaussian_probability
from mdn import MDN
import pandas as pd
from data import get_iter
from evaluation_causal import curve

epsilon=0.000001
psilon=0.000001
use_cuda = torch.cuda.is_available()
torch.manual_seed(1314)
#device = torch.device("cuda:0" if use_cuda else "cpu")
device_ids = [0,1]
print(use_cuda)
#os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
gpu = use_cuda


class Dataset_torch(torch.utils.data.Dataset):

    def __init__(self, X, y, transform=None):
        self.data = X
        self.label= y
        self.dataLen = X.shape[0]
    def __len__(self):
        return self.dataLen

    def __getitem__(self, index):
        data = self.data[index,:]
        label = self.label[index]

        return data, label



# gamma,delta,psi, y,t,t_representation
# from util import *
#
#
def normalized_log(input,epsilon=1e-6,):
    # print(input.shape)
    # print(torch.unsqueeze((torch.min(input,dim=1)[0]), dim=1).shape)
    temp= input-torch.unsqueeze((torch.min(input,dim=1)[0]), dim=1)
    #prevent 0, which leads to -inf in the log of next step
    normalized_input=(temp+epsilon)/torch.unsqueeze(torch.max(temp,dim=1)[0], dim=1)
    log_n=torch.log(normalized_input)
  #  print(log_n)
    return log_n


def criterion_all(out, y,t, alpha=0.5,beta=0.5,gamma_=0.5,type_t=1,type_y=1, epsilon=1e-6,):
    #return ((out[1].squeeze() - y.squeeze())**2).mean() - alpha * torch.log(out[0] + epsilon).mean()

    ''' Compute sample reweighting (general propensity score)'''
   # print("t_prediction",out[4])
    if type_t==1:
        ps = torch.sum(out[4][0] * gaussian_probability(out[4][1], out[4][2], t.unsqueeze(1)), dim=1)
       # print(ps)
        sample_weight=1/ps
    else:
        sample_weight = 1/(out[4]+epsilon)
 #   print("sample_weight",sample_weight)

    ''' Construct factual loss function (reweight using inverse propensity score) '''
   # print("y_prediction",out[3])

    if type_y == 1:
   #     print(out[3])
        #print(sample_weight)
        risk= (sample_weight *((out[3] - y).pow(2))).mean()
    elif type_y == 2:
        criterion = nn.BCELoss(reduction='none').cuda()
        res =criterion(out[3], y.unsqueeze(dim=1))
        risk= (sample_weight*res).mean()
    elif type_y == 3:
        criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        res = criterion(out[3], y)
        risk=(sample_weight*res).mean()


    ''' Imbalance error '''
    psi_p=normalized_log(out[2])
    # print(out[2])
    # exit()
    # psi_n= out[2]-torch.min(out[2],dim=1)
    # psi_p=psi_n/torch.max(out[2],dim=1)
    # psi_p=torch.log(out[2])
   # print("psi_p",psi_p)
    t_p=normalized_log(out[5])
    criterion=torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    imb_error=1/criterion(psi_p,t_p)


    '''treatment loss'''
    if type_t == 1:
        risk_t = mdn.mdn_loss(out[4][0], out[4][1], out[4][2],t.unsqueeze(1))
    elif type_t == 2:
        criterion = nn.BCELoss().cuda()
        risk_t =criterion(out[4], t)
    elif type_t == 3:
        criterion = nn.CrossEntropyLoss().cuda()
        risk_t = criterion(out[4], t)

    '''discrepancy loss'''

    gamma_p=normalized_log(out[0])
    delta_p=normalized_log(out[1])

    criterion=torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    discrepancy_loss_temp = criterion(gamma_p, delta_p)
    discrepancy_loss_temp += criterion(delta_p, psi_p)
    discrepancy_loss=1/discrepancy_loss_temp

    ''' Total error '''
    tot_error = risk
#    print("factual_risk",risk)

    if alpha > 0:
        tot_error = tot_error + alpha*imb_error
#        print("imb",imb_error)
    if beta > 0:
        tot_error = tot_error + beta * risk_t
  #      print("treatment,",risk_t)
    if gamma_ > 0:
        tot_error = tot_error + gamma_ * discrepancy_loss
    #     print("discrepency",discrepancy_loss)
    # print("tot_error", tot_error)

    return tot_error,risk,imb_error,risk_t,discrepancy_loss

def test(args, model, test_dataset, criterion,type_y=2,type_t=1):
    model.eval()
    loss = 0
    total_samples = 0
    for idx, (inputs, y) in enumerate(test_dataset):
        t = inputs[:, 0].float().cuda()
        x = inputs[:, 1:].float().cuda()
        y = y.float().cuda()
        output = model.forward(t, x)

        n_samples = inputs.shape[0]
        loss_batch,risk,imb_error,risk_t,discrepancy_loss = criterion(output, y,t, alpha=args.alpha,beta=args.beta,gamma_=args.gamma,type_y=type_y,type_t=type_t)
        loss += loss_batch * n_samples
        total_samples += n_samples
    loss_final_average = loss / total_samples
    return loss_final_average

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/ihdp', help='dir of data matrix')
    parser.add_argument('--save_dir', type=str, default='logs/ihdp/eval', help='dir to save result')
    # training
    parser.add_argument('--n_epochs', type=int, default=800, help='num of epochs to train')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size to train to train')
    parser.add_argument('--shuffle', type=bool, default=True, help='if shuffle the dataset')
    # print train info
    parser.add_argument('--verbose', type=int, default=10, help='print train info freq')

    #hyperparameter tuning
    parser.add_argument('--alpha', type=float, default=0.2, help='weight for imbalance error')
    parser.add_argument('--beta', type=float, default=0.5, help='weight for treatment loss')
    parser.add_argument('--gamma', type=float, default=0.2, help='weight for discrepancy loss')

    # plot adrf
    parser.add_argument('--plt_adrf', type=bool, default=True, help='whether to plot adrf curves. (only run two methods if set true; '
                                                                    'the label of fig is only for drnet and vcnet in a certain order)')


    args = parser.parse_args()
    cfg_rep = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
    num_grid = 10
    cfg_y = [(100, 50, 1, 'relu'), (50, 1, 1, 'id')]
    cfg_t= [(100, 50, 1, 'relu'), (50, 1, 1, 'id')]
    degree = 2
    knots = [0.33, 0.66]
    init_lr = 0.0001
    # alpha = 0.5
    lambda_= 5e-3
    momentum = 0.9
    '''specify the type of y an t each time!'''
    type_y=1
    type_t=1

    #assert the valid latent dimension settings
    assert (cfg_y[0][0]==cfg_t[0][0]==2*cfg_rep[-1][1])

    num_epoch = args.n_epochs
    alpha=args.alpha
    beta=args.beta
    gamma=args.gamma

    batch_size=args.batch_size
    shuffle=args.shuffle
    verbose=args.verbose

    load_path = args.data_dir

    model = iCXAI_model(num_grid,degree, knots, cfg_rep=cfg_rep,
                 cfg_y = cfg_y,
                 cfg_t=cfg_t,
                num_gaussians=20,type_t=type_t)
    model._initialize_weights()
    # model_vc=Vcnet( [(6, 50, 1, 'relu'), (50, 50, 1, 'relu')], num_grid, [(50, 50, 1, 'relu'), (50, 1, 1, 'id')], degree, knots)
    # print('The vc model:')
    # print(model_vc)
    model = model.cuda()
    model = torch.nn.parallel.DataParallel(model, device_ids=device_ids, dim=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=lambda_, nesterov=True)

    print('The model:')
    print(model)

    data_matrix = torch.load(args.data_dir + '/data_matrix.pt')
    t_grid_all = torch.load(args.data_dir + '/t_grid.pt')

    idx_train = torch.load('dataset/ihdp/eval/' + str(0) + '/idx_train.pt')
    idx_test = torch.load('dataset/ihdp/eval/' + str(0) + '/idx_test.pt')

    train_matrix = data_matrix[idx_train, :]
    test_matrix = data_matrix[idx_test, :]
    t_grid = t_grid_all[:, idx_test]

    # train_matrix, test_matrix, t_grid = simu_data1(500, 200)
    train_loader = get_iter(data_matrix[idx_train, :], batch_size=471, shuffle=True)
    test_loader = get_iter(data_matrix[idx_test, :], batch_size=data_matrix[idx_test, :].shape[0], shuffle=False)



    for epoch in range(num_epoch):

        for idx, (inputs, y) in enumerate(train_loader):
            #TODO: change the treatment variable
            t = inputs[:, 0].float().cuda()
            x = inputs[:, 1:].float().cuda()
            y=y.float().cuda()
            optimizer.zero_grad()
            out = model.forward(t, x)
           # print(out[4])
            loss,risk,imb_error,risk_t,discrepancy_loss = criterion_all(out, y,t, alpha=alpha,beta=beta,gamma_=gamma,type_y=type_y,type_t=type_t)
            loss.backward()
            optimizer.step()


        if epoch % verbose == 0:
            print('current epoch: ', epoch)
            print('training loss: ', loss.data)
            print('factual loss: ', risk.data)
            print('imb loss: ', imb_error.data)
            print('treatment loss: ', risk_t.data)
            print('discrepancy loss: ', discrepancy_loss.data)

            test_loss = test(args, model, test_loader, criterion_all,type_y=type_y,type_t=type_t)
            print(test_loss)

    #out = model.forward(dataset_train.iloc[:,1], dataset_train)
    t_grid_hat, mse = curve(model, test_matrix, t_grid)

    mse = float(mse)
    print('current loss: ', float(loss.data))
    print('current test loss: ', mse)
    print('-----------------------------------------------------------------')
    grid=[]
    grid.append(t_grid_hat)
