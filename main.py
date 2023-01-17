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


def criterion_all(out, y,t, alpha=0.5,beta=0.5,gamma_=0.5,type_t=1,type_y=1, epsilon=1e-6,):
    #return ((out[1].squeeze() - y.squeeze())**2).mean() - alpha * torch.log(out[0] + epsilon).mean()

    ''' Compute sample reweighting (general propensity score)'''
   # print("t_prediction",out[4])
    if type_t==1:
        sample_weight = torch.sum(out[4][0] * gaussian_probability(out[4][1], out[4][2], t.unsqueeze(1)), dim=1)
    else:
        sample_weight = 1/(out[4]+epsilon)
 #   print("sample_weight",sample_weight)
    ''' Construct factual loss function '''
   # print("y_prediction",out[3])

    if type_y == 1:
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
    psi_p=torch.nn.functional.log_softmax(out[2], dim=1)
    t_p=torch.nn.functional.log_softmax(out[5], dim=1)
    criterion=torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    imb_error=-criterion(psi_p,t_p)


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

    gamma_p=torch.nn.functional.log_softmax(out[0], dim=1)
    delta_p=torch.nn.functional.log_softmax(out[1], dim=1)

    criterion=torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    discrepancy_loss = -criterion(gamma_p, delta_p)
    discrepancy_loss -= criterion(delta_p, psi_p)

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
    parser.add_argument('--data_dir', type=str, default='dataset/simu1/eval/0', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='logs/simu1/eval', help='dir to save result')

    # training
    parser.add_argument('--n_epochs', type=int, default=800, help='num of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size to train to train')
    parser.add_argument('--shuffle', type=bool, default=True, help='if shuffle the dataset')
    # print train info
    parser.add_argument('--verbose', type=int, default=10, help='print train info freq')

    #hyperparameter tuning
    parser.add_argument('--alpha', type=float, default=0.01, help='weight for imbalance error')
    parser.add_argument('--beta', type=float, default=0.5, help='weight for treatment loss')
    parser.add_argument('--gamma', type=float, default=0.01, help='weight for discrepancy loss')


    args = parser.parse_args()
    cfg_rep = [(9, 50, 1, 'relu'), (50, 50, 1, 'relu')]
    num_grid = 10
    cfg_y = [(100, 50, 1, 'relu'), (50, 1, 1, 'sigmoid')]
    cfg_t= [(100, 50, 1, 'relu'), (50, 1, 1, 'sigmoid')]
    degree = 2
    knots = [0.33, 0.66]
    init_lr = 0.005
    # alpha = 0.5
    lambda_= 5e-3
    momentum = 0.9
    type_y=2
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

    gauss_params = {
        'n_samples': 250,
        'dim': 10,
        'n_clusters': 10,
        'distance_to_center': 5,
        'test_size': 0.25,
        'upper_weight': 1,
        'lower_weight': -1,
        'seed': 564,
        'sigma': None,
        'sparsity': 0.5
    }

    if gauss_params is None:
        gauss_params = {
            'n_samples': 2500,
            'dim': 20,
            'n_clusters': 10,
            'distance_to_center': 5,
            'test_size': 0.25,
            'upper_weight': 1,
            'lower_weight': -1,
            'seed': 564,
            'sigma': None,
            'sparsity': 0.25
        }
    #
    dataset_train, probs_train, masks_train, weights_train, masked_weights_train, cluster_idx_train = generate_gaussians(
        gauss_params['n_samples'],
        gauss_params['dim'],
        gauss_params['n_clusters'],
        gauss_params['distance_to_center'],
        gauss_params['test_size'],
        gauss_params['upper_weight'],
        gauss_params['lower_weight'],
        gauss_params['seed'],
        gauss_params['sigma'],
        gauss_params['sparsity']).dgp_vars(data_name="train")

    dataset_test, probs_test, masks_test, weights_test, masked_weights_test, cluster_idx_test = generate_gaussians(
        gauss_params['n_samples'],
        gauss_params['dim'],
        gauss_params['n_clusters'],
        gauss_params['distance_to_center'],
        gauss_params['test_size'],
        gauss_params['upper_weight'],
        gauss_params['lower_weight'],
        gauss_params['seed'],
        gauss_params['sigma'],
        gauss_params['sparsity']).dgp_vars(data_name="test")




 #   loader_train, loader_test = return_loaders(data_name="synthetic", batch_size=5,gauss_params=gauss_params)

    model = iCXAI_model(num_grid,degree, knots, cfg_rep=cfg_rep,
                 cfg_y = cfg_y,
                 cfg_t=cfg_t,
                num_gaussians=20,type_t=type_t)
    #model._initialize_weights()
    # model_vc=Vcnet( [(6, 50, 1, 'relu'), (50, 50, 1, 'relu')], num_grid, [(50, 50, 1, 'relu'), (50, 1, 1, 'id')], degree, knots)
    # print('The vc model:')
    #print(model_vc)
    model = model.cuda()
    model = torch.nn.parallel.DataParallel(model, device_ids=device_ids, dim=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=lambda_, nesterov=True)

    print('The model:')
    print(model)

    print("Dataset shape",dataset_train.shape)
    print(dataset_train)

#     if type_t==1:
#     #treatment data normalization
# #        print("Before Normarlize",dataset_train.iloc[:,0])
#         dataset_train.iloc[:,0]=dataset_train.iloc[:,0]-np.min(dataset_train.iloc[:,0])
#         dataset_train.iloc[:, 0]= dataset_train.iloc[:,0]/np.max(dataset_train.iloc[:,0])
#         print("After normarlize",dataset_train.iloc[:,0])
    training_data = Dataset_torch(torch.tensor(dataset_train.iloc[:,:gauss_params['dim']].values),
                                  torch.tensor(dataset_train['y'].values))
    test_data = Dataset_torch(torch.tensor(dataset_test.iloc[:,:gauss_params['dim']].values),
                              torch.tensor(dataset_test['y'].values))
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)


    for epoch in range(num_epoch):

        for idx, (inputs, y) in enumerate(train_dataloader):
            #TODO: change the treatment variable
            t = inputs[:, 0].float().cuda()
            x = inputs[:, 1:].float().cuda()
            y=y.float().cuda()
            optimizer.zero_grad()
            out = model.forward(t, x)
            loss,risk,imb_error,risk_t,discrepancy_loss = criterion_all(out, y,t, alpha=alpha,beta=beta,gamma_=gamma,type_y=2,type_t=1)
            loss.backward()
            optimizer.step()


        if epoch % verbose == 0:
            print('current epoch: ', epoch)
            print('training loss: ', loss.data)
            print('factual loss: ', risk.data)
            print('imb loss: ', imb_error.data)
            print('treatment loss: ', risk_t.data)
            print('discrepancy loss: ', discrepancy_loss.data)

            test_loss = test(args, model, test_dataloader, criterion_all,type_y=2,type_t=1)
            print(test_loss)

    out = model.forward(t, x)