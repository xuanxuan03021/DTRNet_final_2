import torch.nn as nn
import torch
import os
import argparse
from model import Black_box_model
from OpenXAI.openxai.dataloader import return_loaders
from OpenXAI.openxai.dataloader import return_loaders
from openxai.Explainer import Explainer

import numpy as np
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


def criterion_all(y_pred, y,type_y=1):
    #return ((out[1].squeeze() - y.squeeze())**2).mean() - alpha * torch.log(out[0] + epsilon).mean()

    ''' Compute sample reweighting (general propensity score)'''
    ''' Construct factual loss function '''

    if type_y == 1:
        risk= (y_pred - y).pow(2).mean()
    elif type_y == 2:
        criterion = nn.BCELoss().cuda()
        risk =criterion(y_pred, y.unsqueeze(dim=1))
    elif type_y == 3:
        criterion = nn.CrossEntropyLoss().cuda()
        risk = criterion(y_pred, y)
    return risk


def test(args, model, test_dataset, criterion,type_y=2):
    model.eval()
    loss = 0
    total_samples = 0
    for idx, (inputs, y,weight,mask,masked_weight,pro,cluster_index) in enumerate(test_dataset):
        x = inputs.float().cuda()
        y = y.float().cuda()
        y_pred = model.forward(x)
        loss_batch = criterion(y_pred, y,type_y=type_y)
        n_samples = inputs.shape[0]
        loss += loss_batch * n_samples
        total_samples += n_samples
    loss_final_average = loss / total_samples
    return loss_final_average


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train black box model')

    # training
    parser.add_argument('--n_epochs', type=int, default=100, help='num of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size to train')
    parser.add_argument('--shuffle', type=bool, default=True, help='if shuffle the dataset')

    # print train info
    parser.add_argument('--verbose', type=int, default=10, help='print train info freq')

    args = parser.parse_args()
    num_epoch = args.n_epochs
    batch_size=args.batch_size
    shuffle=args.shuffle
    verbose=args.verbose

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
    cfg_rep = [(20, 100, 1, 'relu'),(100, 100, 1, 'relu'), (100, 1, 1, 'sigmoid')]

    init_lr = 0.001
    # alpha = 0.5
    lambda_= 5e-3
    momentum = 0.9


    loader_train, loader_test = return_loaders(data_name="synthetic", batch_size=5,gauss_params=gauss_params)
    model = Black_box_model(cfg_rep=cfg_rep)
    data_iter = iter(loader_test)
    data = next(data_iter)
    print(data)
    input()
    labels = labels.type(torch.int64)


    model = model.cuda()
    #model = torch.nn.parallel.DataParallel(model, device_ids=device_ids, dim=0)

    print('The model:')
    print(model)
    print(model.parameters())

    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=lambda_, nesterov=True)

    for epoch in range(num_epoch):

        for idx, (inputs, y,weight,mask,masked_weight,pro,cluster_index) in enumerate(loader_train):
            x = inputs.float().cuda()
            y=y.float().cuda()
            optimizer.zero_grad()
            y_pred = model.forward(x)
            loss =criterion_all(y_pred, y,type_y=2)
            loss.backward()
            optimizer.step()

        if epoch % verbose == 0:
            print('current epoch: ', epoch)
            print('training loss: ', loss.data)
            test_loss = test(args, model, loader_test, criterion_all,type_y=2)
            print(test_loss)
    # Hyperparameters for Lime
    lime_mode = 'tabular'
    lime_sample_around_instance = True
    lime_kernel_width = 0.75
    lime_n_samples = 1000
    lime_discretize_continuous = False
    lime_standard_deviation = float(np.sqrt(0.03))

    data_all = torch.FloatTensor(loader_train.dataset.data)
    # You can supply your own set of hyperparameters like so:
    param_dict_lime = dict()
    param_dict_lime['dataset_tensor'] = data_all
    param_dict_lime['std'] = lime_standard_deviation
    param_dict_lime['mode'] = lime_mode
    param_dict_lime['sample_around_instance'] = lime_sample_around_instance
    param_dict_lime['kernel_width'] = lime_kernel_width
    param_dict_lime['n_samples'] = lime_n_samples
    param_dict_lime['discretize_continuous'] = lime_discretize_continuous
    lime = Explainer(method='lime',
                     model=model,
                     dataset_tensor=data_all,
                     param_dict_lime=param_dict_lime)

    lime_custom = lime.get_explanation(inputs_batch,
                                       label=labels)
    print(lime_custom)