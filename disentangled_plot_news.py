import torch
import math
import numpy as np
import pandas as pd

import os
import json
import time
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import seaborn as sns
from based_on_vcnet import Vcnet
from data import get_iter
from based_on_vcnet_evaluation import curve

import argparse

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1314)
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    print(use_cuda)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

    device_ids = [1]

    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)
    gpu = use_cuda

    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/news', help='dir of data matrix')
    parser.add_argument('--data_split_dir', type=str, default='dataset/news/eval', help='dir of data split')
   # parser.add_argument('--data_split_dir', type=str, default='dataset/news/tune', help='dir of data split')
    parser.add_argument('--save_dir', type=str, default='logs/news/eval', help='dir to save result')
    # common
    parser.add_argument('--num_dataset', type=int, default=10, help='num of datasets to train')

    # training
    parser.add_argument('--n_epochs', type=int, default=800, help='num of epochs to train')

    # print train info
    parser.add_argument('--verbose', type=int, default=10, help='print train info freq')


    args = parser.parse_args()

    seed = 10
    torch.manual_seed(seed)
    np.random.seed(seed)
    # fixed parameter for optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9

    # targeted regularization optimizer
    tr_wd = 1e-5

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    # data
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
            cfg_density = [(498, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(100, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model_initial = Vcnet(cfg_density, num_grid, cfg, degree, knots).cuda()
            model_trained = Vcnet(cfg_density, num_grid, cfg, degree, knots).cuda()

        if model_name == 'Vcnet_disentangled':
            init_lr = 0.00001
            alpha = 0.4
            beta=0.1
            gamma=0.2
            Result['Vcnet_disentangled'] = []

        for _ in range(num_dataset):

            cur_save_path = save_path + '/' + str(_)
            if not os.path.exists(cur_save_path):
                os.makedirs(cur_save_path)

            idx_train = torch.load(args.data_split_dir + '/' + str(_) + '/idx_train.pt')
            idx_test = torch.load(args.data_split_dir + '/' + str(_) + '/idx_test.pt')

            train_matrix = data_matrix[idx_train, :]
            test_matrix = data_matrix[idx_test, :]
            t_grid = t_grid_all[:, idx_test]

            train_loader = get_iter(data_matrix[idx_train, :], batch_size=500, shuffle=True)
            test_loader = get_iter(data_matrix[idx_test, :], batch_size=data_matrix[idx_test, :].shape[0],
                                   shuffle=False)
            # reinitialize model
            model_initial._initialize_weights()
            model_trained._initialize_weights()

            # to load
            checkpoint = torch.load('logs/news/eval/' + str(_) + '/Vcnet_disentangled_ckpt.pth.tar')
            model_trained.load_state_dict(checkpoint['model_state_dict'])


            checkpoint = torch.load('logs/news/eval/' + str(_) + '/Vcnet_disentangledno_beta_ckpt.pth.tar')
            model_initial.load_state_dict(checkpoint['model_state_dict'])

            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0].cuda()
                x = inputs[:, 1:].cuda()
                break
            g, Q, gamma, delta, psi, g_psi = model_initial.forward(t, x)
            g_trained, Q_trained, gamma_trained, delta_trained, psi_trained, g_psi_trained = model_trained.forward(t, x)

            gamma = pd.DataFrame(gamma.cpu().detach().numpy())
            gamma["type"] = "gamma"
            delta = pd.DataFrame(delta.cpu().detach().numpy())
            delta["type"] = "delta"
            psi = pd.DataFrame(psi.cpu().detach().numpy())
            psi["type"] = "upsilon"
            embeddings_all = pd.concat([gamma, delta, psi], axis=0)
            print(embeddings_all)

            gamma_trained = pd.DataFrame(gamma_trained.cpu().detach().numpy())
            gamma_trained["type"] = "gamma"
            delta_trained = pd.DataFrame(delta_trained.cpu().detach().numpy())
            delta_trained["type"] = "delta"
            psi_trained = pd.DataFrame(psi_trained.cpu().detach().numpy())
            psi_trained["type"] = "upsilon"
            embeddings_all_trained = pd.concat([gamma_trained, delta_trained, psi_trained], axis=0)
            print(embeddings_all_trained)

            tsne = TSNE(n_components=2, n_iter=5000, perplexity=5, verbose=1, random_state=123)
            z = tsne.fit_transform(embeddings_all.iloc[:, :50])
            tsne_initial = pd.DataFrame()
            tsne_initial["type"] = embeddings_all["type"]
            tsne_initial["comp-1"] = z[:, 0]
            tsne_initial["comp-2"] = z[:, 1]

            fig, axs = plt.subplots(figsize=(6, 6))

            tsne = TSNE(n_components=2, n_iter=5000, perplexity=5, verbose=1, random_state=123)
            z_trained = tsne.fit_transform(embeddings_all_trained.iloc[:, :50])
            tsne_train = pd.DataFrame()
            tsne_train["type"] = embeddings_all_trained["type"]
            tsne_train["comp-1"] = z_trained[:, 0]
            tsne_train["comp-2"] = z_trained[:, 1]

            sns.scatterplot(x="comp-1", y="comp-2", hue=tsne_initial.type.tolist(),
                            palette=sns.color_palette("hls", 3),
                            data=tsne_initial).set(title="Deep Representations of model with no beta T-SNE projection")
            axs.set_ylim(-200, 200)
            axs.set_xlim(-200, 200)

            fig.savefig(cur_save_path +"initial_disentangled_no_beta_fig_news_" + str(_) + ".png")

            fig_trained, axs_trained = plt.subplots(figsize=(6, 6))

            sns.scatterplot(x="comp-1", y="comp-2", hue=tsne_train.type.tolist(),
                            palette=sns.color_palette("hls", 3),
                            data=tsne_train).set(title="Deep Representations of trained model T-SNE projection")
            axs_trained.set_ylim(-200, 200)
            axs_trained.set_xlim(-200, 200)

            fig_trained.savefig(cur_save_path +"trained_disebtangled_trained_fig_news_" + str(_) + ".png")