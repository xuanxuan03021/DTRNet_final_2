import os
import numpy as np

from simu1 import simu_data1
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate simulate data')
    parser.add_argument('--save_dir', type=str, default='dataset/simu1', help='dir to save generated data')
    parser.add_argument('--num_eval', type=int, default=50, help='num of dataset for evaluating the methods')
    parser.add_argument('--num_tune', type=int, default=20, help='num of dataset for tuning the parameters')

    args = parser.parse_args()
    save_path = args.save_dir

    for _ in range(args.num_tune):
        print('generating tuning set: ', _)
        data_path = os.path.join(save_path, 'tune', str(_))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        train_matrix, test_matrix, t_grid,t_grid_mise = simu_data1(500, 200)

        data_file = os.path.join(data_path, 'train.txt')
        np.savetxt(data_file, train_matrix.numpy())
        data_file = os.path.join(data_path, 'test.txt')
        np.savetxt(data_file, test_matrix.numpy())
        data_file = os.path.join(data_path, 't_grid.txt')
        np.savetxt(data_file, t_grid.numpy())
        data_file = os.path.join(data_path, 't_grid_mise.txt')
        np.savetxt(data_file, t_grid_mise.numpy())
    for _ in range(args.num_eval):
        print('generating eval set: ', _)
        data_path = os.path.join(save_path, 'eval', str(_))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        train_matrix, test_matrix, t_grid, t_grid_mise = simu_data1(500, 200)

        data_file = os.path.join(data_path, 'train.txt')
        np.savetxt(data_file, train_matrix.numpy())
        data_file = os.path.join(data_path, 'test.txt')
        np.savetxt(data_file, test_matrix.numpy())
        data_file = os.path.join(data_path, 't_grid.txt')
        np.savetxt(data_file, t_grid.numpy())
        data_file = os.path.join(data_path, 't_grid_mise.txt')
        np.savetxt(data_file, t_grid_mise.numpy())
