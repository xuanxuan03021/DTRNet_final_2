import torch
import numpy as np
import json
from data import get_iter

# def curve(model, test_matrix, t_grid, targetreg=None):
#     n_test = t_grid.shape[1]
#     t_grid_hat = torch.zeros(2, n_test)
#     t_grid_hat[0, :] = t_grid[0, :]
#
#     test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)
#
#     if targetreg is None:
#         for _ in range(n_test):
#             for idx, (inputs, y) in enumerate(test_loader):
#                 t = inputs[:, 0].cuda()
#                 t *= 0
#                 t += t_grid[0, _]
#                 x = inputs[:, 1:].cuda()
#                 break
#             out = model.forward(t, x)
#             out = out[1].data.squeeze()
#             out = out.mean()
#             t_grid_hat[1, _] = out
#         mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
#         return t_grid_hat, mse
#     else:
#         for _ in range(n_test):
#             for idx, (inputs, y) in enumerate(test_loader):
#                 t = inputs[:, 0]
#                 t *= 0
#                 t += t_grid[0, _]
#                 x = inputs[:, 1:]
#                 break
#             out = model.forward(t, x)
#             tr_out = targetreg(t).data
#             g = out[0].data.squeeze()
#             out = out[1].data.squeeze() + tr_out / (g + 1e-6)
#             out = out.mean()
#             t_grid_hat[1, _] = out
#         mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
#         return t_grid_hat, mse

import torch
import numpy as np
import json
from data import get_iter
epsilon=1e-6
def curve(model, test_matrix, t_grid,t_grid_mise, targetreg=None):
    n_test = t_grid.shape[1]
    t_grid_hat = torch.zeros(2, n_test)
    t_grid_hat[0, :] = t_grid[0, :]
    t_grid_mise_hat = torch.zeros(n_test, n_test)

    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    if targetreg is None:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0].cuda()
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:].cuda()
                break
            out = model.forward(t, x)
            out = out[1].data.squeeze()
            t_grid_mise_hat[:,_]=out
            out = out.mean()
            t_grid_hat[1, _] = out
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        print(t_grid_mise_hat.shape,t_grid_mise.shape)
        mise=np.square(t_grid_mise_hat-t_grid_mise).mean().data

        return t_grid_hat, mse,mise
    else:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(t, x)
            tr_out = targetreg(t).data
            g = out[0].data.squeeze()
            out = out[1].data.squeeze() + tr_out / (g + 1e-6)
            t_grid_mise_hat[:,_]=out
            out = out.mean()
            t_grid_hat[1, _] = out
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        mise=np.square(t_grid_mise_hat-t_grid_mise).mean().data

        return t_grid_hat, mse, mise