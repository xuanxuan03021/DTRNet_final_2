from synthetic_dataset import generate_gaussians
import torch.nn as nn
import torch
# [can choose from "test","train","all"]
import mdn
from mdn import gaussian_probability
from mdn import MDN

class Truncated_power():
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        """
        #x is the treatment dim=1
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        #dim=1*5
        out = torch.zeros(x.shape[0], self.num_of_basis)
        #degree=2
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out # bs, num_of_basis


class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots
        self.islastlayer = islastlayer
        self.isbias = isbias
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)
        x_feature = x[:, 1:]
        x_treat = x[:, 0]
        #check weight here
        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T # bs, outd, d
        # print("++++++")
        # print(x_treat)
        x_treat_basis = self.spb.forward(x_treat).cuda() # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1).cuda()
        # print(x_feature_weight)
        # print(x_treat_basis_)

        # x_feature_weight * x_treat_basis; bs, outd, d
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2) # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T #bs,outd
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)

        # concat the treatment for intermediate layer
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out


class Black_box_model(nn.Module):
    def __init__(self,cfg_rep=[(6, 50, 1, 'relu'), (50, 50, 1, 'relu')]):
        super(Black_box_model,self).__init__()
        self.cfg_rep=cfg_rep
        density_blocks = []
        for layer_idx, layer_cfg in enumerate(cfg_rep):
            # fc layer
            density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())
            else:
                print('No activation')
        self.model=nn.Sequential(*density_blocks)
    def forward(self,x):
        y=self.model(x)
        return y


class iCXAI_model(nn.Module):
    def __init__(self, num_grid,degree, knots, cfg_rep=[(6, 50, 1, 'relu'), (50, 50, 1, 'relu')],
                 cfg_y = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')],
                 cfg_t=[(50, 50, 1, 'relu'), (50, 1, 1, 'id')],
                 num_gaussians=20,type_t=1):
        super(iCXAI_model, self).__init__()
        self.cfg_rep=cfg_rep
        self.cfg_y=cfg_y
        self.cfg_t=cfg_t
        self.degree = degree
        self.knots = knots
        self.type_t=type_t
        self.num_grid = num_grid
        self.num_gaussians=num_gaussians
        self.gamma_network=self.static_network(cfg_density= self.cfg_rep)
        self.delta_network=self.static_network(cfg_density= self.cfg_rep)
        self.psi_network=self.static_network(cfg_density= self.cfg_rep)
        self.y_network=self.dynamics_network(self.cfg_y)
        if type_t==1:
            self.t_network,self.t_rep= self.static_network(self.cfg_t,last_layer=True,continous_t=1)
        else:
            self.t_network,self.t_rep= self.static_network(self.cfg_t,last_layer=True,continous_t=0)


        if (2*self.cfg_rep[-1][1]) != self.cfg_y[0][0] :
            print("latent dimensions of z and the input deminsion of y prediction network does not match!")
            raise ValueError
        if (2*self.cfg_rep[-1][1]) != self.cfg_t[0][0] :
            print("latent dimensions of z and the input deminsion of t prediction network does not match!")
            raise ValueError
        if self.cfg_rep[-1][1] != self.cfg_t[-2][1] :
            print("latent dimensions of z (psi) and the last hidden layer of t prediction network does not match!")
            raise ValueError

    def static_network(self,cfg_density,last_layer=False,continous_t=0):
        #cfg_density: input dim , output dim, if bias, activation function
        density_blocks = []
        for layer_idx, layer_cfg in enumerate(cfg_density):

            # fc layer
            density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())
            else:
                print('No activation')

        if continous_t==1:
            density_blocks=density_blocks[:-2]
            Model_last_layer = nn.Sequential(*(density_blocks))
            density_blocks.append(mdn.MDN(self.cfg_t[-1][0], self.cfg_t[-1][1], self.num_gaussians))
            if last_layer:
                Model = nn.Sequential(*density_blocks)
                return Model, Model_last_layer
            else:
                Model = nn.Sequential(*density_blocks)
                return Model
        else:
            if last_layer:
                Model = nn.Sequential(*density_blocks)
                Model_last_layer = nn.Sequential(*(density_blocks[:-2]))
                return Model,Model_last_layer
            else:
                Model = nn.Sequential(*density_blocks)
                return Model

    def dynamics_network(self,cfg):      # construct the dynamics network
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0))
        blocks.append(last_layer)

        model_d= nn.Sequential(*blocks)
        return  model_d


    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            # elif isinstance(m, Density_Block):
            #     m.weight.data.normal_(0, 0.01)
            #     if m.isbias:
            #         m.bias.data.zero_()

    def forward(self,t,x):

        #TODO: check if it needs normalization
        gamma= self.gamma_network(x)
        #print("gamma",gamma[:10])
        delta= self.delta_network(x)
       # print("delta",delta[:10])

        psi= self.psi_network(x)
       # print("ppsi",psi[:10])
        #concatenate corresponding representation
        gamma_delta = torch.cat((gamma, delta), 1)
        delta_psi = torch.cat((delta, psi), 1)
        t_delta_psi= torch.cat((torch.unsqueeze(t, 1), delta_psi), 1)
        y= self.y_network(t_delta_psi)
       # print("gamma_delta",gamma_delta[:10])
        t=self.t_network(gamma_delta)
      #  print("real t",t[:10])
        t_representation=self.t_rep(gamma_delta)

        return gamma,delta,psi, y,t,t_representation




