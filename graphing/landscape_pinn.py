from pyDOE import lhs
import numpy as np
import scipy.io
import time
import torch
from collections import OrderedDict
import numpy as np
import torch 
from pyhessian import hessian
from pyhessian import * # get the dataset
# from pyhessian import hessian # Hessian computation
# from density_plot import get_esd_plot # ESD plot
# from pytorchcv.model_provider import get_model as ptcv_get_model # model

import matplotlib.pyplot as plt
import plotly.graph_objects as go
seed=42
np.random.seed(seed)
torch.manual_seed(seed)

device = "cuda"

# deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        self.depth = len(layers) - 1

        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)

        for module in self.layers.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)

    def forward(self, x):
        out = self.layers(x)
        return out

class PhysicsInformedNN():
    def __init__(self, X_u, u, X_f, c, all_x, all_t, layers, lb, ub):

        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # x and t data for states and collocation points
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        self.all_x = torch.tensor(all_x).float().to(device)
        self.all_t = torch.tensor(all_t).float().to(device)
        self.delta_x = 1/128
        # state data
        self.u = torch.tensor(u).float().to(device)
        # conserved quantity
        self.c = c

        self.layers = layers

        self.dnn = DNN(layers).to(device)

        # ADAM optimizer instead of LBFGS
        self.optimizer = torch.optim.Adam(
            self.dnn.parameters(),
            lr=0.001,  # Standard ADAM learning rate
            betas=(0.9, 0.999),  # Default ADAM betas
            eps=1e-8,  # Default ADAM epsilon
            weight_decay=0,  # No L2 regularization
        )

        self.iter = 0
        self.max_epochs = 10000

    def net_u(self, x, t):

        u = self.dnn(torch.cat([x, t], dim=1))

        return u


    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        # advection equation
        f = u_t + 0.25 * u_x
        return f

    def loss_func(self, X_u_train, u_train, X_f_train):
        self.optimizer.zero_grad()

        x_u = torch.tensor(X_u_train[:, 0:1], requires_grad=True).float().to(device)
        t_u = torch.tensor(X_u_train[:, 1:2], requires_grad=True).float().to(device)
        x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(device)
        t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).float().to(device)
        u = torch.tensor(u_train).float().to(device)


        u_pred = self.net_u(x_u, t_u)
        f_pred = self.net_f(x_f, t_f)
        loss_u = torch.mean((u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)

        loss = loss_u + loss_f

        # loss.backward()

        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
            )

        return loss

    def train(self):
        self.dnn.train()

        self.optimizer.step(self.loss_func)


    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        c = torch.sum(torch.reshape(u, (100, 256))*self.delta_x, dim=1)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        c = c.detach().cpu().numpy()
        return u, f, c
    

layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

N_u = 100
N_f = 5000

data = np.load('../advection_solution_new.npy', allow_pickle=True).item()

# Extract the saved arrays
x = data['x']
t = data['t']
Exact = data['u']

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]

lb = X_star.min(0)
ub = X_star.max(0)

# boundary points
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
uu1 = Exact[0:1,:].T
xx2 = np.hstack((X[:,0:1], T[:,0:1]))
uu2 = Exact[:,0:1]
xx3 = np.hstack((X[:,-1:], T[:,-1:]))
uu3 = Exact[:,-1:]

X_u_train = np.vstack([xx1, xx2, xx3])
X_f_train = lb + (ub-lb)*lhs(2, N_f)

# conserved quantity
c = np.mean(np.sum(Exact*1/128, axis=1))

X_f_train = np.vstack((X_f_train, X_u_train))
u_train = np.vstack([uu1, uu2, uu3])

# take random sample of data
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx,:]

model = PhysicsInformedNN(X_u_train, u_train, X_f_train, c, x, t, layers, lb, ub)

# Load the saved parameters
model.dnn.load_state_dict(torch.load('pinn_adam.pth'))
model.dnn.eval()


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

X_u_train_tensor = torch.tensor(X_u_train, dtype=torch.float32).to(device)
u_train_tensor = torch.tensor(u_train, dtype=torch.float32).to(device)

hessian_comp = hessian(model, data=(X_u_train_tensor, u_train_tensor, X_f_train), cuda=True)

top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=1)
print(f"The top Hessian eigenvalue of this model is {top_eigenvalues}")

def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=50)
# print(top_eigenvalues)

abs_top_eigenvalues = np.abs(np.array(top_eigenvalues))
print("condition number", np.max(abs_top_eigenvalues)/np.min(abs_top_eigenvalues))
