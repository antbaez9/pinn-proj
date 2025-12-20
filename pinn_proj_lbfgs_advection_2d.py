

# ## Libraries and Dependencies
import argparse

parser = argparse.ArgumentParser(description='Physics-Informed Neural Network with optional projection layer')
parser.add_argument('--eq', type=str, help='Equation type')
parser.add_argument('--quantity', type=str, help='Equation type')
parser.add_argument('--proj', action='store_true', 
                    help='Enable projection layer (default: disabled)')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='Epsilon value (default: 1e-6)')
parser.add_argument('--data', type=str,
                    help='Epsilon value (default: 1e-6)')
args = parser.parse_args()


import sys
sys.path.insert(0, '../Utilities/')

import torch
from collections import OrderedDict
from pyDOE import lhs
import numpy as np
import scipy.io
import time
import warnings
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")

import os
seed = int(os.environ.get('SEED'))
torch.manual_seed(seed)
np.random.seed(seed)

# np.random.seed(seed)
# torch.manual_seed(seed)

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# ## Physics-informed Neural Networks

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
    def __init__(self, X_u, u, X_f, c_momentum, c_energy, all_x, all_y, all_t, layers, lb, ub, nu):

        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # x and t data for states and collocation points
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.y_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 2:3], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.y_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 2:3], requires_grad=True).float().to(device)
        self.all_x = torch.tensor(all_x).float().to(device)
        self.all_y = torch.tensor(all_y).float().to(device)
        self.all_t = torch.tensor(all_t).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.delta_xy = 1/16
        # conserved quantity
        self.c_momentum = torch.tensor(c_momentum).to(device)
        self.c_energy = torch.tensor(c_energy).to(device)

        self.layers = layers
        self.nu = nu

        self.dnn = DNN(layers).to(device)

        # ADAM optimizer instead of LBFGS
        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=args.eps,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )

        self.iter = 0
        self.max_epochs = 50000 

    def net_u(self, x, y, t):

        u = self.dnn(torch.cat([x, y, t], dim=1))

        if not args.proj:
            return u
        else:
            if args.quantity == "momentum":

                volume_xy = 4
                mesh_t, mesh_x, mesh_y = torch.meshgrid([t.squeeze(1).detach().to(device), self.all_x.detach().to(device), self.all_y.detach().to(device)], indexing='ij')
                t_by_xy = torch.concat((mesh_x.unsqueeze(3), mesh_y.unsqueeze(3), mesh_t.unsqueeze(3)), dim=-1)

                # use integral on mesh to find current momentum of system
                if t.shape[0] == 100:
                    integral_u_dx = torch.sum(self.dnn(t_by_xy)*self.delta_xy**2, dim=(1,2)).to(device)

                else:
                    integral_u_dx = torch.zeros((t.shape[0], 1), dtype=torch.float32, device=t.device)
                    n_batches = 5
                    batch_size = int(t.shape[0]/n_batches)
                    for i in range(0, t.shape[0], batch_size):
                        end_idx = min(i + batch_size, t.shape[0])
                        integrand = self.dnn(t_by_xy[i:end_idx, :, :])
                        integral_u_dx[i:end_idx, :] = torch.sum(integrand*self.delta_xy**2, dim=(1,2))

                second_term = integral_u_dx / volume_xy

                # create tensor of conserved quantity values
                c_tensor = torch.full(x.shape, self.c_momentum).to(device)
                third_term = c_tensor / volume_xy

                # return result of projection
                return u - second_term + third_term

            elif args.quantity == "energy":
                mesh_t, mesh_x, mesh_y = torch.meshgrid([t.squeeze(1).detach().to(device), self.all_x.detach().to(device), self.all_y.detach().to(device)], indexing='ij')
                t_by_xy = torch.concat((mesh_x.unsqueeze(3), mesh_y.unsqueeze(3), mesh_t.unsqueeze(3)), dim=-1)

                # use integral on mesh to find current momentum of system
                if t.shape[0] == 100:
                    integral_u_dx = torch.sum(self.dnn(t_by_xy)**2*self.delta_xy**2, dim=(1,2)).to(device)
                else:
                    integral_u_dx = torch.zeros((t.shape[0], 1), dtype=torch.float32, device=t.device)
                    n_batches = 5
                    batch_size = int(t.shape[0]/n_batches)
                    for i in range(0, t.shape[0], batch_size):
                        end_idx = min(i + batch_size, t.shape[0])
                        integrand = self.dnn(t_by_xy[i:end_idx, :, :])
                        integral_u_dx[i:end_idx, :] = torch.sum(integrand**2*self.delta_xy**2, dim=(1,2))

                projection = torch.sqrt(torch.tensor(self.c_energy)/integral_u_dx).to(device)
                projected_u = projection * u

                # return u
                return projected_u
            elif args.quantity == "both":
                volume_xy = 4
                delta_xy = 1/16
                n = 32

                mesh_t, mesh_x, mesh_y = torch.meshgrid([t.squeeze(1).detach().to(device), self.all_x.detach().to(device), self.all_y.detach().to(device)], indexing='ij')
                t_by_xy = torch.concat((mesh_x.unsqueeze(3), mesh_y.unsqueeze(3), mesh_t.unsqueeze(3)), dim=-1)

                # momentum
                if t.shape[0] == 100:
                    integral_u_dx = torch.sum(self.dnn(t_by_xy)*self.delta_xy**2, dim=(1,2)).to(device)
                else:
                    integral_u_dx = torch.zeros((t.shape[0], 1), dtype=torch.float32, device=t.device)
                    n_batches = 5
                    batch_size = int(t.shape[0]/n_batches)
                    for i in range(0, t.shape[0], batch_size):
                        end_idx = min(i + batch_size, t.shape[0])
                        integrand = self.dnn(t_by_xy[i:end_idx, :, :])
                        integral_u_dx[i:end_idx, :] = torch.sum(integrand*self.delta_xy**2, dim=(1,2))
                second_term = integral_u_dx / volume_xy

                c_momentum_tensor = torch.full(x.shape, self.c_momentum).to(device)
                c_energy_tensor = torch.full(x.shape, self.c_energy).to(device)

                second_term_expanded = second_term.unsqueeze(1).unsqueeze(1).repeat(1, n, n, 1)

                # energy
                if t.shape[0] == 100:
                    integral_u_dx_2 = torch.sum((self.dnn(t_by_xy)-second_term_expanded)**2*self.delta_xy**2, dim=(1,2)).to(device)
                else:
                    integral_u_dx_2 = torch.zeros((t.shape[0], 1), dtype=torch.float32, device=t.device)
                    n_batches = 5
                    batch_size = int(t.shape[0]/n_batches)
                    for i in range(0, t.shape[0], batch_size):
                        end_idx = min(i + batch_size, t.shape[0])
                        integrand_2 = self.dnn(t_by_xy[i:end_idx, :, :, :])-second_term_expanded[i:end_idx, :, :, :]
                        integral_u_dx_2[i:end_idx, :] = torch.sum(integrand_2**2, dim=(1,2))

                projected_u = (u - second_term)*torch.sqrt((c_energy_tensor/delta_xy**2 - (c_momentum_tensor**2)/(n**2*delta_xy**4))/integral_u_dx_2)+c_momentum_tensor/volume_xy

                return projected_u


    def net_f(self, x, y, t):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, y, t)
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
        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        # advection equation
        f = u_t + 0.25 * (u_x + u_y)
        return f
       

    def loss_func(self):
        self.optimizer.zero_grad()

        u_pred = self.net_u(self.x_u, self.y_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.y_f, self.t_f)
        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)

        loss = loss_u + loss_f

        loss.backward()

        self.iter += 1
        #print('iter', self.iter, 'loss', loss.item())
        # if self.iter % 100 == 0:
        #     print(
        #         'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
        #     )
        return loss

    def train(self):
        self.dnn.train()

        # Backward and optimize
        self.optimizer.step(self.loss_func)

    def predict(self, X):

        if args.quantity == "momentum":
            x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
            y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
            t = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)

            self.dnn.eval()
            u = self.net_u(x, y, t)
            f = self.net_f(x, y, t)
            c = torch.sum(torch.reshape(u, (10, 32, 32))*self.delta_xy**2, dim=(1, 2))
            u = u.detach().cpu().numpy()
            f = f.detach().cpu().numpy()
            c = c.detach().cpu().numpy()
            return u, f, c
        
        elif args.quantity == "energy":
            x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
            y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
            t = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)

            self.dnn.eval()
            u = self.net_u(x, y, t)
            f = self.net_f(x, y, t)
            c = torch.sum(torch.reshape(u, (10, 32, 32))**2*self.delta_xy**2, dim=(1, 2))
            u = u.detach().cpu().numpy()
            f = f.detach().cpu().numpy()
            c = c.detach().cpu().numpy()
            return u, f, c
    
        elif args.quantity == "both":

            x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
            y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
            t = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)

            self.dnn.eval()
            u = self.net_u(x, y, t)
            c_momentum = torch.sum(torch.reshape(u, (5, 32, 32))*self.delta_xy**2, dim=(1,2))
            c_energy = torch.sum(torch.reshape(u**2, (5, 32, 32))*self.delta_xy**2, dim=(1,2))

            f = self.net_f(x, y, t)
            u = u.detach().cpu().numpy()
            f = f.detach().cpu().numpy()
            c_momentum  = c_momentum.detach().cpu().numpy()
            c_energy = c_energy.detach().cpu().numpy()

            return u, f, c_momentum, c_energy
    
        else:
            raise ValueError(f"Unsupported equation type: '{args.quantity}'.")


# ## Configurations

nu = 0.1
noise = 0.0

N_u = 1000
N_f = 10000
layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]

data = np.load(f'data/{args.data}', allow_pickle=True).item()

# Extract the saved arrays
x = data['x']
y = data['y']
t = data['t']
Exact = data['u']

delta_xy = 1/16

# print(Exact.shape)
X, T, Y = np.meshgrid(x, t, y)

X_star = np.hstack((X.flatten()[:,None], Y.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]

lb = X_star.min(0)
ub = X_star.max(0)

# boundary points
# t = t_min
xx1 = np.concatenate((X[0:1,:,:].T, Y[0:1,:,:].T, T[0:1,:,:].T), axis=2)
uu1 = Exact[0:1,:,:].T
# x0
xx2 = np.transpose(np.concatenate((X[:,0:1,:], Y[:,0:1,:], T[:,0:1,:]), axis=1), axes=(0, 2, 1))
uu2 = np.transpose(Exact[:,0:1,:], axes=(0, 2, 1))
# y0
xx3 = np.concatenate((X[:,:,0:1], Y[:,:,0:1], T[:,:,0:1]), axis=2)
uu3 = Exact[:,:,0:1]
# x_max
xx4 = np.transpose(np.concatenate((X[:,-1:,:], Y[:,-1:,:], T[:,-1:,:]), axis=1), axes=(0, 2, 1))
uu4 = np.transpose(Exact[:,-1:,:], axes=(0, 2, 1))
# y_max
xx5 = np.concatenate((X[:,:,-1:], Y[:,:,-1:], T[:,:,-1:]), axis=2)
uu5 = Exact[:,:,-1:]

# X_u_train = np.vstack([xx1, xx2, xx3, xx4, xx5]).reshape(464*64,3)
X_u_train = np.vstack([xx1, xx2, xx3, xx4, xx5]).reshape(432*32,3)
X_f_train = lb + (ub-lb)*lhs(3, N_f)

# conserved quantity
c_momentum = np.mean(np.sum((Exact)*delta_xy**2, axis=(1,2)))
c_energy = np.mean(np.sum((Exact)**2*delta_xy**2, axis=(1,2)))

# print(c_energy)

# X_f_train = np.vstack((X_f_train, X_u_train[::4]))
# u_train = np.vstack([uu1, uu2, uu3, uu4, uu5]).reshape(464*64,1)
u_train = np.vstack([uu1, uu2, uu3, uu4, uu5]).reshape(432*32,1)

# print(X_f_train.shape)

# take random sample of data
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx, :]

# ## Training

# In[8]:


model = PhysicsInformedNN(X_u_train, u_train, X_f_train, c_momentum, c_energy, x, y, t, layers, lb, ub, nu)

start = time.time()

model.train()

end = time.time()


# torch.save(model.dnn.state_dict(), 'graphing/pinn_proj_adam.pth')
# torch.save(model.dnn.state_dict(), 'graphing/pinn_adam.pth')
# print("Model saved")

print(end-start)
print(model.iter)







if args.quantity == "momentum" or args.quantity == "energy":
    u_pred = np.zeros(u_star.shape)
    c_pred = np.zeros((100,))

    for i in range(0, 100, 10): 
        X_star_batch = np.hstack((X[i:i+10, :, :].flatten()[:,None], Y[i:i+10, :, :].flatten()[:,None], T[i:i+10, :, :].flatten()[:,None]))
        u_pred_batch, f_pred_batch, c_pred_batch = model.predict(X_star_batch)

        u_pred[1024*i:1024*(i+10), :] = u_pred_batch
        c_pred[i:i+10] = c_pred_batch

elif args.quantity == "both":
    u_pred = np.zeros(u_star.shape)
    c_momentum_pred = np.zeros((100,))
    c_energy_pred = np.zeros((100,))

    batch_size = 5
    for i in range(0, 100, batch_size): 
        X_star_batch = np.hstack((X[i:i+batch_size, :, :].flatten()[:,None], Y[i:i+batch_size, :, :].flatten()[:,None], T[i:i+batch_size, :, :].flatten()[:,None]))
        u_pred_batch, f_pred_batch, c_momentum_pred_batch, c_energy_pred_batch = model.predict(X_star_batch)

        u_pred[1024*i:1024*(i+batch_size), :] = u_pred_batch
        c_momentum_pred[i:i+batch_size] = c_momentum_pred_batch
        c_energy_pred[i:i+batch_size] = c_energy_pred_batch

    # error_c_momentum = np.sum(abs(c_momentum_pred-c_momentum))
    # error_c_energy = np.sum(abs(c_energy_pred-c_energy))

error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

print('%e' % (error_u))

if args.quantity == "momentum":
    error_c = np.sum(abs(c_pred-c_momentum))
    print('%e' % (error_c))

elif args.quantity == "energy":
    error_c = np.sum(abs(c_pred-c_energy))
    print('%e' % (error_c))

elif args.quantity == "both":
    error_c_momentum = np.sum(abs(c_momentum_pred-c_momentum))
    error_c_energy = np.sum(abs(c_energy_pred-c_energy))
    print('%e' % (error_c_momentum))
    print('%e' % (error_c_energy))


import csv
import os
import fcntl  # For file locking on Unix systems
from pathlib import Path

# Add this code at the end of each Python file
# ============================================

# Configuration
csv_filename = 'results.csv'
if args.proj:
    file_identifier = f"{args.eq}_{args.quantity}_proj"  # Gets filename without .py extension
else:
    file_identifier = f"{args.eq}_{args.quantity}_pinn"  # Gets filename without .py extension

# The error values you want to save (modify these variable names as needed)
if args.quantity == "momentum" or args.quantity == "energy":
    error_values = [error_u, error_c]  
elif args.quantity == "both":
    error_values = [error_u, error_c_momentum, error_c_energy]  
time_values = [end-start, model.iter]

# Prepare the row to write
row_data = [file_identifier] + [f'{val:.6e}' for val in error_values] + [f'{val}' for val in time_values]

# Thread-safe CSV writing with file locking
def write_to_csv_safe(filename, row):
    """Write a row to CSV with file locking to prevent corruption from concurrent writes"""
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        # Lock the file to prevent concurrent writes
        fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
        
        writer = csv.writer(csvfile)
        
        # Write header if file is new/empty
        if not file_exists or os.path.getsize(filename) == 0:
            header = ['File', 'Error u', 'Error c', 'Time', 'Epochs']
            writer.writerow(header)
        
        # Write the data row
        writer.writerow(row)
        
        # File is automatically unlocked when context exits

# Write results to CSV
try:
    write_to_csv_safe(csv_filename, row_data)
except Exception as e:
    print(f"Error writing to CSV: {e}")
    # Fallback: write to a backup file