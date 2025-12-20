

# ## Libraries and Dependencies
import argparse

parser = argparse.ArgumentParser(description='Physics-Informed Neural Network with optional projection layer')
parser.add_argument('--eq', type=str, help='Equation type')
parser.add_argument('--quantity', type=str, help='Equation type')
parser.add_argument('--eps', type=float, default=1e-8,
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

sc_weight = int(os.environ.get('SC_WEIGHT', 10))



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
    def __init__(self, X_u, u, X_f, c_momentum, c_energy, all_x, all_t, layers, lb, ub, nu):

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
        self.u = torch.tensor(u).float().to(device).detach()
        # conserved quantity
        self.c_momentum = torch.tensor(c_momentum).to(device)
        self.c_energy = torch.tensor(c_energy).to(device)

        if args.eq == "react-diff":
            if args.quantity == "momentum" or args.quantity == "both":
                grad_c_momentum = (c_momentum[2:] - c_momentum[:-2]) / (2*0.01)
                first_point = (-3*c_momentum[0] + 4*c_momentum[1] - c_momentum[2]) / (2*0.01)      
                last_point = (3*c_momentum[-1] - 4*c_momentum[-2] + c_momentum[-3]) / (2*0.01)     
                grad_c_momentum = np.concatenate([[first_point], grad_c_momentum, [last_point]])        
                self.grad_c_momentum = torch.tensor(grad_c_momentum).float().to(device)
            if args.quantity == "energy" or args.quantity == "both":
                grad_c_energy = (c_energy[2:] - c_energy[:-2]) / (2*0.01)
                first_point = (-3*c_energy[0] + 4*c_energy[1] - c_energy[2]) / (2*0.01)      
                last_point = (3*c_energy[-1] - 4*c_energy[-2] + c_energy[-3]) / (2*0.01)     
                grad_c_energy = np.concatenate([[first_point], grad_c_energy, [last_point]])        
                self.grad_c_energy = torch.tensor(grad_c_energy).float().to(device)



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

    def net_u(self, x, t):

        u = self.dnn(torch.cat([x, t], dim=1))

        if args.quantity == "momentum":

            volume_x = 2
            # form mesh containing passed in t values and all x values 
            mesh_t, mesh_x = torch.meshgrid([t.squeeze(1).detach(), self.all_x.detach()], indexing='ij')
            t_by_x = torch.concat((mesh_x.unsqueeze(2), mesh_t.unsqueeze(2)), dim=-1)

            # use integral on mesh to find current momentum of system
            if t.shape[0] == 100:
                integral_u_dx = torch.sum(self.dnn(t_by_x)*self.delta_x, dim=1)
            else:
                integral_u_dx = []
                n_batches = 5
                batch_size = int(t.shape[0]/n_batches)
                for i in range(0, t.shape[0], batch_size):
                    integral_part = torch.sum(self.dnn(t_by_x[i:i+batch_size, :, :])*self.delta_x, dim=1)
                    integral_u_dx.append(integral_part)

                integral_u_dx = torch.cat(integral_u_dx, dim=0)

            second_term = integral_u_dx / volume_x


            if args.eq == "react-diff":
                # find index of closest t value
                t_index_closest = torch.argmin(torch.abs(t-self.all_t.unsqueeze(0)), dim=1).to(device)

                # approximate c value
                delta_t = t - self.all_t.unsqueeze(1)[t_index_closest]
                approx_c = self.c_momentum[t_index_closest] + self.grad_c_momentum[t_index_closest] * delta_t.squeeze(1)

                c_tensor = approx_c.unsqueeze(1).to(device)
                third_term = c_tensor / volume_x
            else:
                # create tensor of conserved quantity values
                c_tensor = torch.full(x.shape, self.c_momentum).to(device)
                third_term = c_tensor / volume_x

            # return result of projection
            loss_c = torch.mean((second_term - third_term) ** 2).to(device)
            return u, loss_c

        elif args.quantity == "energy":
            volume_x = 2
            # form mesh containing passed in t values and all x values 
            mesh_t, mesh_x = torch.meshgrid([t.squeeze(1).detach(), self.all_x.detach()], indexing='ij')
            t_by_x = torch.concat((mesh_x.unsqueeze(2), mesh_t.unsqueeze(2)), dim=-1)

            # use integral on mesh to find current momentum of system
            if t.shape[0] == 100:
                integral_u_dx = torch.sum(self.dnn(t_by_x)**2*self.delta_x, dim=1).to(device)
            else:
                integral_u_dx = []
                n_batches = 5
                batch_size = int(t.shape[0]/n_batches)
                for i in range(0, t.shape[0], batch_size):
                    integral_part = torch.sum(self.dnn(t_by_x[i:i+batch_size, :, :])**2*self.delta_x, dim=1)
                    integral_u_dx.append(integral_part)

                integral_u_dx = torch.cat(integral_u_dx, dim=0)

            second_term = integral_u_dx / volume_x

            if args.eq == "react-diff":
                # find index of closest t value
                t_index_closest = torch.argmin(torch.abs(t-self.all_t.unsqueeze(0)), dim=1).to(device)

                # approximate c value
                delta_t = t - self.all_t.unsqueeze(1)[t_index_closest]
                approx_c = self.c_energy[t_index_closest] + self.grad_c_energy[t_index_closest] * delta_t.squeeze(1)

                c_tensor = approx_c.unsqueeze(1).to(device)
                third_term = c_tensor / volume_x
            else:
                # create tensor of conserved quantity values
                c_tensor = torch.full(x.shape, self.c_energy).to(device)
                third_term = c_tensor / volume_x

            # return result of projection
            loss_c = torch.mean((second_term - third_term) ** 2).to(device)
            return u, loss_c



        elif args.quantity == "both":

            volume_x = 2
            delta_x = 1/128

            mesh_t, mesh_x = torch.meshgrid([t.squeeze(1).to(device).detach(), self.all_x.to(device).detach()], indexing='ij')
            t_by_x = torch.concat((mesh_x.unsqueeze(2), mesh_t.unsqueeze(2)), dim=-1)

            # use integral on mesh to find current momentum of system
            if t.shape[0] == 100:
                integral_u_dx = torch.sum(self.dnn(t_by_x)*self.delta_x, dim=1)
            else:
                integral_u_dx = []
                n_batches = 5
                batch_size = int(t.shape[0]/n_batches)
                for i in range(0, t.shape[0], batch_size):
                    integral_part = torch.sum(self.dnn(t_by_x[i:i+batch_size, :, :])*self.delta_x, dim=1)
                    integral_u_dx.append(integral_part)

                integral_u_dx = torch.cat(integral_u_dx, dim=0)
            second_term_momentum = integral_u_dx / volume_x

            # use integral on mesh to find current momentum of system
            if t.shape[0] == 100:
                integral_u_dx_2 = torch.sum(self.dnn(t_by_x)**2*self.delta_x, dim=1)
            else:
                integral_u_dx_2 = []
                n_batches = 5
                batch_size = int(t.shape[0]/n_batches)
                for i in range(0, t.shape[0], batch_size):
                    integral_part = torch.sum(self.dnn(t_by_x[i:i+batch_size, :, :])**2*self.delta_x, dim=1)
                    integral_u_dx_2.append(integral_part)

                integral_u_dx_2 = torch.cat(integral_u_dx_2, dim=0)
            second_term_energy = integral_u_dx_2 / volume_x


            if args.eq == "react-diff":
                # find index of closest t value
                t_index_closest = torch.argmin(torch.abs(t-self.all_t.unsqueeze(0)), dim=1).to(device)

                # approximate c value
                delta_t = t - self.all_t.unsqueeze(1)[t_index_closest]

                nearest_c_momentum = self.c_momentum[t_index_closest]
                nearest_c_energy = self.c_energy[t_index_closest]

                c_momentum_tensor = (nearest_c_momentum + self.grad_c_momentum[t_index_closest] * delta_t.squeeze(1))/volume_x
                c_energy_tensor = (nearest_c_energy + self.grad_c_energy[t_index_closest] * delta_t.squeeze(1))/volume_x

                loss_c = (torch.mean((second_term_momentum - c_momentum_tensor.unsqueeze(1)) ** 2) + torch.mean((second_term_energy - c_energy_tensor.unsqueeze(1)) ** 2)).to(device)
                return u, loss_c
            else:

                c_momentum_tensor = (torch.full(x.shape, self.c_momentum)/volume_x).to(device)
                c_energy_tensor = (torch.full(x.shape, self.c_energy)/volume_x).to(device)

                loss_c = (torch.mean((second_term_momentum - c_momentum_tensor) ** 2) + torch.mean((second_term_energy - c_energy_tensor) ** 2)).to(device)
                return u, loss_c


    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, t)[0]

        if args.eq == "advection":
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

            # advection equation
            f = u_t + 0.25 * u_x
            return f
        elif args.eq == "kdv":
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
            u_xxx = torch.autograd.grad(
                u_xx, x,
                grad_outputs=torch.ones_like(u_x),
                retain_graph=True,
                create_graph=True
            )[0]

            # kdv equation
            f = u_t + u * u_x + 0.0025 * u_xxx
            return f
        elif args.eq == "wave":
            u_t = torch.autograd.grad(
                u, t,
                grad_outputs=torch.ones_like(u),
                retain_graph=True,
                create_graph=True
            )[0]
            u_tt = torch.autograd.grad(
                u_t, t,
                grad_outputs=torch.ones_like(u_t),
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

            # wave equation
            f = u_tt - 0.25**2 * u_xx
            return f
        elif args.eq == "react-diff":
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

            # reaction-diffusion equation
            f = u_t - 0.1 * u_xx - 0.5 * u
            return f
        else:
            raise ValueError(f"Unsupported equation type: '{args.eq}'.")

    def loss_func(self):
        self.optimizer.zero_grad()

        u_pred, loss_c = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)

        # if self.iter % 100 == 0:
        #     print(loss_u/loss_c)

        # if args.eq == "kdv":
        #     loss = loss_u + loss_f + 100*loss_c
        # else:
        #     loss = loss_u + loss_f + 10*loss_c        #     
        # 
        loss = loss_u + loss_f + sc_weight*loss_c

        loss.backward()

        self.iter += 1
        # if self.iter % 100 == 0:
        #     print(
        #         'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
        #     )
            
        return loss

    def train(self):
        self.dnn.train()

        self.optimizer.step(self.loss_func)


    def predict(self, X):

        if args.quantity == "momentum":
            x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
            t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

            self.dnn.eval()
            u = self.net_u(x, t)[0]
            f = self.net_f(x, t)
            c = torch.sum(torch.reshape(u, (100, 256))*self.delta_x, dim=1)
            u = u.detach().cpu().numpy()
            f = f.detach().cpu().numpy()
            c = c.detach().cpu().numpy()
            return u, f, c
    
        elif args.quantity == "energy":
            x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
            t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

            self.dnn.eval()
            u = self.net_u(x, t)[0]
            f = self.net_f(x, t)
            c = torch.sum(torch.reshape(u**2, (100, 256))*self.delta_x, dim=1)
            u = u.detach().cpu().numpy()
            f = f.detach().cpu().numpy()
            c = c.detach().cpu().numpy()
            return u, f, c
    
        elif args.quantity == "both":

            x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
            t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

            self.dnn.eval()
            u = self.net_u(x, t)[0]
            c_momentum = torch.sum(torch.reshape(u, (100, 256))*self.delta_x, dim=1)
            c_energy = torch.sum(torch.reshape(u**2, (100, 256))*self.delta_x, dim=1)

            f = self.net_f(x, t)
            u = u.detach().cpu().numpy()
            f = f.detach().cpu().numpy()
            c_momentum  = c_momentum.detach().cpu().numpy()
            c_energy = c_energy.detach().cpu().numpy()

            return u, f, c_momentum, c_energy
    
        else:
            raise ValueError(f"Unsupported equation type: '{args.quantity}'.")


# ## Configurations

nu = 0.01/np.pi
noise = 0.0

N_u = 100
N_f = 10000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

if args.eq == "advection":
    data = np.load('data/advection_solution.npy', allow_pickle=True).item()

elif args.eq == "kdv":
    data = np.load('data/kdv_solution.npy', allow_pickle=True).item()

elif args.eq == "wave":
    data = np.load('data/wave_solution.npy', allow_pickle=True).item()

elif args.eq == "react-diff":
    data = np.load('data/reaction-diffusion_solution.npy', allow_pickle=True).item()
    
else:
    raise ValueError(f"Unsupported equation type: '{args.eq}'.")


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


X_f_train = np.vstack((X_f_train, X_u_train))
u_train = np.vstack([uu1, uu2, uu3])

if args.eq == "react-diff":
    c_momentum = np.sum((Exact)*1/128, axis=1)
    c_energy = np.sum((Exact)**2*1/128, axis=1)
else:
    c_momentum = np.mean(np.sum((Exact)*1/128, axis=1))
    c_energy = np.mean(np.sum((Exact)**2*1/128, axis=1))

idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx, :]

# ## Training

# In[8]:


model = PhysicsInformedNN(X_u_train, u_train, X_f_train, c_momentum, c_energy, x, t, layers, lb, ub, nu)

start = time.time()

model.train()

end = time.time()


print(end-start)
print(model.iter)

if args.quantity == "momentum" or args.quantity == "energy":
    u_pred, f_pred, c_pred = model.predict(X_star)

    # torch.save(c_pred, f'../graphing/data/c_pred_sc_{args.eq}_{args.quantity}.pth')

elif args.quantity == "both":
    u_pred, f_pred, c_pred_momentum, c_pred_energy = model.predict(X_star)

error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

print('%e' % (error_u))

if args.quantity == "momentum":
    error_c = np.sum(abs(c_pred-c_momentum))
    print('%e' % (error_c))

elif args.quantity == "energy":
    error_c = np.sum(abs(c_pred-c_energy))
    print('%e' % (error_c))

elif args.quantity == "both":
    error_c_momentum = np.sum(abs(c_pred_momentum-c_momentum))
    error_c_energy = np.sum(abs(c_pred_energy-c_energy))
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
file_identifier = f"{args.eq}_{args.quantity}_sc"  # Gets filename without .py extension

# The error values you want to save (modify these variable names as needed)
if args.quantity == "momentum" or args.quantity == "energy":
    error_values = [error_u, error_c]  
elif args.quantity == "both":
    error_values = [error_u, error_c_momentum, error_c_energy]  
time_values = [end-start, model.iter]

# Prepare the row to write
# row_data = [file_identifier] + [f'{val:.6e}' for val in error_values] + [f'{val}' for val in time_values]
row_data = [file_identifier] + [sc_weight] + [f'{val:.6e}' for val in error_values] + [f'{val}' for val in time_values]

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