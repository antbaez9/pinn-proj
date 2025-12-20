import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def reaction_term(u, k=1.0):
    """
    Define your reaction term here.
    Example: Simple first-order reaction k*u
    """
    return k * u

def solve_reaction_diffusion_1d(D=1.0, L=1.0, T=1.0, Nx=256, dt=0.01, 
                               initial_condition=None, save_file="reaction-diffusion_solution.npy"):
    """
    Solve the 1D reaction-diffusion equation using Crank-Nicolson method
    with Neumann boundary conditions (no flux)
    
    Parameters:
    -----------
    D : float
        Diffusion coefficient
    L : float
        Domain length
    T : float
        Total simulation time
    Nx : int
        Number of spatial grid points
    dt : float
        Time step
    initial_condition : function or array
        Initial concentration profile. If None, a Gaussian profile will be used
    save_file : str
        Filename to save results (.npy)
    
    Returns:
    --------
    x : array
        Spatial grid
    t : array
        Time points
    u : array
        Solution array (dimensions: Nt+1 x Nx)
    """
    # Setup spatial grid
    dx = L / Nx
    x = np.linspace(0, L, Nx)

    # Setup time grid
    Nt = int(T / dt)
    t = np.linspace(0, T, Nt+1)
    
    # Initialize solution array
    u = np.zeros((Nt+1, Nx))
    
    # Set initial condition
    if initial_condition is None:
        # Default: Gaussian pulse
        sigma = 0.5
        u[0, :] = np.exp(-((x - L/2)**2) / (sigma**2))

    elif callable(initial_condition):
        u[0, :] = initial_condition(x)
    else:
        u[0, :] = initial_condition
    
    # Compute stability parameter
    r = D * dt / (dx**2)
    print(f"Stability parameter r = {r:.4f} (should be < 0.5 for explicit, OK for implicit)")
    
    # Setup matrices for Crank-Nicolson scheme
    # Create the diffusion matrices properly
    
    # Left-hand side matrix (implicit part)
    main_diag_A = np.ones(Nx) * (1 + r)
    off_diag_A = np.ones(Nx-1) * (-r/2)
    
    # Right-hand side matrix (explicit part) 
    main_diag_B = np.ones(Nx) * (1 - r)
    off_diag_B = np.ones(Nx-1) * (r/2)
    
    # Create sparse matrices
    A = sparse.diags([off_diag_A, main_diag_A, off_diag_A], [-1, 0, 1], 
                     shape=(Nx, Nx), format='csr')
    B = sparse.diags([off_diag_B, main_diag_B, off_diag_B], [-1, 0, 1], 
                     shape=(Nx, Nx), format='csr')
    
    # Apply Neumann boundary conditions (no flux: du/dx = 0)
    # Left boundary (x=0): u[0] - u[1] = 0 or u[0] = u[1]
    # This is implemented by setting the derivative to zero
    
    # Left boundary modification
    A[0, 0] = 1 + r/2
    A[0, 1] = -r/2
    B[0, 0] = 1 - r/2  
    B[0, 1] = r/2
    
    # Right boundary modification  
    A[-1, -1] = 1 + r/2
    A[-1, -2] = -r/2
    B[-1, -1] = 1 - r/2
    B[-1, -2] = r/2
    
    print(f"Starting time integration with Nt = {Nt} steps")
    
    # Time integration
    for n in range(Nt):
        # Compute reaction contribution (explicit treatment)
        reaction_contrib = dt * reaction_term(u[n, :], k=0.5)
        
        # Setup right-hand side including both diffusion and reaction
        rhs = B.dot(u[n, :]) + reaction_contrib
        
        # Solve the linear system
        u[n+1, :] = spsolve(A, rhs)
    
    # Save results
    save_file = "reaction-diffusion_solution.npy"
    np.save(save_file, {'x': x, 't': t, 'u': u})
    
    return x, t, u

if __name__ == "__main__":
    # Example usage with more stable parameters
    x, t, u = solve_reaction_diffusion_1d(D=0.1, L=2, dt=0.01, T=0.99, Nx=256)

    # Proper integration using trapezoidal rule
    dx = x[1] - x[0]
    
    # Calculate mass and energy using proper integration
    mass = np.trapz(u, x, axis=1)  # Integrate over space
    energy = np.trapz(u**2, x, axis=1)  # Integrate over space

    print(x.shape, t.shape, u.shape)

    # Report conservation
    print(f"Mass conservation: {mass[-1] - mass[0]}")
    print(f"Energy conservation: {energy[-1] - energy[0]}")
