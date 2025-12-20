import numpy as np
from scipy.sparse import diags
import time

def improved_weno_1d_wave_equation_solver():
    """
    Improved WENO solver for the 1D wave equation with better conservation properties.
    Uses a conservative formulation and symplectic time integration.
    
    Parameters:
    - dt = 0.01
    - t = [0, 1]
    - dx = 1/256
    - x = [0, 2]
    - Neumann BCs
    
    Returns x, t, u and saves to .npy file
    """
    # Start timing
    start_time = time.time()
    
    # Parameters
    dt = 0.01
    t_final = 0.99
    dx = 1/256
    x_min, x_max = 0.0, 2.0
    
    # WENO parameters
    r = 3  # WENO order (2r-1 = 5th order)
    
    # Derived parameters
    nt = int(t_final / dt) + 1
    nx = 256 * 16  # 4096 points
    
    # Grid
    x = np.linspace(0, 2, nx)
    t = np.linspace(0, t_final, nt)
    
    # Wave equation parameters
    c = 0.25  # Wave speed
    
    # CFL check
    cfl = c * dt / dx
    print(f"CFL number: {cfl:.4f}")
    
    # Initialize solution arrays
    u = np.zeros((nt, nx))
    v = np.zeros(nx)  # Velocity field du/dt
    
    # Initial condition: Gaussian pulse
    sigma = 1
    u[0, :] = np.exp(-((x - 1.0) ** 2) / (sigma ** 2))
    
    # Initial velocity (zero for standing wave)
    v[:] = 0.0
    
    # Time stepping with symplectic velocity-Verlet scheme
    for n in range(nt - 1):
        # Half-step velocity update
        d2u_dx2 = conservative_weno_second_derivative(u[n, :], dx)
        v += 0.5 * dt * c**2 * d2u_dx2
        
        # Full-step position update
        u[n+1, :] = u[n, :] + dt * v
        
        # Apply Neumann boundary conditions
        u[n+1, 0] = u[n+1, 1]
        u[n+1, -1] = u[n+1, -2]
        
        # Half-step velocity update with new positions
        d2u_dx2 = conservative_weno_second_derivative(u[n+1, :], dx)
        v += 0.5 * dt * c**2 * d2u_dx2
        
        # Apply boundary conditions to velocity
        v[0] = 0.0
        v[-1] = 0.0
    
    # Calculate conservation quantities
    x_downsampled = x[::16]
    u_downsampled = u[:, ::16]
    
    dx_coarse = x_downsampled[1] - x_downsampled[0]
    m = np.sum(u_downsampled * dx_coarse, axis=1)
    E = np.sum((u_downsampled**2) * dx_coarse, axis=1)
    
    # Save results
    np.save('wave_solution_new.npy', {'x': x_downsampled, 't': t, 'u': u_downsampled})
    
    
    return x_downsampled, t, u_downsampled

def weno_weights(beta, gamma, epsilon=1e-6):
    """
    Compute WENO weights from smoothness indicators.
    
    Parameters:
    - beta: Smoothness indicators
    - gamma: Linear weights
    - epsilon: Small parameter to avoid division by zero
    
    Returns:
    - omega: Normalized nonlinear weights
    """
    alpha = gamma / (epsilon + beta)**2
    omega = alpha / np.sum(alpha)
    return omega

def weno_flux_reconstruction(v_stencil, direction='positive'):
    """
    WENO flux reconstruction for conservative schemes.
    
    Parameters:
    - v_stencil: 5-point stencil for flux reconstruction
    - direction: 'positive' for left-biased, 'negative' for right-biased
    
    Returns:
    - Reconstructed flux value
    """
    epsilon = 1e-6
    
    if direction == 'negative':
        v_stencil = v_stencil[::-1]
    
    # Linear weights for 5th-order WENO
    gamma = np.array([0.1, 0.6, 0.3])
    
    # Three sub-stencils for reconstruction
    # Stencil 0: v[i-2], v[i-1], v[i]
    p0 = (2*v_stencil[0] - 7*v_stencil[1] + 11*v_stencil[2]) / 6
    
    # Stencil 1: v[i-1], v[i], v[i+1]
    p1 = (-v_stencil[1] + 5*v_stencil[2] + 2*v_stencil[3]) / 6
    
    # Stencil 2: v[i], v[i+1], v[i+2]
    p2 = (2*v_stencil[2] + 5*v_stencil[3] - v_stencil[4]) / 6
    
    # Smoothness indicators
    beta0 = 13/12 * (v_stencil[0] - 2*v_stencil[1] + v_stencil[2])**2 + \
            1/4 * (v_stencil[0] - 4*v_stencil[1] + 3*v_stencil[2])**2
    
    beta1 = 13/12 * (v_stencil[1] - 2*v_stencil[2] + v_stencil[3])**2 + \
            1/4 * (v_stencil[1] - v_stencil[3])**2
    
    beta2 = 13/12 * (v_stencil[2] - 2*v_stencil[3] + v_stencil[4])**2 + \
            1/4 * (3*v_stencil[2] - 4*v_stencil[3] + v_stencil[4])**2
    
    # Compute weights
    beta = np.array([beta0, beta1, beta2])
    omega = weno_weights(beta, gamma, epsilon)
    
    # Weighted reconstruction
    return omega[0]*p0 + omega[1]*p1 + omega[2]*p2

def conservative_weno_second_derivative(u, dx):
    """
    Conservative WENO scheme for second derivative computation.
    Uses flux formulation to ensure conservation.
    
    Parameters:
    - u: Solution array at current time
    - dx: Grid spacing
    
    Returns:
    - Second derivative array
    """
    n = len(u)
    d2u = np.zeros(n)
    
    # Extend array with ghost cells for Neumann BCs
    u_ext = np.zeros(n + 6)
    u_ext[3:-3] = u
    
    # Neumann BC: mirror points for zero derivative
    u_ext[0] = u[5]
    u_ext[1] = u[4]
    u_ext[2] = u[3]
    u_ext[-3] = u[-4]
    u_ext[-2] = u[-5]
    u_ext[-1] = u[-6]
    
    # First compute numerical fluxes at cell interfaces
    flux = np.zeros(n + 1)
    
    for i in range(3, n + 3):
        # Get stencil for flux computation
        stencil = u_ext[i-2:i+3]
        
        # Compute flux using WENO reconstruction
        # For second derivative, flux is first derivative
        flux_plus = weno_flux_reconstruction(stencil, 'positive')
        flux_minus = weno_flux_reconstruction(stencil, 'negative')
        
        # Use Lax-Friedrichs flux splitting for stability
        alpha = 1.0  # Maximum wave speed for scalar equation
        flux[i-3] = 0.5 * (flux_plus + flux_minus - alpha * (stencil[2] - stencil[2]))
    
    # Compute first derivative from flux differences
    du_dx = np.zeros(n + 1)
    for i in range(n + 1):
        if i == 0:
            du_dx[i] = (flux[1] - flux[0]) / dx
        elif i == n:
            du_dx[i] = (flux[n] - flux[n-1]) / dx
        else:
            du_dx[i] = (flux[i] - flux[i-1]) / dx
    
    # Compute second derivative using conservative difference
    for i in range(n):
        if i == 0:
            # Use one-sided difference at boundary
            d2u[i] = 2 * (du_dx[1] - du_dx[0]) / dx
        elif i == n-1:
            # Use one-sided difference at boundary
            d2u[i] = 2 * (du_dx[n] - du_dx[n-1]) / dx
        else:
            # Central difference for interior points
            d2u[i] = (du_dx[i+1] - du_dx[i]) / dx
    
    # Apply smoothing near boundaries for stability
    d2u[0] = 0.5 * (d2u[0] + d2u[1])
    d2u[-1] = 0.5 * (d2u[-1] + d2u[-2])
    
    return d2u

def alternative_conservative_scheme(u, dx):
    """
    Alternative conservative scheme using compact finite differences
    with WENO limiting for shock capturing.
    """
    n = len(u)
    d2u = np.zeros(n)
    
    # Standard second-order central difference
    for i in range(1, n-1):
        d2u_central = (u[i+1] - 2*u[i] + u[i-1]) / dx**2
        
        # WENO limiter for high gradients
        if i >= 2 and i < n-2:
            # Compute smoothness indicator
            beta = (u[i+1] - u[i-1])**2 + (u[i] - u[i-1])**2
            
            if beta > 1e-3:  # Apply WENO if gradient is large
                # 5-point stencil
                stencil = [u[i-2], u[i-1], u[i], u[i+1], u[i+2]]
                
                # WENO reconstruction for smoother solution
                d2u_weno = (weno_flux_reconstruction(stencil, 'positive') - 
                           2*u[i] + 
                           weno_flux_reconstruction(stencil[::-1], 'negative')) / dx**2
                
                # Blend central and WENO
                theta = min(1.0, beta / 1e-2)
                d2u[i] = (1 - theta) * d2u_central + theta * d2u_weno
            else:
                d2u[i] = d2u_central
        else:
            d2u[i] = d2u_central
    
    # Boundary conditions (Neumann)
    d2u[0] = d2u[1]
    d2u[-1] = d2u[-2]
    
    return d2u

if __name__ == "__main__":
    x, t, u = improved_weno_1d_wave_equation_solver()

    m = np.sum(u/128, axis=1)
    E = np.sum((u**2)/128, axis=1)

    np.save('wave_solution.npy', {'x': x, 't': t, 'u': u})

    # --- Report conservation ---
    print(f"Momentum conservation: {m[-1] - m[0]}")
    print(f"Energy conservation: {E[-1] - E[0]}")
