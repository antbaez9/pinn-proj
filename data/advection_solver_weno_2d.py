import numpy as np
from scipy.sparse import diags
from scipy.linalg import norm
import time
import tqdm

def weno5_reconstruction_vectorized(v_array, direction='positive'):
    """
    Vectorized 5th-order WENO reconstruction with WENO-Z improvement
    v_array: array of shape (n_points, 5) containing stencils
    Returns: array of shape (n_points,) with reconstructed values
    """
    epsilon = 1e-6  # Improved epsilon for better stability
    
    v = v_array.T  # Shape (5, n_points) for easier indexing
    
    if direction == 'positive':
        # Ideal weights for left-biased stencil
        d0, d1, d2 = 3/10, 6/10, 1/10
        
        # Smoothness indicators
        beta0 = 13/12 * (v[0] - 2*v[1] + v[2])**2 + 1/4 * (v[0] - 4*v[1] + 3*v[2])**2
        beta1 = 13/12 * (v[1] - 2*v[2] + v[3])**2 + 1/4 * (v[1] - v[3])**2
        beta2 = 13/12 * (v[2] - 2*v[3] + v[4])**2 + 1/4 * (3*v[2] - 4*v[3] + v[4])**2
        
        # WENO-Z improvement
        tau = np.abs(beta0 - beta2)
        
        # Nonlinear weights
        alpha0 = d0 * (1 + (tau / (epsilon + beta0))**2)
        alpha1 = d1 * (1 + (tau / (epsilon + beta1))**2)
        alpha2 = d2 * (1 + (tau / (epsilon + beta2))**2)
        
        # Normalize
        omega_sum = alpha0 + alpha1 + alpha2
        omega0 = alpha0 / omega_sum
        omega1 = alpha1 / omega_sum
        omega2 = alpha2 / omega_sum
        
        # Reconstruct
        p0 = (2*v[0] - 7*v[1] + 11*v[2]) / 6
        p1 = (-v[1] + 5*v[2] + 2*v[3]) / 6
        p2 = (2*v[2] + 5*v[3] - v[4]) / 6
        
    else:  # negative velocity
        # Ideal weights for right-biased stencil
        d0, d1, d2 = 1/10, 6/10, 3/10
        
        # Smoothness indicators
        beta0 = 13/12 * (v[4] - 2*v[3] + v[2])**2 + 1/4 * (v[4] - 4*v[3] + 3*v[2])**2
        beta1 = 13/12 * (v[3] - 2*v[2] + v[1])**2 + 1/4 * (v[3] - v[1])**2
        beta2 = 13/12 * (v[2] - 2*v[1] + v[0])**2 + 1/4 * (3*v[2] - 4*v[1] + v[0])**2
        
        # WENO-Z improvement
        tau = np.abs(beta0 - beta2)
        
        # Nonlinear weights
        alpha0 = d0 * (1 + (tau / (epsilon + beta0))**2)
        alpha1 = d1 * (1 + (tau / (epsilon + beta1))**2)
        alpha2 = d2 * (1 + (tau / (epsilon + beta2))**2)
        
        # Normalize
        omega_sum = alpha0 + alpha1 + alpha2
        omega0 = alpha0 / omega_sum
        omega1 = alpha1 / omega_sum
        omega2 = alpha2 / omega_sum
        
        # Reconstruct
        p0 = (-v[4] + 5*v[3] + 2*v[2]) / 6
        p1 = (2*v[3] + 5*v[2] - v[1]) / 6
        p2 = (11*v[2] - 7*v[1] + 2*v[0]) / 6
    
    return omega0*p0 + omega1*p1 + omega2*p2

def compute_flux_x_vectorized(u, cx):
    """
    Vectorized computation of numerical flux in x-direction
    u: 2D array (ny, nx)
    cx: advection velocity in x-direction
    """
    ny, nx = u.shape
    flux = np.zeros((ny, nx+1))
    
    # Extend with ghost cells (improved Neumann BC: mirror for better symmetry)
    u_ext = np.zeros((ny, nx+4))
    u_ext[:, 2:nx+2] = u
    
    # Improved Neumann BC: use mirroring for better symmetry
    u_ext[:, 1] = u[:, 0]
    u_ext[:, 0] = u[:, 0]
    u_ext[:, nx+2] = u[:, nx-1]
    u_ext[:, nx+3] = u[:, nx-1]
    
    # Vectorized stencil extraction for interior points
    if nx > 1:
        stencils = np.zeros((ny, nx-1, 5))
        for k in range(5):
            stencils[:, :, k] = u_ext[:, k:k+nx-1]
        
        # Apply WENO reconstruction to all interior points at once
        direction = 'positive' if cx >= 0 else 'negative'
        for j in range(ny):
            flux[j, 1:nx] = cx * weno5_reconstruction_vectorized(stencils[j], direction)
    
    # Zero-flux boundary conditions
    flux[:, 0] = 0.0
    flux[:, nx] = 0.0
    
    return flux

def compute_flux_y_vectorized(u, cy):
    """
    Vectorized computation of numerical flux in y-direction
    u: 2D array (ny, nx)
    cy: advection velocity in y-direction
    """
    ny, nx = u.shape
    flux = np.zeros((ny+1, nx))
    
    # Extend with ghost cells (improved Neumann BC: mirror for better symmetry)
    u_ext = np.zeros((ny+4, nx))
    u_ext[2:ny+2, :] = u
    
    # Improved Neumann BC: use mirroring for better symmetry
    u_ext[1, :] = u[0, :]
    u_ext[0, :] = u[0, :]
    u_ext[ny+2, :] = u[ny-1, :]
    u_ext[ny+3, :] = u[ny-1, :]
    
    # Vectorized stencil extraction for interior points
    if ny > 1:
        stencils = np.zeros((ny-1, nx, 5))
        for k in range(5):
            stencils[:, :, k] = u_ext[k:k+ny-1, :]
        
        # Apply WENO reconstruction to all interior points at once
        direction = 'positive' if cy >= 0 else 'negative'
        for i in range(nx):
            flux[1:ny, i] = cy * weno5_reconstruction_vectorized(stencils[:, i], direction)
    
    # Zero-flux boundary conditions
    flux[0, :] = 0.0
    flux[ny, :] = 0.0
    
    return flux

def tvd_rk3_step_2d_unsplit(u, cx, cy, dx, dy, dt):
    """
    Third-order TVD Runge-Kutta time integration - UNSPLIT METHOD
    Applies both x and y operators together at each stage (properly conservative)
    This is the correct way to handle 2D advection without splitting errors
    """
    # Stage 1: u1 = u - dt * L(u)
    flux_x = compute_flux_x_vectorized(u, cx)
    flux_y = compute_flux_y_vectorized(u, cy)
    u1 = u - dt/dx * np.diff(flux_x, axis=1) - dt/dy * np.diff(flux_y, axis=0)
    
    # Stage 2: u2 = (3u + u1 - dt * L(u1)) / 4
    flux_x = compute_flux_x_vectorized(u1, cx)
    flux_y = compute_flux_y_vectorized(u1, cy)
    u2 = (3*u + u1 - dt/dx * np.diff(flux_x, axis=1) - dt/dy * np.diff(flux_y, axis=0)) / 4
    
    # Stage 3: u_new = (u + 2*u2 - 2*dt * L(u2)) / 3
    flux_x = compute_flux_x_vectorized(u2, cx)
    flux_y = compute_flux_y_vectorized(u2, cy)
    u_new = (u + 2*u2 - 2*dt/dx * np.diff(flux_x, axis=1) - 2*dt/dy * np.diff(flux_y, axis=0)) / 3
    
    return u_new

def solve_2d_advection_equation(u_initial, x, y, t, cx=1.0, cy=1.0):
    """
    Solve the 2D advection equation u_t + cx*u_x + cy*u_y = 0 
    using improved WENO5 with unsplit TVD-RK3 for better conservation
    """
    nx = len(x) - 1
    ny = len(y) - 1
    nt = len(t) - 1
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = t[1] - t[0]
    
    # Improved CFL check for 2D unsplit method
    cfl_x = abs(cx) * dt / dx
    cfl_y = abs(cy) * dt / dy
    cfl_2d = np.sqrt(cfl_x**2 + cfl_y**2)  # 2D CFL number
    
    print(f"CFL Analysis:")
    print(f"  CFL_x = {cfl_x:.4f}")
    print(f"  CFL_y = {cfl_y:.4f}")
    print(f"  CFL_2D = {cfl_2d:.4f}")
    
    # For unsplit WENO5 with TVD-RK3, CFL should be < ~0.6
    if cfl_2d > 0.6:
        print(f"  WARNING: CFL_2D = {cfl_2d:.4f} > 0.6 may cause instability")
    else:
        print(f"  CFL condition satisfied")
    
    # Initialize solution array
    u = np.zeros((nt+1, ny+1, nx+1))
    u[0, :, :] = u_initial.copy()
    
    # Store initial conservation quantities
    mass_initial = np.sum(u[0, :, :]) * dx * dy
    energy_initial = np.sum(u[0, :, :]**2) * dx * dy
    
    print(f"\nInitial Conditions:")
    print(f"  Mass: {mass_initial:.6e}")
    print(f"  Energy: {energy_initial:.6e}")
    
    # Time integration
    for n in tqdm.tqdm(range(nt), desc="Time integration"):
        # Apply unsplit TVD-RK3 time stepping
        u[n+1, :, :] = tvd_rk3_step_2d_unsplit(u[n, :, :], cx, cy, dx, dy, dt)
        
        # Check for numerical instability
        if np.any(np.isnan(u[n+1, :, :])) or np.max(np.abs(u[n+1, :, :])) > 1e6:
            print(f"\nSolution diverging at time step {n+1}")
            return u[:n+1, :, :]
        
        # Periodic conservation check
        if (n+1) % 10 == 0:
            mass = np.sum(u[n+1, :, :]) * dx * dy
            energy = np.sum(u[n+1, :, :]**2) * dx * dy
            mass_error = abs(mass - mass_initial) / (mass_initial + 1e-10)
            energy_error = abs(energy - energy_initial) / (energy_initial + 1e-10)
            
            if mass_error > 1e-6 or energy_error > 0.05:
                print(f"\nStep {n+1}: Mass err = {mass_error:.2e}, Energy err = {energy_error:.2e}")
    
    return u

def compute_conservation_metrics(u, dx, dy):
    """
    Compute conservation metrics for the 2D solution
    """
    nt = u.shape[0]
    mass = np.zeros(nt)
    energy = np.zeros(nt)
    max_val = np.zeros(nt)
    min_val = np.zeros(nt)
    
    for i in range(nt):
        mass[i] = np.sum(u[i, :, :]) * dx * dy
        energy[i] = np.sum(u[i, :, :]**2) * dx * dy
        max_val[i] = np.max(u[i, :, :])
        min_val[i] = np.min(u[i, :, :])
    
    return mass, energy, max_val, min_val

def main():
    # Set up computational domain
    resolution = 256
    x = np.linspace(0, 2, resolution)
    y = np.linspace(0, 2, resolution)
    t = np.linspace(0, 0.99, 100)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = t[1] - t[0]
    
    # Create 2D coordinate arrays
    X, Y = np.meshgrid(x, y)
    
    # Set advection velocities
    cx = 0.25
    cy = 0.25
    
    # Initial condition (2D Gaussian pulse)
    sigma = 1
    x0, y0 = 1, 1
    u_initial = np.exp(-((X - x0)**2 + (Y - y0)**2) / sigma**2)
    
    print("=" * 60)
    print("2D ADVECTION EQUATION SOLVER - IMPROVED VERSION")
    print("=" * 60)
    print(f"\nProblem Setup:")
    print(f"  Grid: {resolution}x{resolution} points")
    print(f"  Domain: [0,2] x [0,2]")
    print(f"  Time: {len(t)} steps, dt = {dt:.4f}")
    print(f"  Velocities: cx = {cx}, cy = {cy}")
    print(f"  Method: WENO5 + unsplit TVD-RK3")
    print(f"  BC: Zero-flux (Neumann)")
    print()
    
    # Solve the 2D advection equation
    start_time = time.time()
    u_full = solve_2d_advection_equation(u_initial, x, y, t, cx, cy)
    elapsed_time = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print(f"Computation completed in {elapsed_time:.2f} seconds")
    print(f"{'=' * 60}")
    
    # Compute conservation metrics on full solution
    mass, energy, max_val, min_val = compute_conservation_metrics(u_full, dx, dy)
    
    # Report conservation analysis
    mass_error = abs(mass[-1] - mass[0]) / (abs(mass[0]) + 1e-10)
    energy_error = abs(energy[-1] - energy[0]) / (abs(energy[0]) + 1e-10)
    
    print(f"\nConservation Analysis:")
    print(f"  Mass conservation error: {mass_error:.6e}")
    print(f"  Energy conservation error: {energy_error:.6e}")
    print(f"\nAbsolute Changes:")
    print(f"  Δ Mass: {mass[-1] - mass[0]:.6e}")
    print(f"  Δ Energy: {energy[-1] - energy[0]:.6e}")
    print(f"\nSolution Bounds:")
    print(f"  Initial: min = {min_val[0]:.6f}, max = {max_val[0]:.6f}")
    print(f"  Final:   min = {min_val[-1]:.6f}, max = {max_val[-1]:.6f}")
    
    # Check TVD property (no new extrema)
    if min_val[-1] < min_val[0] - 1e-6 or max_val[-1] > max_val[0] + 1e-6:
        print(f"  WARNING: TVD property violated (new extrema created)")
    else:
        print(f"  TVD property satisfied (no new extrema)")
    
    # Downsample for storage
    u = u_full[:, ::8, ::8]
    print(f"\nDownsampled solution shape: {u.shape}")
    
    # Save solution
    filename = 'advection_solution_2d.npy'
    np.save(filename, {
        'x': x[::8],
        'y': y[::8],
        't': t,
        'u': u,
        'mass': mass,
        'energy': energy,
    })
    
    print(f"Solution saved to {filename}")
    print("=" * 60)

if __name__ == "__main__":
    main()