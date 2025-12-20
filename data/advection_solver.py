import numpy as np
from scipy.sparse import diags
from scipy.linalg import norm
import time

def weno5_reconstruction(v, direction='positive'):
    """
    5th-order WENO reconstruction for the 1D advection equation
    Returns the reconstructed values at cell boundaries
    direction: 'positive' for left-biased stencil, 'negative' for right-biased
    """
    epsilon = 1e-10  # Small number to avoid division by zero
    
    if direction == 'positive':
        # For positive velocity (information comes from left)
        # Ideal weights for left-biased stencil
        d0 = 3/10
        d1 = 6/10
        d2 = 1/10
        
        # Smoothness indicators
        beta0 = 13/12 * (v[0] - 2*v[1] + v[2])**2 + 1/4 * (v[0] - 4*v[1] + 3*v[2])**2
        beta1 = 13/12 * (v[1] - 2*v[2] + v[3])**2 + 1/4 * (v[1] - v[3])**2
        beta2 = 13/12 * (v[2] - 2*v[3] + v[4])**2 + 1/4 * (3*v[2] - 4*v[3] + v[4])**2
        
        # WENO-Z improvement for better accuracy at critical points
        tau = abs(beta0 - beta2)
        
        # Nonlinear weights with WENO-Z
        alpha0 = d0 * (1 + (tau / (epsilon + beta0))**2)
        alpha1 = d1 * (1 + (tau / (epsilon + beta1))**2)
        alpha2 = d2 * (1 + (tau / (epsilon + beta2))**2)
        
        # Normalize
        omega_sum = alpha0 + alpha1 + alpha2
        omega0 = alpha0 / omega_sum
        omega1 = alpha1 / omega_sum
        omega2 = alpha2 / omega_sum
        
        # Reconstruct using optimal stencils
        p0 = (2*v[0] - 7*v[1] + 11*v[2]) / 6
        p1 = (-v[1] + 5*v[2] + 2*v[3]) / 6
        p2 = (2*v[2] + 5*v[3] - v[4]) / 6
        
    else:  # negative velocity (information comes from right)
        # Ideal weights for right-biased stencil
        d0 = 1/10
        d1 = 6/10
        d2 = 3/10
        
        # Smoothness indicators (same as positive but applied to flipped stencil)
        beta0 = 13/12 * (v[4] - 2*v[3] + v[2])**2 + 1/4 * (v[4] - 4*v[3] + 3*v[2])**2
        beta1 = 13/12 * (v[3] - 2*v[2] + v[1])**2 + 1/4 * (v[3] - v[1])**2
        beta2 = 13/12 * (v[2] - 2*v[1] + v[0])**2 + 1/4 * (3*v[2] - 4*v[1] + v[0])**2
        
        # WENO-Z improvement
        tau = abs(beta0 - beta2)
        
        # Nonlinear weights with WENO-Z
        alpha0 = d0 * (1 + (tau / (epsilon + beta0))**2)
        alpha1 = d1 * (1 + (tau / (epsilon + beta1))**2)
        alpha2 = d2 * (1 + (tau / (epsilon + beta2))**2)
        
        # Normalize
        omega_sum = alpha0 + alpha1 + alpha2
        omega0 = alpha0 / omega_sum
        omega1 = alpha1 / omega_sum
        omega2 = alpha2 / omega_sum
        
        # Reconstruct using optimal stencils (right-biased)
        p0 = (-v[4] + 5*v[3] + 2*v[2]) / 6
        p1 = (2*v[3] + 5*v[2] - v[1]) / 6
        p2 = (11*v[2] - 7*v[1] + 2*v[0]) / 6
    
    return omega0*p0 + omega1*p1 + omega2*p2

def compute_numerical_flux(u, c, dx):
    """
    Compute the numerical flux using 5th-order WENO scheme with proper upwinding
    Implements zero-flux boundary conditions (Neumann BC for closed boundaries)
    """
    n = len(u)
    flux = np.zeros(n+1)
    
    # Extend array with ghost cells for Neumann BCs (zero gradient)
    u_ext = np.zeros(n+4)
    u_ext[2:n+2] = u
    
    # Neumann boundary conditions (zero gradient): du/dx = 0 at boundaries
    # Left boundary: extrapolate with zero gradient
    u_ext[1] = u[0]     # First-order: copy boundary value
    u_ext[0] = u[0]     # Maintain constant value outside domain
    
    # Right boundary: extrapolate with zero gradient  
    u_ext[n+2] = u[n-1]  # First-order: copy boundary value
    u_ext[n+3] = u[n-1]  # Maintain constant value outside domain
    
    if c >= 0:  # Positive velocity
        # Interior fluxes: reconstruct from left using upwind stencil
        for i in range(1, n):
            stencil = u_ext[i-1:i+4]
            flux[i] = c * weno5_reconstruction(stencil, 'positive')
        
        # Boundary fluxes for zero-flux BC (no flow crosses boundaries)
        # This is the physical boundary condition for a closed domain
        flux[0] = 0.0   # No inflow at left boundary
        flux[n] = 0.0   # No outflow at right boundary
        
    else:  # Negative velocity
        # Interior fluxes: reconstruct from right using upwind stencil
        for i in range(1, n):
            stencil = u_ext[i-1:i+4]
            flux[i] = c * weno5_reconstruction(stencil, 'negative')
        
        # Boundary fluxes for zero-flux BC
        flux[0] = 0.0   # No outflow at left boundary
        flux[n] = 0.0   # No inflow at right boundary
    
    return flux

def tvd_rk3_step(u, c, dx, dt):
    """
    Third-order TVD Runge-Kutta time integration
    Ensures stability for hyperbolic conservation laws
    """
    # Stage 1
    flux = compute_numerical_flux(u, c, dx)
    u1 = u - dt/dx * np.diff(flux)
    
    # Stage 2
    flux = compute_numerical_flux(u1, c, dx)
    u2 = (3*u + u1 - dt/dx * np.diff(flux)) / 4
    
    # Stage 3
    flux = compute_numerical_flux(u2, c, dx)
    u_new = (u + 2*u2 - 2*dt/dx * np.diff(flux)) / 3
    
    return u_new

def solve_advection_equation(u_initial, x, t, c=1.0):
    """
    Solve the 1D advection equation u_t + c*u_x = 0 using WENO5 scheme
    with zero-flux boundary conditions
    """
    nx = len(x) - 1
    nt = len(t) - 1
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # Check CFL condition
    cfl = abs(c) * dt / dx
    if cfl > 0.6:  # More conservative CFL for WENO5 with RK3
        print(f"Warning: CFL = {cfl:.3f} > 0.6. Consider reducing dt for stability.")
    
    # Initialize solution array
    u = np.zeros((nt+1, nx+1))
    u[0, :] = u_initial.copy()
    
    # Store conservation quantities
    mass_initial = np.sum(u[0, :]) * dx
    energy_initial = np.sum(u[0, :]**2) * dx
    
    print(f"Initial mass: {mass_initial:.6e}")
    print(f"Initial energy: {energy_initial:.6e}")
    
    # Time integration
    for n in range(nt):
        # Apply TVD-RK3 time stepping
        u[n+1, :] = tvd_rk3_step(u[n, :], c, dx, dt)
        
        # Check for numerical instability
        if np.any(np.isnan(u[n+1, :])) or np.max(np.abs(u[n+1, :])) > 1e6:
            print(f"Solution diverging at time step {n+1}")
            return u[:n+1, :]
        
        # Periodic conservation check
        if (n+1) % 10 == 0:
            mass = np.sum(u[n+1, :]) * dx
            energy = np.sum(u[n+1, :]**2) * dx
            mass_error = abs(mass - mass_initial) / (mass_initial + 1e-10)
            energy_error = abs(energy - energy_initial) / (energy_initial + 1e-10)
            
            if mass_error > 1e-8 or energy_error > 0.01:
                print(f"Step {n+1}: Mass error = {mass_error:.2e}, Energy error = {energy_error:.2e}")
    
    return u

def compute_conservation_metrics(u, dx):
    """
    Compute conservation metrics for the solution
    """
    nt, nx = u.shape
    mass = np.zeros(nt)
    momentum = np.zeros(nt)
    energy = np.zeros(nt)
    
    for i in range(nt):
        mass[i] = np.sum(u[i, :]) * dx
        momentum[i] = np.sum(u[i, :]**2) * dx / 2  # For advection equation
        energy[i] = np.sum(u[i, :]**2) * dx
    
    return mass, momentum, energy

def main():
    # Set up the computational domain (keeping your parameters)
    x = np.linspace(0, 2, 256)  # spatial domain [0, 2]
    t = np.linspace(0, 0.99, 100)  # time domain [0, 0.99]
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # Set the advection velocity
    c = 0.25
    
    # Initial condition (Gaussian pulse)
    sigma = 0.1
    u_initial = np.exp(-((x - 1) / sigma)**2)
    
    print(f"CFL number: {c * dt / dx:.4f}")
    print(f"Grid: {len(x)} points, dx = {dx:.4f}")
    print(f"Time: {len(t)} steps, dt = {dt:.4f}")
    
    # Solve the advection equation
    start_time = time.time()
    u = solve_advection_equation(u_initial, x, t, c)
    elapsed_time = time.time() - start_time
    
    print(f"\nComputation time: {elapsed_time:.3f} seconds")
    print(f"Solution shape: {u.shape}")
    
    # Compute conservation metrics
    mass, momentum, energy = compute_conservation_metrics(u, dx)
    
    # Report conservation errors
    mass_error = (mass[-1] - mass[0]) / (abs(mass[0]) + 1e-10)
    momentum_error = (momentum[-1] - momentum[0]) / (abs(momentum[0]) + 1e-10)
    energy_error = (energy[-1] - energy[0]) / (abs(energy[0]) + 1e-10)
    
    print(f"\nConservation Analysis:")
    print(f"Mass conservation error: {mass_error:.6e}")
    print(f"Momentum conservation error: {momentum_error:.6e}")
    print(f"Energy conservation error: {energy_error:.6e}")
    
    print(f"\nAbsolute changes:")
    print(f"Δ Mass: {mass[-1] - mass[0]:.6e}")
    print(f"Δ Energy: {energy[-1] - energy[0]:.6e}")
    
    # Save the solution
    np.save('advection_solution.npy', {
        'x': x,
        't': t,
        'u': u,
    })
    
    print(f"\nSolution saved to 'advection_solution_fixed.npy'")

if __name__ == "__main__":
    main()