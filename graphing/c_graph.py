import matplotlib.pyplot as plt
import numpy as np

# Set up matplotlib styling to match your format
plt.rcParams.update({
    'font.size': 16,          # Default font size
    'axes.titlesize': 26,     # Title font size
    'axes.labelsize': 24,     # X and Y label font size
    'xtick.labelsize': 18,    # X tick label font size
    'ytick.labelsize': 18,    # Y tick label font size
    'legend.fontsize': 24     # Legend font size
})

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",  # Computer Modern for math
})


import torch

# gt2 = 1.4934674763300624
# gt5 = 1.195918752835773
# gt4 = 0.44139492374871303
# gt3 = 0.9866768037384155
# gt6 = 0.7059462080870169
# gt1 = 0.6241722165222272

# Your data lists - replace these with your actual data
# Each should be a list with your data values

eqs = ["advection", "wave", "kdv"]
quantities = ["momentum", "energy"]


pinn_data = [torch.load(f"data/c_pred_pinn_{eq}_{quant}.pth", weights_only=False) for eq in eqs for quant in quantities]  # 6 lists for PINN data
soft_data = [torch.load(f"data/c_pred_sc_{eq}_{quant}.pth", weights_only=False) for eq in eqs for quant in quantities]  # 6 lists for PINN data
proj_data = [torch.load(f"data/c_pred_proj_{eq}_{quant}.pth", weights_only=False) for eq in eqs for quant in quantities]  # 6 lists for PINN data
ground_truth = [np.mean(torch.load(f"data/c_true_proj_{eq}_{quant}.pth", weights_only=False)) for eq in eqs for quant in quantities]  # 6 lists for PINN data


# Plot titles for each subplot
titles = ['Advection Eq. - Linear Integral', 
          'Wave Eq. - Linear Integral', 
          'KdV Eq. - Linear Integral',
          'Advection Eq. - Quadratic Integral', 
          'Wave Eq. - Quadratic Integral', 
          'KdV Eq. - Quadratic Integral']

# Y-axis labels (customize as needed)
y_labels = ['c(t)', '', '', 'c(t)', '', '']

# Y-axis limits for each plot (adjust as needed)
# y_limits = [(0.62, 0.632), (1.46, 1.52), (0.986, 0.9875), 
#            (0.441, 0.442), (1.15, 1.25), (0.704, 0.707)]
y_limits = []
for i in range(6):
    # Combine all data for this subplot
    all_data = pinn_data[i].tolist() + soft_data[i].tolist() + proj_data[i].tolist()
    
    if i == 1:
        print(pinn_data[i].tolist())

    # Find min and max values
    data_min = min(all_data)
    data_max = max(all_data)

    print(data_min, data_max)
    
    # Calculate distances from ground truth
    dist_below = ground_truth[i] - data_min
    dist_above = data_max - ground_truth[i]
    
    # Take the maximum distance and multiply by 1.1
    max_dist = max(dist_below, dist_above) * 1.1
    
    # Set symmetric limits around ground truth
    y_min = ground_truth[i] - max_dist
    y_max = ground_truth[i] + max_dist
    
    y_limits.append((y_min, y_max))


# Create 2x3 subplot grid
fig, axes = plt.subplots(2, 3, figsize=(22, 10))

# Plot each subplot
for i in range(6):
    row, col = i // 3, i % 3
    ax = axes[row, col]
    
    # Create x-axis values
    x = range(len(pinn_data[i]))
    
    # Plot the three lines
    ax.plot(x, pinn_data[i], 'b-', label='PINN', linewidth=3, alpha=0.7)
    ax.plot(x, soft_data[i], 'g-', label='PINN-SC', linewidth=5, alpha=0.7)
    ax.plot(x, proj_data[i], 'r-', label='PINN-Proj', linewidth=3, alpha=0.7)
    
    # Add ground truth line
    ax.axhline(y=ground_truth[i], color='black', linestyle='--', linewidth=2, label='Ground Truth')
    
    # Set y-limits
    print(y_limits[i])
    ax.set_ylim(y_limits[i])
    
    # Customize each subplot
    ax.set_xlabel('Time')
    ax.set_ylabel(y_labels[i])
    ax.set_title(titles[i])

# Create unified legend at the bottom center
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, 
          bbox_to_anchor=(0.5, -0.02), fontsize=26)  # Changed from 0.02 to -0.02

# Adjust layout to make room for legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  

# Save the figure
plt.savefig('c_plots.png', dpi=300, bbox_inches='tight')
# plt.show()