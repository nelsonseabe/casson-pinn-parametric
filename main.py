import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import time

# --- Global Configuration ---
dpi = 700

# LaTeX Styling and Transparent Background Configuration
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],

    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white"
})

# 1. Device-Agnostic Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Seed set to: {seed} | Running on: {device}")
    return device

device = set_seed(7)

# 1. Define the Physical Parameters (Typical Values for melted chocolate)
R = 0.025           # Pipe radius
dp_dz = -15000      # Pressure Gradient
tau_y = 10.0        # Yield Stress (Pa)

# --- Sensitivity Analysis: List of mu_c values ---
mu_c_list = [1.5, 2.5, 3.5, 4.0, 5] 
sensitivity_pinn_results = []
sensitivity_num_results = []

# Analytical Solution for the Casson Fluid in a Pipe
def casson_analytical_velocity(r_vals, current_mu_c, current_r_plug):
    u = np.zeros_like(r_vals)
    for i,r in enumerate(r_vals):
        if r < current_r_plug:
            # Plug flow region (constant velocity)
            term_1 = (1/current_mu_c) * (1/4) * abs(dp_dz) * (R**2 - current_r_plug**2)
            term_2 = (1/current_mu_c) * (2/3) * np.sqrt(2 * tau_y * abs(dp_dz)) * (R**1.5 - current_r_plug**1.5)
            term_3 = (1/current_mu_c) * tau_y * (R - current_r_plug)
            u[i] = term_1 - term_2 + term_3
        else:
            term_1 = (1/current_mu_c) * (1/4) * abs(dp_dz) * (R**2 - r**2)
            term_2 = (1/current_mu_c) * (2/3) * np.sqrt(2 * tau_y * abs(dp_dz)) * (R**1.5 - r**1.5)
            term_3 = (1/current_mu_c) * tau_y * (R - r)
            u[i] = term_1 - term_2 + term_3
    return u

# 2. b) Numerical Approach (Finite Difference)
def casson_numerical_solver(N, current_mu_c):
    r_num = np.linspace(0, R, N)
    dr = r_num[1] - r_num[0]
    u_num = np.zeros(N)
    u_num[-1] = 0 
    
    # Integrate backwards from the wall to the center
    for i in range(N-1, 0, -1):
        r_current = r_num[i]
        tau = (r_current / 2) * abs(dp_dz)
        if tau > tau_y:
            du_dr = - (1/current_mu_c) * (np.sqrt(tau) - np.sqrt(tau_y))**2
        else:
            du_dr = 0
        u_num[i-1] = u_num[i] - du_dr * dr
    return r_num, u_num

# 3. Neural Network Architecture (PARAMETRIC: 2 Inputs [r, mu_c])
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 1)
        )

    def forward(self, r, mu):
        # Concatenate radius and viscosity as features
        x = torch.cat([r, mu], dim=1)
        return self.net(x)

def compute_losses(model, r_coll, mu_coll, r_bc, mu_bc):
    # --- Physics Loss ---
    r_coll.requires_grad = True
    u_pred = model(r_coll, mu_coll)
    
    # Calculate du/dr using automatic differentiation
    du_dr = torch.autograd.grad(u_pred, r_coll, 
                                grad_outputs=torch.ones_like(u_pred), 
                                create_graph=True)[0]
    
    # Physics: tau = (r/2) * |dp/dz|
    tau_actual = (r_coll / 2) * abs(dp_dz)
    
    # 1. Shear Region (tau > tau_y): Enforce the Casson constitutive equation
    shear_mask = tau_actual > tau_y
    loss_shear = torch.tensor(0.0, requires_grad=True).to(device)
    if shear_mask.any():
        expected_du_dr_shear = - ( (torch.sqrt(tau_actual[shear_mask]) - np.sqrt(tau_y))**2 ) / mu_coll[shear_mask]
        loss_shear = torch.mean((du_dr[shear_mask] - expected_du_dr_shear)**2)

    # 2. Plug Region (tau <= tau_y): Enforce rigid body behavior (zero gradient)
    plug_mask = ~shear_mask
    loss_plug = torch.tensor(0.0, requires_grad=True).to(device)
    if plug_mask.any():
        # Ideally, du/dr in the center of the pipe must be exactly 0
        loss_plug = torch.mean(du_dr[plug_mask]**2)

    # Total Physics Residual
    loss_physics = loss_shear + loss_plug

    # --- Boundary Condition Loss (u(R, mu) = 0) ---
    u_bc_pred = model(r_bc, mu_bc)
    loss_bc = torch.mean(u_bc_pred**2)
    
    return loss_physics, loss_bc

# --- 5. PARAMETRIC TRAINING SETUP ---
model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 50000

r_vals = torch.linspace(0, R, 50)
mu_vals = torch.linspace(min(mu_c_list), max(mu_c_list), 20)
R_grid, MU_grid = torch.meshgrid(r_vals, mu_vals, indexing='ij')

r_coll = R_grid.reshape(-1, 1).to(device)
mu_coll = MU_grid.reshape(-1, 1).to(device)

mu_bc = torch.linspace(min(mu_c_list), max(mu_c_list), 50).view(-1, 1).to(device)
r_bc = (torch.ones_like(mu_bc) * R).to(device)

print("\nStarting Parametric Training (Single Model for all mu_c)...")
start_time = time.time()
history_l_phys = [] 
history_l_bc = []

for epoch in range(epochs):
    optimizer.zero_grad()
    l_phys, l_bc = compute_losses(model, r_coll, mu_coll, r_bc, mu_bc)
    total_loss = l_phys + 10 * l_bc
    
    total_loss.backward()
    optimizer.step()
    
    history_l_phys.append(l_phys.item())
    history_l_bc.append(l_bc.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss.item():.6f} (Phys: {l_phys.item():.6f}, BC: {l_bc.item():.6f})")

print(f"Training Complete. Time: {time.time() - start_time:.2f}s")

# --- SAVE THE TRAINED MODEL ---
torch.save(model.state_dict(), 'parametric_casson_pinn.pth')
print("Model saved to 'parametric_casson_pinn.pth'")

# --- 7. Plotting Training Losses ---
plt.figure(figsize=(10, 6), dpi=dpi)
plt.plot(history_l_phys, label=r'Differential Equation ($\mathcal{L}_{DE}$) Loss', color='orange', alpha=0.8)
plt.plot(history_l_bc, label=r'Boundary Condition ($\mathcal{L}_{BC}$) Loss', color='green', alpha=0.8)
plt.yscale('log')
plt.title(r'\textbf{Parametric Training Loss Components}', fontsize=14)
plt.xlabel(r'\textbf{Epochs}', fontsize=12)
plt.ylabel(r'\textbf{Mean Squared Error}', fontsize=12)
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
# Save with transparent background
plt.savefig('parametric_training_losses.png', dpi=dpi, transparent='white')
plt.show()

# --- 8. Evaluation Loop for Individual mu_c Values ---
for mu_c in mu_c_list:
    print(f"\nEvaluating Results for mu_c = {mu_c} Pa.s")
    r_plug = (2 * tau_y) / abs(dp_dz)
    
    r_plot = np.linspace(0, R, 100)
    u_analytical = casson_analytical_velocity(r_plot, mu_c, r_plug)
    r_num, u_numerical = casson_numerical_solver(100, mu_c)

    with torch.no_grad():
        r_test_torch = torch.tensor(r_plot, dtype=torch.float32).view(-1, 1).to(device)
        mu_test_torch = (torch.ones_like(r_test_torch) * mu_c).to(device)
        u_pinn = model(r_test_torch, mu_test_torch).cpu().numpy().flatten()

    sensitivity_pinn_results.append(u_pinn)
    sensitivity_num_results.append(u_numerical)

    plt.figure(figsize=(12, 7), dpi=dpi)
    plt.plot(r_plot, u_analytical, 'b-', linewidth=4, alpha=0.6, label=r'\textbf{Analytical Solution}')
    plt.plot(r_num, u_numerical, 'go', markersize=4, alpha=0.5, label=r'\textbf{Numerical (Finite Diff)}')
    plt.plot(r_plot, u_pinn, 'r--', linewidth=2, label=r'\textbf{PINN Prediction}')
    plt.axvline(x=r_plug, color='k', linestyle=':', label=r'\textbf{Plug Radius}')

    plt.title(r'\textbf{Parametric Evaluation: } $\mu_c = ' + str(mu_c) + r'$ \textbf{ Velocity Profile}', fontsize=14)
    plt.xlabel(r'\textbf{Radial Distance, } $r$ \textbf{ (m)}', fontsize=12)
    plt.ylabel(r'\textbf{Velocity, } $u_z$ \textbf{ (m/s)}', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'velocity_profile_comparison_mu_{mu_c}.png', dpi=dpi, transparent='white')
    plt.show()

# --- 9. FINAL SENSITIVITY ANALYSIS GRAPH ---
plt.figure(figsize=(12, 8), dpi=dpi)
colors = plt.cm.plasma(np.linspace(0, 0.8, len(mu_c_list)))

for i, mu in enumerate(mu_c_list):
    plt.plot(r_num, sensitivity_num_results[i], 'o', markersize=3, color=colors[i], alpha=0.3, label=r'\textbf{Numerical ($\mu_c = ' + str(mu) + r'$)}')
    plt.plot(r_plot, sensitivity_pinn_results[i], '--', color=colors[i], linewidth=2, label=r'\textbf{PINN ($\mu_c = ' + str(mu) + r'$)}')

plt.title(r'\textbf{Parametric Sensitivity Analysis: Impact of Plastic Viscosity ($\mu_c$)}', fontsize=16)
plt.xlabel(r'\textbf{Radial Distance, } $r$ \textbf{ (m)}', fontsize=12)
plt.ylabel(r'\textbf{Velocity, } $u_z$ \textbf{ (m/s)}', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig('parametric_viscosity_sensitivity.png', dpi=dpi, transparent='white')
plt.show()

# --- 10. COMPUTATIONAL BENCHMARKING SECTION ---
print("\n" + "="*50)
print("SECTION 10: COMPUTATIONAL PERFORMANCE BENCHMARKING")
print(f"Benchmarking with N = 1,000,000 collocation points")
print("="*50)

benchmark_mu_list = [1.8, 2.7, 3.2, 4.5]
N_bench = 1000000
benchmark_data = []

model_inference = PINN().to(device)
model_inference.load_state_dict(torch.load('parametric_casson_pinn.pth'))
model_inference.eval()

for mu_bench in benchmark_mu_list:
    start_num = time.time()
    _, _ = casson_numerical_solver(N_bench, mu_bench)
    end_num = time.time()
    num_duration = end_num - start_num

    r_bench_tensor = torch.linspace(0, R, N_bench).view(-1, 1).to(device)
    mu_bench_tensor = (torch.ones_like(r_bench_tensor) * mu_bench).to(device)
    
    start_pinn = time.time()
    with torch.no_grad():
        _ = model_inference(r_bench_tensor, mu_bench_tensor)
    end_pinn = time.time()
    pinn_duration = end_pinn - start_pinn

    benchmark_data.append({
        'Viscosity (mu_c)': mu_bench,
        'Numerical Time (s)': f"{num_duration:.6f}",
        'PINN Time (s)': f"{pinn_duration:.6f}",
        'Speedup Factor': f"{num_duration / pinn_duration:.2f}x"
    })

df_bench = pd.DataFrame(benchmark_data)
print(df_bench.to_string(index=False))

# --- 11. SPECIFIC PREDICTION AND PLOTTING SECTION ---
mu_c_prediction = 3.2 
r_plot_res = 100 

r_num_pred, u_num_pred = casson_numerical_solver(r_plot_res, mu_c_prediction)

r_pred_plot = np.linspace(0, R, r_plot_res)
with torch.no_grad():
    r_pred_torch = torch.tensor(r_pred_plot, dtype=torch.float32).view(-1, 1).to(device)
    mu_pred_torch = (torch.ones_like(r_pred_torch) * mu_c_prediction).to(device)
    u_pinn_pred = model_inference(r_pred_torch, mu_pred_torch).cpu().numpy().flatten()

plt.figure(figsize=(10, 6), dpi=dpi)
plt.plot(r_num_pred, u_num_pred, 'go', markersize=4, alpha=0.6, label=r'\textbf{Numerical ($\mu_c = 3.2$)}')
plt.plot(r_pred_plot, u_pinn_pred, 'r-', linewidth=2, label=r'\textbf{PINN Prediction ($\mu_c = 3.2$)}')
plt.title(r'\textbf{Blind Prediction Analysis: Viscosity } $\mu_c = 3.2$', fontsize=14)
plt.xlabel(r'\textbf{Radial Distance, } $r$ \textbf{ (m)}', fontsize=12)
plt.ylabel(r'\textbf{Velocity, } $u_z$ \textbf{ (m/s)}', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('specific_prediction_analysis.png', dpi=dpi, transparent='white')
plt.show()

# --- 12. TERMINAL OUTPUT: ERROR ANALYSIS TABLES ---
print("\n" + "="*80)
print("SECTION 12: ERROR ANALYSIS TABLES FOR DOCUMENTATION")
print("="*80)

# Table 1: Comparative Error Analysis for mu_c = 5.0
mu_target = 5.0
r_plug_target = (2 * tau_y) / abs(dp_dz)
r_points = np.linspace(0, R, 10) # 10 samples for the table

# Gather data for mu_c = 5.0
u_ana_5 = casson_analytical_velocity(r_points, mu_target, r_plug_target)
_, u_num_5 = casson_numerical_solver(10, mu_target)

with torch.no_grad():
    r_tensor_5 = torch.tensor(r_points, dtype=torch.float32).view(-1, 1).to(device)
    mu_tensor_5 = torch.full_like(r_tensor_5, mu_target)
    u_pinn_5 = model_inference(r_tensor_5, mu_tensor_5).cpu().numpy().flatten()

df_table_1 = pd.DataFrame({
    'Radius (m)': r_points,
    'Analytical': u_ana_5,
    'PINN': u_pinn_5,
    'Numerical': u_num_5,
    'PINN Error': np.abs(u_ana_5 - u_pinn_5),
    'Num Error': np.abs(u_ana_5 - u_num_5)
})

print("\nTable 1: Comparative Error Analysis for mu_c = 5.0 (Analytical vs. PINN vs. Numerical)")
print("-" * 90)
print(df_table_1.to_string(index=False, float_format="%.6f"))


# Table 2: Blind Prediction Error Analysis for mu_c = 3.2
mu_blind = 3.2
r_points_blind = np.linspace(0, R, 10)

# Gather data for mu_c = 3.2
_, u_num_32 = casson_numerical_solver(10, mu_blind)

with torch.no_grad():
    r_tensor_32 = torch.tensor(r_points_blind, dtype=torch.float32).view(-1, 1).to(device)
    mu_tensor_32 = torch.full_like(r_tensor_32, mu_blind)
    u_pinn_32 = model_inference(r_tensor_32, mu_tensor_32).cpu().numpy().flatten()

df_table_2 = pd.DataFrame({
    'Radius (m)': r_points_blind,
    'Numerical (m/s)': u_num_32,
    'PINN (m/s)': u_pinn_32,
    'Abs Error (m/s)': np.abs(u_num_32 - u_pinn_32)
})

print("\nTable 2: Blind Prediction Error Analysis for mu_c = 3.2 (Numerical vs. PINN)")
print("-" * 75)
print(df_table_2.to_string(index=False, float_format="%.6f"))
print("\n" + "="*80)