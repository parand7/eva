# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:41:16 2026

@author: Asus
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem

T_sp = 20   # setpoint temp
T_o = 10    # c, outdoor temp
ρ = 1.2     # kg/m³ air density
c = 1.05e3  # J/(kgK) air specific heat
h_out = 25   # W/(m^2*K) outdoor convection coefficient
H_out = 3       # m, height of a bulding
l_out = 14   # m, lenght of zone 1
l_in = 4.56     # lenght of inside wall
S_out = H_out * l_out   # S outdoor walls z_1
λ_c = 0.55      # W/(m*K), thermal conductivity, Concerete medium, engineering toolbox
ρ_c = 2300  # kg/m³, Density
c_c = 880    # J/(kg⋅K), Specific heat
w_c1 = 0.15  # m, width outdoor wall concrete
λ_i = 0.05     # W/(m*K), insulator thermal conductivity
ρ_i = 55    #Density , kg/m³
c_i = 1210  #Specific heat, J/(kg⋅K)
w_i1 = 0.07   # m, width insulator
S_ins = H_out * (l_out - (4 * w_c1))     # S insulator zone 1
S_in = H_out * l_in

ACH = 0.05
V_room = S_in * H_out
V_dot = ρ * c * ACH / 3600 * V_room

K_p = 0     # controller proportional gain. zero = off. infinity = perfect 
n_q = 18
n_θ = 15

# temperature nodes
θ = [f'θ{i}' for i in range(n_θ)]

# flow-rate branches
q = [f'q{i}' for i in range(n_q)]

A = np.zeros([n_q, n_θ])       # n° of branches X n° of nodes
A[0, 0] = 1                     # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1   # branch 2: node 1 -> node 2
A[3, 2], A[3, 3] = -1, 1
A[4, 3], A[4, 4] = -1, 1
A[5, 4], A[5, 5] = -1, 1
A[6, 5], A[6, 6] = -1, 1
A[7, 6], A[7, 7] = -1, 1
A[8, 7], A[8, 8] = -1, 1
A[9, 8], A[9, 9] = -1, 1
A[10, 9], A[10, 10] = -1, 1
A[11, 10], A[11, 11] = -1, 1
A[12, 11], A[12, 12] = -1, 1
A[13, 12], A[13, 13] = -1, 1
A[14, 13], A[14, 14] = -1, 1
A[15, 14] = 1
A[16, 5] = 1
A[17, 5], A[17, 9] = -1, 1

G = np.zeros([n_q])
G[0] = h_out * S_out
G[1] = λ_c / (w_c1 / 2) * S_out
G[2] = λ_c / (w_c1 / 2) * S_out
G[3] = λ_i / (w_i1 / 2) * S_ins
G[4] = λ_i / (w_i1 / 2) * S_ins
G[5] = h_out * S_ins
G[6] = h_out * S_ins
G[7] = λ_c / (w_c1 / 2) * S_in
G[8] = λ_c / (w_c1 / 2) * S_in
G[9] = h_out * S_ins
G[10] = h_out * S_ins
G[11] = λ_i / (w_i1 / 2) * S_ins
G[12] = λ_i / (w_i1 / 2) * S_ins
G[13] = λ_c / (w_c1 / 2) * S_out
G[14] = λ_c / (w_c1 / 2) * S_out
G[15] = h_out * S_out
G[16] = K_p
G[17] = ρ * c * ACH / 3600 * V_room
G = np.diag(G)

f = np.zeros(n_θ)


C = np.zeros(n_θ)
C[0] = ρ_c * c_c * w_c1 * h_out * l_out
C[1] = ρ_i * c_i * w_i1 * h_out * l_in
C[2] = ρ_c * c_c * w_c1 * h_out * l_out
C[3] = ρ_c * c_c * w_i1 * h_out * l_in
C[4] = ρ_c * c_c * w_c1 * h_out * l_out


G = np.diag(G)

b = np.zeros(n_q, dtype=object)
b[[0, 15]] = "T_o"
b[16] = "T_sp"
b = pd.Series(b, index=q)

f = np.zeros(n_θ, dtype=object)
#f[[0, 18]] = "T_o"
#f[16] = "T_sp"
f = pd.Series(f, index=θ)

y = np.zeros(n_θ, dtype=object)
y[[5, 9]] = 1
y = pd.Series(y, index=θ)

# thermal circuit
A = pd.DataFrame(A, index=q, columns=θ)
G = pd.Series(G, index=q)
C = pd.Series(C, index=θ)
b = pd.Series(b, index=q)
f = pd.Series(f, index=θ)
y = pd.Series(y, index=θ)

TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}

dm4bem.print_TC(TC)

# State-space
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

# Eigenvalues analysis
λ = np.linalg.eig(As)[0]        # eigenvalues of matrix As

# time step
Δtmax = 2 * min(-1. / λ)    # max time step for stability of Euler explicit
dm4bem.print_rounded_time('Δtmax', Δtmax)

dt = dm4bem.round_time(Δtmax)
 
dm4bem.print_rounded_time('dt', dt)
print(f"dt = {dt:.0f} s")

# settling time
t_settle = 4 * max(-1 / λ)
dm4bem.print_rounded_time('t_settle', t_settle)

# duration: next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle / 3600) * 3600
dm4bem.print_rounded_time('duration', duration)


# Create input_data_set
# ---------------------
# time vector
n = int(np.floor(duration / dt))    # number of time steps

To = 10 * np.ones(n)        # outdoor temperature
Ti_sp = 20 * np.ones(n)     # indoor temperature set point

# DateTimeIndex starting at "00:00:00" with a time step of dt
time = pd.date_range(start="2000-01-01 00:00:00",
                           periods=n, freq=f"{int(dt)}s")

data = {'T_o': To, 'T_sp': Ti_sp}
input_data_set = pd.DataFrame(data, index=time)

# inputs in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)

# Eigenvalues analysis
λ = np.linalg.eig(As)[0]        # eigenvalues of matrix As
λ_stable = λ[np.real(λ) < 0]

dm4bem.print_rounded_time('dt', dt)
print(f"dt = {dt:.0f} s")

# settling time
t_settle = 4 * max(-1 / λ)
dm4bem.print_rounded_time('t_settle', t_settle)

# Initial conditions
θ_exp = pd.DataFrame(index=u.index)     # empty df with index for explicit Euler
θ_imp = pd.DataFrame(index=u.index)     # empty df with index for implicit Euler

θ0 = 0.0                    # initial temperatures
θ_exp[As.columns] = θ0      # fill θ for Euler explicit with initial values θ0
θ_imp[As.columns] = θ0      # fill θ for Euler implicit with initial values θ0

I = np.eye(As.shape[0])     # identity matrix
for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As)\
        @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
    θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * As)\
        @ (θ_imp.iloc[k] + dt * Bs @ u.iloc[k])
        
# outputs
y_exp = (Cs @ θ_exp.T + Ds @  u.T).T
y_imp = (Cs @ θ_imp.T + Ds @  u.T).T

# plot results
y = pd.concat([y_exp, y_imp], axis=1, keys=['Explicit', 'Implicit'])
# Flatten the two-level column labels into a single level
y.columns = y.columns.get_level_values(0)

ax = θ_exp.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Indoor temperature, $\\theta_i$ / °C')
ax.set_title(f'Time step: $dt$ = {dt:.0f} s; $dt_{{max}}$ = {Δtmax:.0f} s')
plt.show()




filename = r'C:\Users\Asus\Desktop\Zhaw Courses\Eva\project\ESP_CM_Albacete-Los.Llanos.AFB.082800_TMYx.2004-2018.epw'

data, meta = dm4bem.read_epw(filename, coerce_year=None)

Tout = data['temp_air']

filename = r'C:\Users\Asus\Desktop\Zhaw Courses\Eva\project\ESP_CM_Albacete-Los.Llanos.AFB.082800_TMYx.2004-2018.epw'
[data, meta] = dm4bem.read_epw(filename, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data

