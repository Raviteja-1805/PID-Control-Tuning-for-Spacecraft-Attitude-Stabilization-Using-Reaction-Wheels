# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:46:29 2026

@author: Teja3
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Quaternion Functions
# =========================

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,   
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def normalize(q):
    return q / np.linalg.norm(q)

# =========================
# System Parameters
# =========================

I = np.diag([0.083169476, 0.383845376, 0.423209364])
I_inv = np.linalg.inv(I)

beta = np.deg2rad(50)

B = np.array([
    [np.cos(beta), 0, -np.cos(beta), 0],
    [0, np.cos(beta), 0, -np.cos(beta)],
    [np.sin(beta), np.sin(beta), np.sin(beta), np.sin(beta)]
])

# =========================
# Initial Conditions
# =========================

q = normalize(np.array([0.5, 0.5, 0.5, 0.5]))
qd = np.array([1, 0, 0, 0])

omega = np.array([0.2, -0.1, 0.15])

# PID Gains (you can tune later)
Kp = 8
Ki = 0.02
Kd = 6

e_int = np.zeros(3)

# =========================
# Simulation Settings
# =========================

dt = 0.01
T = 20

# Storage (lists!)
time = []
angle_error = []
omega_log = []
tau_log = []
u_log = []

def simulate(I):

    I_inv = np.linalg.inv(I)

    # Initial conditions (reset every run!)
    q = normalize(np.array([0.5, 0.5, 0.5, 0.5]))
    qd = np.array([1, 0, 0, 0])
    omega = np.array([0.2, -0.1, 0.15])
    e_int = np.zeros(3)

    time = []
    angle_error = []

    for t in np.arange(0, T, dt):

        qe = quat_mult(quat_conj(qd), q)
        if qe[0] < 0:
            qe = -qe

        e = qe[1:]

        e_int += e * dt
        tau_des = -Kp*e - Kd*omega - Ki*e_int

        u = -np.linalg.pinv(B) @ tau_des
        tau = -B @ u

        omega_dot = I_inv @ (tau - np.cross(omega, I @ omega))
        omega += omega_dot * dt

        omega_q = np.hstack(([0], omega))
        q_dot = 0.5 * quat_mult(q, omega_q)
        q += q_dot * dt
        q = normalize(q)

        angle = 2 * np.arccos(np.clip(qe[0], -1, 1)) * 180/np.pi

        time.append(t)
        angle_error.append(angle)

    return np.array(time), np.array(angle_error)

# =========================
# Simulation Loop
# =========================

for t in np.arange(0, T, dt):

    # Quaternion error
    qe = quat_mult(quat_conj(qd), q)
    
    if qe[0] < 0:
        qe = -qe
        
    e = qe[1:]
    
    # PID control
    e_int += e * dt
    tau_des = -Kp*e - Kd*omega - Ki*e_int
    
    # Reaction wheels
    u = -np.linalg.pinv(B) @ tau_des
    tau = -B @ u
    
    # Dynamics
    omega_dot = I_inv @ (tau - np.cross(omega, I @ omega))
    omega += omega_dot * dt
    
    # Quaternion update
    omega_q = np.hstack(([0], omega))
    q_dot = 0.5 * quat_mult(q, omega_q)
    q += q_dot * dt
    q = normalize(q)
    
    # Angle error (deg)
    angle = 2 * np.arccos(np.clip(qe[0], -1, 1)) * 180/np.pi
    
    # Store data
    time.append(t)
    angle_error.append(angle)
    omega_log.append(omega.copy())
    tau_log.append(tau.copy())
    u_log.append(u.copy())

# =========================
# Convert to Arrays 
# =========================

time = np.array(time)
angle_error = np.array(angle_error)
omega_log = np.array(omega_log)
tau_log = np.array(tau_log)
u_log = np.array(u_log)

# =========================
# Cost Function
# =========================

threshold = 0.5
settling_time = T

for i in range(len(angle_error)):
    if np.all(angle_error[i:] < threshold):
        settling_time = time[i]
        break

overshoot = np.max(angle_error)
steady_state_error = angle_error[-1]
control_energy = np.sum(np.linalg.norm(tau_log, axis=1)**2) * dt

w1, w2, w3, w4 = 1.0, 1.0, 2.0, 0.1

J = (w1 * settling_time +
     w2 * overshoot +
     w3 * steady_state_error +
     w4 * control_energy)

print("Cost J:", J)

# =========================
# Plots
# =========================

# 1. Attitude Error
plt.figure()
plt.plot(time, angle_error)
plt.title("Attitude Error (deg)")
plt.xlabel("Time (s)")
plt.ylabel("Error (deg)")
plt.grid()

# 2. Angular Velocity
plt.figure()
plt.plot(time, omega_log)
plt.title("Angular Velocity")
plt.xlabel("Time (s)")
plt.ylabel("rad/s")
plt.legend(["wx", "wy", "wz"])
plt.grid()

# 3. Control Torque
plt.figure()
plt.plot(time, tau_log)
plt.title("Control Torque")
plt.xlabel("Time (s)")
plt.ylabel("Nm")
plt.legend(["Tx", "Ty", "Tz"])
plt.grid()

# 4. Reaction Wheel Torques
plt.figure()
plt.plot(time, u_log)
plt.title("Reaction Wheel Torques")
plt.xlabel("Time (s)")
plt.ylabel("Torque")
plt.legend(["Wheel1", "Wheel2", "Wheel3", "Wheel4"])
plt.grid()

plt.show()

# =========================
# Robustness Test
# =========================

I_variations = [
    I,
    1.05 * I,   # +5%
    0.95 * I,   # -5%
    np.diag([0.09, 0.37, 0.44])  # random variation
]

labels = ['Nominal', '+5%', '-5%', 'Random']

plt.figure()

for i, I_test in enumerate(I_variations):
    t_sim, err_sim = simulate(I_test)
    plt.plot(t_sim, err_sim, label=labels[i])

plt.title("Robustness to Inertia Uncertainty")
plt.xlabel("Time (s)")
plt.ylabel("Attitude Error (deg)")
plt.legend()
plt.grid()

plt.show()