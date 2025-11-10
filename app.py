import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Real-Time Sensor Fusion Visualizer", layout="wide")

st.title("🛰️ Real-Time Multi-Sensor Data and Image Fusion Simulator")
st.markdown("### Simulating Lidar, Radar, and Camera Data with Bayesian + Kalman Fusion")

# --- Simulation parameters ---
time_steps = st.slider("Simulation Steps", 20, 200, 50)
true_velocity = st.slider("True Object Velocity (m/s)", 0.5, 5.0, 2.0)
noise_levels = {"Lidar": 1.0, "Radar": 1.5, "Camera": 2.0}

# --- Generate true trajectory ---
true_positions = np.linspace(0, true_velocity * time_steps, time_steps)

# --- Simulate noisy sensor readings ---
lidar = true_positions + np.random.normal(0, noise_levels["Lidar"], time_steps)
radar = true_positions + np.random.normal(0, noise_levels["Radar"], time_steps)
camera = true_positions + np.random.normal(0, noise_levels["Camera"], time_steps)

# --- Bayesian Fusion ---
def bayesian_fusion(values, variances):
    weights = 1 / np.array(variances)
    fused_value = np.sum(weights * values) / np.sum(weights)
    fused_var = 1 / np.sum(weights)
    return fused_value, fused_var

bayes_fused = []
for i in range(time_steps):
    val, var = bayesian_fusion(
        [lidar[i], radar[i], camera[i]],
        [noise_levels["Lidar"]**2, noise_levels["Radar"]**2, noise_levels["Camera"]**2],
    )
    bayes_fused.append(val)

# --- Kalman Filter Implementation ---
x_est = 0.0
P = 1.0
Q = 0.01  # Process noise
R = 1.5   # Measurement noise
kalman_estimates = []

for z in bayes_fused:
    # Prediction
    x_pred = x_est + true_velocity
    P_pred = P + Q
    # Update
    K = P_pred / (P_pred + R)
    x_est = x_pred + K * (z - x_pred)
    P = (1 - K) * P_pred
    kalman_estimates.append(x_est)

# --- Display Plots ---
fig, ax = plt.subplots()
ax.plot(true_positions, label="True Position", color="black", linewidth=2)
ax.scatter(range(time_steps), lidar, s=10, label="Lidar", alpha=0.6)
ax.scatter(range(time_steps), radar, s=10, label="Radar", alpha=0.6)
ax.scatter(range(time_steps), camera, s=10, label="Camera", alpha=0.6)
ax.plot(bayes_fused, label="Bayesian Fused", color="orange", linestyle="--")
ax.plot(kalman_estimates, label="Kalman Fused", color="green", linewidth=2)
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.subheader("🧩 Simulated Image Fusion using Laplacian and Gradient Methods")

# --- Synthetic Image Simulation ---
img1 = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(img1, (100, 100), 60, 180, -1)
img2 = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img2, (50, 50), (150, 150), 220, -1)

# Laplacian Pyramid Fusion
def laplacian_fusion(A, B):
    gpA = [A.copy()]
    gpB = [B.copy()]
    for i in range(3):
        A = cv2.pyrDown(A)
        B = cv2.pyrDown(B)
        gpA.append(A)
        gpB.append(B)
    lpA = [gpA[-1]]
    lpB = [gpB[-1]]
    for i in range(3, 0, -1):
        GE_A = cv2.pyrUp(gpA[i])
        GE_B = cv2.pyrUp(gpB[i])
        L = cv2.addWeighted(GE_A, 0.5, GE_B, 0.5, 0)
        lpA.append(L)
    fused = lpA[0]
    for i in range(1, len(lpA)):
        fused = cv2.pyrUp(fused)
        fused = cv2.addWeighted(fused, 0.5, lpA[i], 0.5, 0)
    return fused

laplacian_result = laplacian_fusion(img1, img2)

# Gradient Fusion
gradient_result = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

# Display
col1, col2, col3 = st.columns(3)
col1.image(img1, caption="Sensor Image 1 (Lidar)", use_column_width=True)
col2.image(img2, caption="Sensor Image 2 (Camera)", use_column_width=True)
col3.image(laplacian_result, caption="Fused Image (Laplacian)", use_column_width=True)
st.image(gradient_result, caption="Gradient Fusion Result", use_column_width=True)

st.success("✅ Real-time Multi-Sensor Data and Image Fusion Simulation Complete!")
