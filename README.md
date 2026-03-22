# Parametric Physics-Constrained Neural Network for Casson Fluid Flow

This is my first project Physics Informed Machine Learning.
This project presents a parametric physics-informed neural network (PINN-style model) for simulating steady 1D Casson fluid flow in a cylindrical pipe.

## 🚀 Key Features

- Parametric learning across viscosity values (μ_c)
- No labeled training data required
- Captures plug flow behavior in yield-stress fluids
- Achieves ~18× speedup over numerical methods at inference
- Implemented using PyTorch

## 🧠 Method Overview

The model takes:
- Radial coordinate (r)
- Plastic viscosity (μ_c)

and predicts:
- Velocity profile u(r)

The loss function enforces:
- Casson constitutive relationship
- Plug flow condition (du/dr = 0)
- No-slip boundary condition (u(R) = 0)

## ⚠️ Important Note

This is a **physics-constrained neural network** rather than a classical PINN, as it enforces the constitutive relation instead of the full Navier–Stokes PDE.

## 📊 Results

- Mean error: ~1e-3 m/s
- Accurate plug region representation
- Strong generalization to unseen viscosity values

## 🎬 Visualization

A Manim animation of the training and some results is included:

`media/casson_parametric_PINN.mp4`
