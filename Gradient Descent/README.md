# Gradient Descent (Linear Regression)

## Overview
This project implements **Gradient Descent ** to train a **Linear Regression model** by optimizing parameters \( w \) (slope) and \( b \) (bias).

---

## Problem Setup
- Dataset:  
  `x = [1,2,3,4]`  
  `y = [2,3,5,6]`  

- Model:  
  `y = wx + b`

- Goal: Find optimal `w` and `b` for the best-fit line.

---

## Cost Function (Mean Squared Error)

J(w,b) = (1/n) * Σ (yᵢ - (wxᵢ + b))²

---

## Approach

### 1. Initialization
- `w = 0`, `b = 0`
- Learning rate `α = 0.05`

---

### 2. Gradient Calculation

Dw = (2/n) * Σ [-xᵢ(yᵢ - y_pred)]  
Db = (2/n) * Σ [-(yᵢ - y_pred)]

---

### 3. First Iteration

- Initial predictions = 0  
- Errors = `[2, 3, 5, 6]`

Gradients:
- `Dw = -23.5`
- `Db = -8.0`

---

### 4. Parameter Update

w = w - α * Dw  
b = b - α * Db  

Updated values:
- `w = 1.175`
- `b = 0.4`

New model:
y = 1.175x + 0.4

---

## Performance

- Initial MSE: `18.5`  
- After 1 iteration: `0.5522`  

➡️ Shows rapid error reduction

---

## Implementation Details
- Built using **NumPy**
- Custom functions:
  - Prediction (`line`)
  - Cost (`MSE`)
  - Gradient Descent loop
- Iterations: `1000`

---

## Key Learnings
- Gradient Descent minimizes loss step-by-step  
- Gradients determine update direction  
- Learning rate impacts convergence  
- Significant improvement occurs early  

---

## Conclusion
This project demonstrates how Gradient Descent works mathematically and programmatically to find the best-fit line in Linear Regression.

---
