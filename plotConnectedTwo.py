import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# parameters
g = -9.81
l = 1.0
m = 0.1
M = 1

alpha = m / (M + 2*m)

# system of first order ODEs
def system(t, y):
    theta1, omega1, theta2, omega2 = y
    
    # denominators
    D1 = 1 - alpha * np.cos(theta1)
    D2 = 1 - alpha * np.cos(theta2)
    

    A1 = ( (g/l)*np.sin(theta1)
           + alpha*(-np.sin(theta1)*omega1**2
           - np.sin(theta2)*omega2**2)*np.cos(theta1) ) / D1
    
    B1 = alpha * np.cos(theta2) * np.cos(theta1) / D1
    
    A2 = ( (g/l)*np.sin(theta2)
           + alpha*(-np.sin(theta1)*omega1**2
           - np.sin(theta2)*omega2**2)*np.cos(theta2) ) / D2
    
    B2 = alpha * np.cos(theta1) * np.cos(theta2) / D2
    

    matrix = np.array([[1, -B1],
                       [-B2, 1]])
    
    rhs = np.array([A1, A2])
    
    theta_dd = np.linalg.solve(matrix, rhs)
    
    theta1_dd, theta2_dd = theta_dd
    
    return [omega1,
            theta1_dd,
            omega2,
            theta2_dd]


T = 30
t_span = (0, T)
t_eval = np.linspace(0, T, 1000)

# initial conditions
theta1_0 = 0.5
omega1_0 = 0.0
theta2_0 = 0.3
omega2_0 = 0.0

y0 = [theta1_0, omega1_0, theta2_0, omega2_0]

# Solve using RK4.5
sol = solve_ivp(system, t_span, y0, method='RK45', t_eval=t_eval)

# Plot
plt.figure()
plt.plot(sol.t, sol.y[0], label=r'$\theta_1(t)$')
plt.plot(sol.t, sol.y[2], label=r'$\theta_2(t)$')
plt.xlabel("t")
plt.ylabel("Angle")
plt.legend()
plt.show()  
