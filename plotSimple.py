import numpy as np
import matplotlib.pyplot as plt

omega0 = 1.0
h = 0.01
T = 10
N = int(T/h)

theta = np.zeros(N)
omega = np.zeros(N)
t = np.linspace(0, T, N)
# initial conditions
theta[0] = 0.5      
omega[0] = 0.0      

def f(theta, omega):
    return np.array([omega, -omega0**2 * np.sin(theta)])

for i in range(N-1):
    y = np.array([theta[i], omega[i]])
    k1 = f(*y)
    k2 = f(*(y + 0.5*h*k1))
    k3 = f(*(y + 0.5*h*k2))
    k4 = f(*(y + h*k3))

    y_next = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    theta[i+1], omega[i+1] = y_next

plt.plot(t, theta)
plt.xlabel("t")
plt.ylabel("θ(t)")
plt.show()