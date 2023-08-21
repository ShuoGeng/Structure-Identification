import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def Amatrix(N,graph):
    '''
    N means the particle of this system
    This funtion is used to construct matrix A 
    '''
    A = np.zeros([2*N,2*N])
    for i in range(N):
        A[2*i,2*i+1] = graph[i,i]
        A[2*i+1,2*i] = -graph[i,i]
        for j in range(N):
            if j == i:
                continue
            else:      
                A[2*i,2*j+1] = -graph[i,j]
                A[2*i+1,2*j] = graph[i,j]
                A[2*j,2*i+1] = -graph[i,j]
                A[2*j+1,2*i] = graph[i,j]
    return A

def solve_differential_equation(A, x0, t_start, t_end, tstep):
    """
    Solve first order linear ordinary differential equation dx/dt = Ax
    
    Parameters:
    
    -A: indicates the coefficient matrix A
    
    -x0: indicates the initial vector x0
    
    - t_start: indicates the start time
    
    - t_end: indicates the end time
    
    - dt: indicates the time step


    Return:
    -t: indicates the time array
    -x: solution vector array
    """

    t = np.linspace(t_start,t_end,tstep)
    dt = t[1]-t[0]

    # Initialize
    x = np.zeros((len(t), len(x0)))
    x[0] = x0

    # Solve differential equation
    for i in range(1, len(t)):
        k1 = np.dot(A, x[i-1])
        k2 = np.dot(A, x[i-1] + 0.5 * dt * k1)
        k3 = np.dot(A, x[i-1] + 0.5 * dt * k2)
        k4 = np.dot(A, x[i-1] + dt * k3)
        x[i] = x[i-1] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return t, x

# Example
graph = np.array([[1.3,4.3,0,0],[4.3,2.4,5.2,0],[0,5.2,1.7,3.6],[0,0,3.6,1.6]])
A = Amatrix(4,graph)

t_start = 0
t_end = 20
tstep = 10000
x0 = [0,1,0,0,0,0,0,0]
t, x = solve_differential_equation(A, x0, t_start, t_end, tstep)
plt.plot(t,x[:,0])
plt.plot(t,np.real(tlist))
