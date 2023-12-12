#Fork：
import numpy as np
import matplotlib.pyplot as plt

def Amatrix(N,omega,delta):
    '''
    N means the particle of this system
    omega and delta are the parameters of the equation
    This funtion is used to construct matrix A 
    '''
    A = np.zeros([2*N,2*N])
    for i in range(1,2*(N-1),2):
        i = int(i)
        j = int((i-1)/2)
        A[i-1][i] = omega[j]
        A[i][i-1] = -omega[j]
        A[i-1][i+2] = -delta[j]
        A[i][i+1] = delta[j]
        A[i+1][i] = -delta[j]
        A[i+2][i-1] = delta[j]
    A[2*N-2,2*N-1] = omega[j+1]
    A[2*N-1,2*N-2] = -omega[j+1]
    return A
def solve_differential_equation(A, x0, t_start, t_end, dt):
    t = np.arange(t_start, t_end + dt, dt)
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        k1 = np.dot(A, x[i-1])
        k2 = np.dot(A, x[i-1] + 0.5 * dt * k1)
        k3 = np.dot(A, x[i-1] + 0.5 * dt * k2)
        k4 = np.dot(A, x[i-1] + dt * k3)
        x[i] = x[i-1] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return t, x


J = [4.3,5.2,1.7]
# Define the external magnetic field strength h
h = [1.3,2.4,1.7,2.1]
Am = Amatrix(4,h,J)
A = Am.copy()
A[2,7] = -1.7
A[3,6] = 1.7
A[6,3] = -1.7
A[7,2] = 1.7
A[4,7] = 0
A[5,6] = 0
A[6,5] = 0
A[7,4] = 0
t_start = 0
t_end = 20
dt = 0.0001
x0 = [0,1,0,0,0,0,0,0]
t2, x2 = solve_differential_equation(A, x0, t_start, t_end, dt)
# plt.plot(t,x[:,0])

# Define the chain length and exchange interaction strength J
L = 3
L2 = 4
J = [4.3,5.2,1.7]

# Define the external magnetic field strength h
h = [1.3,2.4,1.7,2.1]

# Create the Pauli spin matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
hbar = 1.0  

# Create the Hamiltonian matrix
H = np.zeros((2**L2, 2**L2), dtype=np.complex128)

# Calculate the exchange interaction terms in the Hamiltonian
for i in range(L-1):
    ip1 = (i + 1) % L  # Periodic boundary conditions
    
    H -= 0.5*J[i] * np.kron(np.kron(np.eye(2**i), sigma_x), np.kron(sigma_x, np.eye(2**(L2-i-2)))) + \
         0.5*J[i] * np.kron(np.kron(np.eye(2**i), sigma_y), np.kron(sigma_y, np.eye(2**(L2-i-2))))  

H -= 0.5*J[L-1] * np.kron(np.kron(np.kron(np.eye(2**1),sigma_x),np.eye(2**1)), sigma_x)+ \
0.5*J[L-1] * np.kron(np.kron(np.kron(np.eye(2**1),sigma_y),np.eye(2**1)), sigma_y)

# Calculate the external magnetic field term in the Hamiltonian
for i in range(L2):
    H -= 0.5*h[i] * np.kron(np.kron(np.eye(2**i), sigma_z), np.eye(2**(L2 - i - 1)))
H = H.astype('float32')
# H = -H

t_total = 20  
num_steps = 10000 
dt = t_total / num_steps  
dt1 = dt
spin = np.array([1j,-1j,1,1,1,1,1,1,1,1])
spin = spin/np.linalg.norm(spin)
time = np.linspace(0, t_total, num_steps)
density_matrices = []
# Define spin-up and spin-down basis vectors
spin_up = np.array([1, 0])
spin_down = np.array([0, 1])

# State of the first node in the y-direction
state_1 = 1/np.sqrt(2) * (spin_up + spin_down)

# Initial state
initial_state = np.kron(state_1, np.kron(spin_up, np.kron(spin_up, spin_up)))

# Initial density matrix
initial_density_matrix = np.outer(initial_state, initial_state.conj())


rho = initial_density_matrix 

for step in range(num_steps):
    # Append the density matrix to the list
    density_matrices.append(rho)

    # Calculate the Hamiltonian
    # Use the fourth-order Runge-Kutta method for time evolution
    k1 = -1j / hbar * (np.dot(H, rho) - np.dot(rho, H)) * dt
    k2 = -1j / hbar * (np.dot(H, rho + 0.5 * k1) - np.dot(rho + 0.5 * k1, H)) * dt
    k3 = -1j / hbar * (np.dot(H, rho + 0.5 * k2) - np.dot(rho + 0.5 * k2, H)) * dt
    k4 = -1j / hbar * (np.dot(H, rho + k3) - np.dot(rho + k3, H)) * dt
    rho = rho + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

# Plot the evolution of density matrix elements over time
density_matrices = np.array(density_matrices)

Sx = np.kron(np.kron(sigma_y, np.eye(2)),np.eye(4))
list = []
for i in range(num_steps):
    list.append(np.trace(Sx@density_matrices[i,:,:]))
    
    
plt.figure()    
plt.plot(np.arange(0,t_total,dt1),-np.real(list))
plt.legend(['Directly'])
plt.xlabel("Time")
plt.figure()
plt.plot(t2,x2[:,0])
plt.xlabel("Time")
plt.legend(['Matrix'])
# plt.legend('Directly')
plt.show()
