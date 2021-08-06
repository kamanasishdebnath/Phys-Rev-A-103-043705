# This code can be used to reproduce the results reported in PHYSICAL REVIEW A 103, 043705 (2021)
# This code can generate Fig. 2 by modifying the parameters used in the article.




import numpy as np
import math
from qutip import*
sqrt= np.sqrt
exp= np.exp
sqrt= np.sqrt
exp= np.exp

# Defining the states of the qubit and readout ion
zero_qubit=      basis(3,0)
one_qubit=       basis(3,1)
excited_qubit=   basis(3,2)

down_readout=    basis(2,0)
up_readout=      basis(2,1)


# Parameters, mu= dipole-dipole interaction, gamma= linewidth of readout ion
# Omega= Rabi drive on readout ion, ntraj= number of realizations for computing average error probability.
gamma= 1.0
Omega= 2.0
mu= 2.0
ntraj=20000


# Collapse operator corresponding to the operator which is being measured i.e. emission from cavity, and on 
# adiabatic elimination this is equivalent to the decay of the readout ion
c_ops= [sqrt(gamma)*(tensor(down_readout*up_readout.dag(), qeye(3)))]

# Unobserved decay rate, this is non-zero for Fig.3
e_ops= []

# Integration time
tlist= np.linspace(0,40,1000)

# Lets define that this is the initial state which we want to readout using ancilla 
C_0= 1.0
C_1= 0.0
psi0= tensor(down_readout, C_0*excited_qubit + C_1*one_qubit).unit()


P1= np.zeros((1000,ntraj), dtype=float)

# Driving of the ancilla
H_drive= 0.5*Omega*(tensor(up_readout*down_readout.dag(), qeye(3)) + tensor(down_readout*up_readout.dag(), qeye(3)))

# Dipole-dipole interaction
H_interaction = tensor(up_readout*up_readout.dag(),excited_qubit*excited_qubit.dag())
H= H_drive + mu*H_interaction   # Total Hamiltonian of the system

# Solving the Stochastic master equation
mcarlo= smesolve(H, psi0, tlist, [], c_ops, e_ops, ntraj = ntraj, options=Options(atol=1e-10, store_states= False, store_final_state= False),nsubsteps=100,method='photocurrent', store_measurement=True,map_func= parallel_map)

Hamiltonian= H - 0.5j*gamma*coll_ops.dag()*coll_ops


# This function calculated the probabilities defined by Eq.(6)
def func1(n):
    # This calculates the propagators at different times specified by tlist
    U = propagator(Hamiltonian, tlist, c_op_list=[], args={}, options=Options(normalize_output = False),unitary_mode='single')
    
    # Time where the quantum jumps (detection event) occured
    indices= np.where(mcarlo.measurement[n][:] != [0.0 + 0.0j])[0]
    number_of_jumps= len(indices)
    rho0 = ket2dm(tensor(down_readout, excited_qubit).unit())
    rho1 = ket2dm(tensor(down_readout, one_qubit).unit())

    trace_rho0=np.zeros((1000),dtype=complex)
    trace_rho1=np.zeros((1000),dtype=complex)

    indOld= 0
    
    # This loop calculates the probabilities 
    # governed by Bayesian rule between two detection events.
    for i, ind in enumerate(indices):
        t2 = tlist[ind]
        tlist2= tlist[indOld:ind]

        for j, time in enumerate(tlist2):
            U[j].dims= [[2, 3], [2, 3]]
            trace_rho0[indOld+j] = (U[j]*rho0*U[j].dag()).tr()
            trace_rho1[indOld+j] = (U[j]*rho1*U[j].dag()).tr()

        indOld = ind
        state0= U[j]*rho0*U[j].dag()
        rho0=   coll_ops*state0*coll_ops.dag()
        state1= U[j]*rho1*U[j].dag()
        rho1=   coll_ops*state1*coll_ops.dag()
    
    
    # Error probability for a single trajectory
    P1= np.abs(np.true_divide(trace_rho1,(trace_rho1 + trace_rho0)))
    
    time_shot= np.min(np.where(np.isnan(P1)==True))
    P1=np.where(np.isnan(P1)==True, P1[time_shot-1], P1)
    P1=(P1 > 0.5)*1 
    return (P1)

# To calculate the average error probability Q_E, ntraj number of trajectories are averaged. 
xxxx = parfor(func1, range(ntraj))