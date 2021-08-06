# This code can be used to reproduce the results reported in PHYSICAL REVIEW A 103, 043705 (2021)
# This code can generate Fig. 5 and by modifying the parameters used in the article.
# Also refer to the comments used in file "Error_probability_ideal_qubit_ion.py"

import numpy as np
import math
from qutip import*
sqrt= np.sqrt
exp= np.exp

# Cavity 1
Omega1= 2.0
mu1= 5.0
gamma1= 1.0

# Cavity 2
Omega2= Omega1
mu2= mu1
gamma2= gamma1

# Time and number of points
T_end1= 20
T_end= 25
num_steps= int(T_end/0.01)
tlist= np.linspace(0,T_end,num_steps)


# Defining the operators
zero_qubit=      basis(3,0)
one_qubit=       basis(3,1)
excited_qubit=   basis(3,2)
down_readout=    basis(2,0)
up_readout=      basis(2,1)


# Dissipators
R_ops1 = sqrt(gamma1)* tensor(down_readout*up_readout.dag(), qeye(3), qeye(2), qeye(3))
R_ops2 = sqrt(gamma2)* tensor(qeye(2), qeye(3), down_readout*up_readout.dag(), qeye(3))
R_ops= [R_ops1 + R_ops2 , R_ops1 - R_ops2]


# Drive Hamiltonians
H_drive1= 0.5*(tensor(up_readout*down_readout.dag(), qeye(3), qeye(2), qeye(3)) + \
                      tensor(down_readout*up_readout.dag(), qeye(3), qeye(2), qeye(3)))


# Note that the drive of the ancillas are turned off at T= 20 (T_end1)
H_drive2= 0.5*(tensor(qeye(2), qeye(3), up_readout*down_readout.dag(), qeye(3)) + \
                      tensor(qeye(2), qeye(3), down_readout*up_readout.dag(), qeye(3)))


# Note that the drive of the ancillas are turned off at T= 20 (T_end1), hence this condition
def drive2(t, args=0):
    if t>T_end1:
        return 0
    else:
        return Omega2

# Qubit-reaout ion interaction Hamiltonian
H_interation1= mu1*tensor(up_readout*up_readout.dag(),excited_qubit*excited_qubit.dag(), qeye(2), qeye(3))
H_interation2= mu2*tensor(qeye(2), qeye(3), up_readout*up_readout.dag(),excited_qubit*excited_qubit.dag())

H_interaction= H_interation1 + H_interation2
H = [H_interaction, [H_drive1,drive2], [H_drive2,drive2]]


# Parameters for initial state, which is a superposition state
C_1a= 1.0;
C_0a= 1.0;

C_1b= 1.0;
C_0b= 1.0;


psi1= tensor(down_readout, C_0a*excited_qubit + C_1a*one_qubit).unit()
psi2= tensor(down_readout, C_0b*excited_qubit + C_1b*one_qubit).unit()




psi0= tensor(psi1, psi2)
sc_ops= R_ops
e_ops= []
c_ops=[]


norm= 1/sqrt(2)
op1= tensor(qeye(2), excited_qubit*excited_qubit.dag(), qeye(2), excited_qubit*excited_qubit.dag())
op2= tensor(qeye(2), one_qubit*one_qubit.dag(), qeye(2), one_qubit*one_qubit.dag())

state_plus = norm*((tensor(qeye(2), one_qubit, qeye(2), excited_qubit)) + (tensor(qeye(2), excited_qubit, qeye(2), one_qubit)))
state_minus = norm*((tensor(qeye(2), one_qubit, qeye(2), excited_qubit)) -  (tensor(qeye(2), excited_qubit, qeye(2), one_qubit)))
op3= (state_plus*state_plus.dag())
op4= (state_minus*state_minus.dag())


num_traj= 20000
expe_1= np.zeros((num_traj, len(tlist)))
expe_2= np.zeros((num_traj, len(tlist)))
expe_3= np.zeros((num_traj, len(tlist)))
expe_4= np.zeros((num_traj, len(tlist)))

# This loop calculates the probabilities of occuring all the states specified by Eq.(14)
for i in range(num_traj):
    mcarlo= mcsolve(H, psi0, tlist, c_ops=sc_ops, e_ops=e_ops, ntraj= 1,\
                     options=Options(atol=1e-14, store_states= False, store_final_state= True))
    expe_1[i,:]= expect(op1,mcarlo.states[0][:])
    expe_2[i,:]= expect(op2,mcarlo.states[0][:])
    expe_3[i,:]= expect(op3,mcarlo.states[0][:])
    expe_4[i,:]= expect(op4,mcarlo.states[0][:])
   