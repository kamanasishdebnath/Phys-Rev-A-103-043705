# This code can be used to reproduce the results reported in PHYSICAL REVIEW A 103, 043705 (2021)
# This code can generate Fig. 6 by modifying the parameters used in the article.
# Also refer to the comments used in file "Error_probability_ideal_qubit_ion.py" 
# This code is similar to "Remote_entanglement.py" but in the presence of detector inefficiency
# Please refer to the comments in "Remote_entanglement.py" file.


import numpy as np
import math
from qutip import*
sqrt= np.sqrt
exp= np.exp

# Cavity 1
Omega1= 2.0
mu1= 30.0
gamma1= 1.0

# Cavity 2
Omega2= Omega1
mu2= mu1
gamma2= gamma1
Gamma= 0.0


# In[10]:


# Time and number of points
T_end1= 20*0 + 0.1
T_end= 25*0 + 5.1
num_steps= int(T_end/0.01)
tlist= np.linspace(0,T_end,num_steps)


# Defining the operators
zero_qubit=      basis(3,0)
one_qubit=       basis(3,1)
excited_qubit=   basis(3,2)
down_readout=    basis(2,0)
up_readout=      basis(2,1)


# Eta is the detector inefficiency
eta= 0.75

# Dissipators
# Note that the detector inefficiency eta is incorporated in the jump opearator

R_ops1 = sqrt(eta)*sqrt(gamma1)* tensor(down_readout*up_readout.dag(), qeye(3), qeye(2), qeye(3))
R_ops2 = sqrt(eta)*sqrt(gamma2)* tensor(qeye(2), qeye(3), down_readout*up_readout.dag(), qeye(3))
sc_ops= [R_ops1 + R_ops2 , R_ops1 - R_ops2]
c_ops= [sqrt(1-eta)*(R_ops1 + R_ops2) , sqrt(1-eta)*(R_ops1 - R_ops2)]
e_ops= []   

dephase_operator= (excited_qubit*excited_qubit.dag() - one_qubit*one_qubit.dag())

       

          
drive_operator= excited_qubit*one_qubit.dag()
# Drive Hamiltonians
H_drive1= 0.5*(tensor(up_readout*down_readout.dag(), qeye(3), qeye(2), qeye(3)) +                tensor(down_readout*up_readout.dag(), qeye(3), qeye(2), qeye(3))) 


# changes has been made below to change the frequency of one of the ancilla
H_drive2= 0.5*(tensor(qeye(2), qeye(3), up_readout*down_readout.dag(), qeye(3)) +                       tensor(qeye(2), qeye(3), down_readout*up_readout.dag(), qeye(3))) 


def drive(t, args=0):
    if t>T_end1:
        return 0
    else:
        return Omega2
    
    
def drive2(t, args=0):
    if t>T_end1:
        return 0
    else:
        return Omega2

# Qubit-reaout ion interaction Hamiltonian
H_interation1= mu1*tensor(up_readout*up_readout.dag(),excited_qubit*excited_qubit.dag(), qeye(2), qeye(3))
H_interation2= mu2*tensor(qeye(2), qeye(3), up_readout*up_readout.dag(),excited_qubit*excited_qubit.dag())

H_interaction= H_interation1 + H_interation2
H = [H_interaction,[H_drive1,drive], [H_drive2,drive2]]


# In[13]:


# Parameters for initial state
C_1a= 1.0;
C_0a= 1.0;

C_1b= 1.0;
C_0b= 1.0;


psi1= tensor(down_readout, C_0a*excited_qubit + C_1a*one_qubit).unit()
psi2= tensor(down_readout, C_0b*excited_qubit + C_1b*one_qubit).unit()

psi0= tensor(psi1, psi2)
norm= 1/sqrt(2)
op1= tensor(qeye(2), excited_qubit*excited_qubit.dag(), qeye(2), excited_qubit*excited_qubit.dag())
op2= tensor(qeye(2), one_qubit*one_qubit.dag(), qeye(2), one_qubit*one_qubit.dag())

state_plus = norm*((tensor(qeye(2), one_qubit, qeye(2), excited_qubit)) + (tensor(qeye(2), excited_qubit, qeye(2), one_qubit)))
state_minus = norm*((tensor(qeye(2), one_qubit, qeye(2), excited_qubit)) -  (tensor(qeye(2), excited_qubit, qeye(2), one_qubit)))
op3= (state_plus*state_plus.dag())
op4= (state_minus*state_minus.dag())

num_traj= 2500
expe_1= np.zeros((num_traj, len(tlist)))
expe_2= np.zeros((num_traj, len(tlist)))
expe_3= np.zeros((num_traj, len(tlist)))
expe_4= np.zeros((num_traj, len(tlist)))




mcarlo= photocurrent_mesolve(H, psi0, tlist, c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops, ntraj=num_traj, nsubsteps=100, store_measurement=False, options=Options(atol=1e-10, store_states= False, store_final_state= True), map_func= parallel_map)    
for i in range(num_traj):
    expe_3[i,:]= expect(op3, mcarlo.states[i][:])
    expe_4[i,:]= expect(op4, mcarlo.states[i][:])
