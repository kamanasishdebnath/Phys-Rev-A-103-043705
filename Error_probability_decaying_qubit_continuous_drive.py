# This code can be used to reproduce the results reported in PHYSICAL REVIEW A 103, 043705 (2021)
# This code can generate Fig. 4 by modifying the parameters used in the article.
# Also refer to the comments used in file "Error_probability_ideal_qubit_ion.py"

import numpy as np
import math
from qutip import*
sqrt= np.sqrt
exp= np.exp

Omega= 2.0
mu= 5.0
gamma= 1.0
gamma_E0= 0.05
gamma_E1= 0.05
Gamma= 0.0


T_end1= 60.0;
num_steps1= int(T_end1/0.008)
tlist1= np.linspace(0,T_end1,num_steps1)



# Defining the operators
zero_qubit=      basis(3,0)
one_qubit=       basis(3,1)
excited_qubit=   basis(3,2)
down_readout=    basis(2,0)
up_readout=      basis(2,1)


# Dissipators
R_ops = tensor(down_readout*up_readout.dag(),  qeye(3))
E0_ops= tensor(qeye(2), zero_qubit*excited_qubit.dag())
E1_ops= tensor(qeye(2),  one_qubit*excited_qubit.dag())
Dephase_ops= tensor(qeye(2),  excited_qubit*excited_qubit.dag() - one_qubit*one_qubit.dag() - zero_qubit*zero_qubit.dag())



sc_ops= [sqrt(gamma)*R_ops]
e_ops= []
c_ops= [sqrt(gamma_E0)*E0_ops, sqrt(gamma_E1)*E1_ops, sqrt(Gamma)*Dephase_ops]

##########################################################################################################################
drive_strength= gamma_E0*50
##########################################################################################################################


# The continuous drive of the qubit ion on |0> to |e> transition is given by H_drive2
# Hamiltonian
H_drive= 0.5*Omega*(tensor(up_readout*down_readout.dag(), qeye(3)) + tensor(down_readout*up_readout.dag(), qeye(3))) 
H_interation= mu*tensor(up_readout*up_readout.dag(),excited_qubit*excited_qubit.dag())
H_drive_2=  2*drive_strength*0.5*(tensor(qeye(2),excited_qubit*zero_qubit.dag()) +  tensor(qeye(2),zero_qubit*excited_qubit.dag()))


H= H_drive + H_interation + H_drive_2


#########################################################################################################
num_traj= 2000
C_1= 0.0;
C_0= 1.0;
psi0= tensor(down_readout, C_0*excited_qubit + C_1*one_qubit).unit()
#########################################################################################################



mcarlo1 = photocurrent_mesolve(H, psi0, tlist1, c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops, \
                               ntraj=num_traj, nsubsteps=100, store_measurement=True, \
                              options=Options(atol=1e-10, store_states= False, store_final_state= False),map_func= parallel_map)


def main_part(n):
    indices= (np.where(mcarlo1.measurement[n][:] != [0.0 + 0.0j])[0])
    number_of_jumps= len(indices)
    coll_ops= tensor(down_readout*up_readout.dag(), qeye(3));
    liou= spre(coll_ops)*spost(coll_ops.dag());

    trace_op= np.zeros((36),dtype = float)
    for j in range(36):
        if j%7 == 0:
            trace_op[j]=1;

    trace_op = Qobj(trace_op).dag()   
    trace_op.dims =  [[1],[[2, 3], [2, 3]]]     

    Hamiltonian= H - 0.5j*gamma*coll_ops.dag()*coll_ops
    rho0 = operator_to_vector(ket2dm(tensor(down_readout, excited_qubit).unit()))
    rho1 = operator_to_vector(ket2dm(tensor(down_readout, one_qubit).unit()))

    trace_rho0=np.zeros((num_steps1),dtype=complex)
    trace_rho1=np.zeros((num_steps1 ),dtype=complex)


    L1= -1.0j*(spre(Hamiltonian)-spost(Hamiltonian.dag())) 
    L2= gamma_E0* lindblad_dissipator(E0_ops, b=None, data_only=False) 
    L3= gamma_E1* lindblad_dissipator(E1_ops, b=None, data_only=False)

    L= L1 + L2 + L3
    uni= qeye(36)
    uni.dims= [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]
    U = uni + L*(tlist1[3]-tlist1[2])     

    indOld= 0
    for i, ind in enumerate(indices):
        t2 = tlist1[ind]
        tlist2= tlist1[indOld:ind]

        for j, time in enumerate(tlist2):
            rho0 = U*rho0
            rho1 = U*rho1


            trace_rho0[indOld+j] = (trace_op*rho0).tr()
            trace_rho1[indOld+j] = (trace_op*rho1).tr()
            rho0 = rho0/trace_rho0[indOld+j]
            rho1 = rho1/trace_rho1[indOld+j]

        indOld = ind


        state0 = U*rho0
        rho0 =   liou * state0
        state1 = U*rho1
        rho1 =   liou * state1

    trace_rho1 = np.cumprod(trace_rho1)
    trace_rho0 = np.cumprod(trace_rho0)

    P1= np.abs(np.true_divide(trace_rho1,(trace_rho0 + trace_rho1)))
    time_shot= np.min(np.where(np.isnan(P1)==True))
    
    P1=np.where(np.isnan(P1)==True, P1[time_shot-1], P1)
    P0= 1-P1
    
    if C_1==1:
        return(P0>0.5)*1
    else:
        return(P1>0.5)*1



final_result = parallel_map(main_part, range(num_traj), num_cpus=7, progress_bar=True)
averaged_result= np.mean(final_result, axis=0)