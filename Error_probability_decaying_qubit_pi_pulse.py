# This code can be used to reproduce the results reported in PHYSICAL REVIEW A 103, 043705 (2021)
# This code can generate Fig. 3 by modifying the parameters used in the article.



import numpy as np
import math
from qutip import*
import itertools 
import time as tt
import random
sqrt= np.sqrt
exp= np.exp
Omega= 2.0
mu= 5.0
gamma= 1.0
gamma_E0= 0.1   # decay rate of qubit ion from |e> to |0>
gamma_E1= 0.1   # decay rate of qubit ion from |e> to |1>
Gamma= 0.0
zero_qubit=      basis(3,0)
one_qubit=       basis(3,1)
excited_qubit=   basis(3,2)
down_readout=    basis(2,0)
up_readout=      basis(2,1)
R_ops = tensor(down_readout*up_readout.dag(),  qeye(3))
E0_ops= tensor(qeye(2), zero_qubit*excited_qubit.dag())
E1_ops= tensor(qeye(2),  one_qubit*excited_qubit.dag())
Dephase_ops= tensor(qeye(2),  excited_qubit*excited_qubit.dag() - one_qubit*one_qubit.dag() - zero_qubit*zero_qubit.dag())

# Collapse operator corresponding to continuous monitoring i.e. of the readout ion at a rate gamma
sc_ops= [sqrt(gamma)*R_ops]

# Collapse operator corresponding to the decay and dephasing of the qubit ion
c_ops= [sqrt(gamma_E0)*E0_ops, sqrt(gamma_E1)*E1_ops, sqrt(Gamma)*Dephase_ops]
H_drive= 0.5*Omega*(tensor(up_readout*down_readout.dag(), qeye(3)) + tensor(down_readout*up_readout.dag(), qeye(3)))
H_interation= mu*tensor(up_readout*up_readout.dag(),excited_qubit*excited_qubit.dag())
H= H_drive + H_interation
########################################################################################################################
C_1= 0.0;
C_0= 1.0;
psi0= tensor(down_readout, C_0*excited_qubit + C_1*one_qubit).unit()
###########################################################################################################################
coll_ops= tensor(down_readout*up_readout.dag(), qeye(3));
liou= spre(coll_ops)*spost(coll_ops.dag());
Hamiltonian= H - 0.5j*gamma*coll_ops.dag()*coll_ops
L1= -1.0j*(spre(Hamiltonian)-spost(Hamiltonian.dag())) 
L2= gamma_E0* lindblad_dissipator(E0_ops, b=None, data_only=False) 
L3= gamma_E1* lindblad_dissipator(E1_ops, b=None, data_only=False)
L= L1 + L2 + L3 

def propagate(psi0, tlist):
    mcarlo = photocurrent_mesolve(H, psi0, tlist, c_ops=c_ops, sc_ops=sc_ops, e_ops=[], \
                               ntraj=1, nsubsteps=100,store_measurement=True,  \
                               options=Options(atol=1e-10, store_states= True, store_final_state= True))
    last_state= mcarlo.states[0][-1]
    measurement_record= mcarlo.measurement[0]
    return(last_state, measurement_record)



num_traj = 2000
T_end1= 30.0
num_steps1= int(T_end1/0.008)
tlist= np.linspace(0,T_end1,num_steps1)
    
mcarlo22 = photocurrent_mesolve(H, psi0, tlist, c_ops=c_ops, sc_ops=sc_ops, e_ops=[], \
                           ntraj=num_traj, nsubsteps=100,store_measurement=True,  \
                           options=Options(atol=1e-10, store_states= True, store_final_state= True),\
                             map_func= parallel_map)

# Keeps track of the final state for application of pi-pulse
final_state22=[]
for i in range(num_traj):
    final_state22.append(mcarlo22.states[i][-1])
mcarlo22.states=[]



# This function applies a pi-pulse and calculates the probability as described for the ideal case with file name 
# Error_probability_ideal_qubit_ion.py
def main_part(n):
    T_end1= 30.0
    #num_steps1= int(T_end1/0.008)
    #tlist1= np.linspace(0,T_end1,num_steps1)
    
    sigma_plus= tensor(qeye(2),excited_qubit*zero_qubit.dag())
    sigma_x = sigma_plus + sigma_plus.dag() + tensor(qeye(2),one_qubit*one_qubit.dag())
    pi_pulse = spre(sigma_x)*spost(sigma_x.dag())
    psi0= tensor(down_readout, C_0*excited_qubit + C_1*one_qubit).unit()

    indices=[]
    final_state= final_state22[n]
    indices.append(np.where(mcarlo22.measurement[n][:] != [0.0 + 0.0j])[0])
    counter= 1
    num_of_pi_pulses= 1
    for i in range(num_of_pi_pulses):
        psi0= sigma_x*final_state*sigma_x.dag()
        further_results= propagate(psi0, np.linspace(T_end1,2*T_end1,num_steps1))
        T_end1= 2*T_end1
        final_state= further_results[0]
        indi= counter*num_steps1 + (np.where(further_results[1] != [0.0 + 0.0j])[0])
        indices.append(indi)
        counter= counter + 1

        
        
    coll_ops= tensor(down_readout*up_readout.dag(), qeye(3));
    liou= spre(coll_ops)*spost(coll_ops.dag());
    trace_op= np.zeros((36),dtype = float)
    for j in range(36):
        if j%7 == 0:
            trace_op[j]=1;
    trace_op = Qobj(trace_op).dag()   
    trace_op.dims =  [[1],[[2, 3], [2, 3]]] 


    Hamiltonian= H - 0.5j*gamma*coll_ops.dag()*coll_ops
    L1= -1.0j*(spre(Hamiltonian)-spost(Hamiltonian.dag())) 
    L2= gamma_E0* lindblad_dissipator(E0_ops, b=None, data_only=False) 
    L3= gamma_E1* lindblad_dissipator(E1_ops, b=None, data_only=False)
    L= L1 + L2 + L3
    uni= qeye(36)
    uni.dims= [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]
    rho0 = operator_to_vector(ket2dm(tensor(down_readout, excited_qubit).unit()))
    rho1 = operator_to_vector(ket2dm(tensor(down_readout, one_qubit).unit()))   


    indices= list(itertools.chain(*indices))
    end_ins= 30.0
    Time_list= np.linspace(0, counter*end_ins , counter*num_steps1)

    U = uni + L*(Time_list[3]-Time_list[2]) 
    limit= len(Time_list)
    trace_rho0=np.zeros((limit),dtype=complex)
    trace_rho1=np.zeros((limit),dtype=complex)     




    check=1;
    indOld= 0
    for i, ind in enumerate(indices):
        notorious_alexander=0
        t2 = Time_list[ind]
        tlist2= Time_list[indOld:ind]       

        for j, time in enumerate(tlist2):
            check= check+1

            if np.mod(check, num_steps1)==0:
                rho0 = pi_pulse*rho0
                rho1 = pi_pulse*rho1
            rho0 = U*rho0
            rho1 = U*rho1
            trace_rho0[indOld+j] = (trace_op*rho0).tr()
            trace_rho1[indOld+j] = (trace_op*rho1).tr()
            rho0 = rho0/trace_rho0[indOld+j]
            rho1 = rho1/trace_rho1[indOld+j]

            if trace_rho0[indOld+j]<0.0:
                trace_rho0[indOld+j:-1]= 0.0
                trace_rho1[indOld+j:-1]= 1.0
                notorious_alexander= 1.0
                counter= indOld+j
                break


            if trace_rho1[indOld+j] < 0.0:
                trace_rho1[indOld+j:-1]= 0.0
                trace_rho0[indOld+j:-1]= 1.0
                notorious_alexander= 1.0
                counter= indOld+j
                break

        if notorious_alexander==1.0:
            break

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

# Parallelization for ntraj number of trajectories
final_result = parallel_map(main_part, range(num_traj), num_cpus=15, progress_bar=True)

# Average error probability
averaged_result= np.mean(final_result, axis=0)
