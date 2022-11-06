import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# Using phasic spiking (PS) parameters
Ne = 4000  # Excitatory
Ni = 1000  # Inhibitory
Nq = 20  # Readout

E_L = - 70  # resting membrane potential [mV]
V_th = - 55  # spike threshold [mV]
V_reset = - 75  # value to reset voltage to after a spike [mV]
V_spike = 20  # value to draw a spike to, when cell spikes [mV]
R_m = 10  # membrane resistance [MOhm]
tau = 10  # membrane time constant [ms]

ve = -65  # Excitatory
vi = -65  # Inhibitory
vq = -65  # Readout

# Columns presynaptic, rows postsynaptic
# initialize synapse values to zero, "S" is "Synapse"
# See = np.dot(0.0, np.ones((Ne, Ne)))  # intialize to 0
# Sei = np.dot(0.0, np.ones((Ni, Ne)))  # intialize to 0
# Sie = np.dot(- 0.0, np.ones((Ne, Ni)))  # intialize to 0
Seq = np.dot(0.03, np.ones((Nq, Ne)))  # all synapses are uniform
Siq = np.dot(- 0.008, np.ones((Nq, Ni)))  # things like 0.008 are weights

# initialize to random
# MATAB的 a*b 确实是 np.dot(a,b),也是 a.dot(b)
eei = 0.01
See = np.dot(eei, np.random.rand(Ne, Ne))
Sei = np.dot(eei, np.random.rand(Ni, Ne))
Sie = np.dot(- 0.0, np.ones((Ne, Ni)))

numPools = 25  # Pools: pools of neurons
Nre = Ne / numPools  # neurons per pool in rows
Nce = Ne / numPools  # neurons per pool in columns
Nri = Ni / numPools
Nci = Ni / numPools

## Define the relationships between neurons
# define synapses in ring structure

# excitatory - excitatory weight
ee = 0.12
# excitatory - inhibitory weight
ei = 0.13
# inhibitory - excitatory weight
ie = -0.205

for i in np.arange(numPools - 1):  ###and we should probably not use np.arange() but just normal [1:10,1:5] instead.
    # Add an extra step for each parameter because it asks for int for each index, but all these are floats
    # Synapses _ excitatory - excitatory
    See_row_start = int(1 + Ne - np.dot(((numPools - 1) / numPools), Ne) + np.dot(Nre, (i - 1)))
    See_row_end = int(Ne - np.dot(((numPools - 1) / numPools), Ne) + np.dot(Nre, i))
    See_col_start = int(1 + np.dot(Nce, (i - 1)))
    See_col_end = int(np.dot(Nce, i))
    See[See_row_start:See_row_end, See_col_start:See_col_end] = ee

    # Synapses _ excitatory - inhibitory
    Sei_row_start = int(1 + np.dot(Nri, (i - 1)))
    Sei_row_end = int(np.dot(Nri, i))
    Sei_col_start = int(1 + np.dot(Nce, (i - 1)))
    Sei_col_end = int(np.dot(Nce, i))
    Sei[Sei_row_start:Sei_row_end, Sei_col_start:Sei_col_end] = ei

    # Synapses _ inhibitory - excitatory
    Sie_row_start = int(1 + Ne - np.dot(((numPools - 1) / numPools), Ne) + np.dot(Nre, (i - 1)))
    Sie_row_end = int(Ne - np.dot(((numPools - 1) / numPools), Ne) + np.dot(Nre, i))
    Sie_col_start = int(1 + np.dot(Nci, (i - 1)))
    Sie_col_end = int(np.dot(Nci, i))
    Sie[Sie_row_start:Sie_row_end, Sie_col_start:Sie_col_end] = ie

See[1:int(Nre), int(np.dot((numPools - 1), Nce) + 1):int(np.dot(numPools, Nce))] = ee
Sei[int(np.dot((numPools - 1), Nri) + 1):int(np.dot(numPools, Nri)),
int(np.dot((numPools - 1), Nce) + 1):int(np.dot(numPools, Nce))] = ei
Sie[1:int(Nre), int(np.dot((numPools - 1), Nci) + 1):int(np.dot(numPools, Nci))] = ie

## Variable and parameter initialization.
# Don't need to change anything here

simtime = 1000  # ms
# simtime=300 # ms

# list of time points when the neurons in this group (excitatory, inhibitory, readout) spike
firingse = []
firingsi = []
firingsq = []

dt = 1

# voltage value variables for plotting
vq_view = np.zeros((Nq, int(simtime / dt)))
ve_view = np.zeros((Ne, int(simtime / dt)))
vi_view = np.zeros((Ni, int(simtime / dt)))
uq_view = np.zeros((1, int(simtime / dt)))

AMPA = 0.5
GABAa = 0.5

# build the postsynaptic voltage functions
# AMPA: fast, ionotropic, excitatory
# NMDA: slow, ionotropic, excitatory
# GABAa:fast, ionotropic, inhibitory
# GABAb:metabotropic, inhibitory

NMDAe = np.arange(0, (100 / dt))
NMDAe = 0.5 * np.exp(-NMDAe * dt * 0.7) + 0.5 * np.exp(-NMDAe * dt * 0.05)
AMPAe = np.arange(0, (100 / dt))
AMPAe = np.exp(-AMPAe * dt * 0.7)
GABAae = np.arange(0, (100 / dt))
GABAae = 0.5 * np.exp(-GABAae * dt * 0.7) + 0.5 * np.exp(-GABAae * dt * 0.05)
GABAbe = np.arange(0, (100 / dt))
GABAbe = np.exp(-GABAbe * dt * 0.7)

# values for postsynaptic potential changes
S_E = np.dot(AMPA, AMPAe) + np.dot((1 - AMPA), NMDAe)
S_I = np.dot(GABAa, GABAae) + np.dot((1 - GABAa), GABAbe)

# plt.plot(S_E, linewidth=3)
# plt.xlim(0, 100)
# plt.title('Postsynaptic Potential')
# plt.xlabel('Time(ms)')
# plt.ylabel('Synaptic Output')
# plt.show()

## Synaptic delays balance the overall network
# synaptic delays in ms
ex_syn_delay = Ne + 100
maxdelay = 35
delaye = [5, 30]
delayi = [1, 15]

ax_delaye = np.random.randint(delaye[0], delaye[1], (Ne, Ne))
ax_delayi = np.random.randint(delayi[0], delayi[1], (Ne, Ni))

# current/input variables
Ie = np.zeros((Ne, int(simtime / dt + np.shape(NMDAe)[0] + maxdelay / dt)))
Ii = np.zeros((Ni, int(simtime / dt + np.shape(NMDAe)[0] + maxdelay / dt)))
Iq = np.zeros((Nq, int(simtime / dt + np.shape(NMDAe)[0] + maxdelay / dt)))

## Main Body
# New inputs are determined by the sum of excitatory and inhibitory outputs
# of fired neurons scaled by the synaptic weights.

ve_tmp = np.empty((0, 1))
vi_tmp = np.empty((0, 1))
vq_tmp = np.empty((0, 1))

firingse = np.empty((0, 2))
firingsi = np.empty((0, 2))
firingsq = np.empty((0, 2))

firingse_update_mean = []

firede = []
firedi = []
firedq = []


for t in np.arange(1, (np.dot((1 / dt), simtime))):  # 1000 ms
    # define the external input into the first excitatory population
    t = int(t)
    if 80 / dt > t > 40:
        # Input into separate populations with normal distribution
        Ie[np.arange(int(Ne / numPools)), t] = np.dot(1.5, np.ones((int(Ne / numPools))))
    # add noise into inputs
    # print(np.random.randn(Ne, 1))
    # print(np.shape(Ie[:, t]))
    # print(np.shape(np.dot(1, np.random.randn(Ne, 1))))
    # print(np.shape(Ie[:, t] + np.dot(1, np.random.randn(Ne, 1))))

    rand_Ne = np.dot(1, np.random.randn(Ne, 1)).reshape((Ne,))
    rand_Ni = np.dot(1, np.random.randn(Ni, 1)).reshape((Ni,))
    rand_Nq = np.dot(1, np.random.randn(Nq, 1)).reshape((Nq,))
    Ie[:, t] = Ie[:, t] + rand_Ne
    Ii[:, t] = Ii[:, t] + rand_Ni
    Iq[:, t] = Iq[:, t] + rand_Nq

    # Uncomment this to have an impulse input for 'active readout'
    #   if t > 200 && t < 250
    #       Iq(:,t) = Iq(:,t) + 1*ones(Nq,1)
    #   end

    # indices of spikes
    # THIS MIGHT BE WRONG, NEED TO CHECK if bugs happen
    # print(ve)
    # ve 在第一个for loop里是一个variable，而且没用，但是后来就变成array了，所以第一个for loop里concat的时候会出错，因为不是想要的array
    # !!! ERROR: Ve_tmp太大了，所有的1-4000都入选了，要看看variable怎么出错了
    firede = np.nonzero(ve_tmp >= V_th)[0]
    firedi = np.nonzero(vi_tmp >= V_th)[0]
    firedq = np.nonzero(vq_tmp >= V_th)[0]

    # firede= np.nonzero(ve_tmp >= V_th)
    # firedi= np.nonzero(vi_tmp >= V_th)
    # firedq= np.nonzero(vq_tmp >= V_th)

    # firing? Or what does this mean?
    # firingse=np.concatenate([firingse,[t + np.dot(0,firede), firede]])
    # print(t)
    # print(firede.ndim)

    indices_firingse_tmp = (t + np.dot(0, firede))[:, np.newaxis]
    neurons_firingse_tmp = firede[:, np.newaxis]
    firingse_tmp = np.append(indices_firingse_tmp, neurons_firingse_tmp, axis=1)

    # if t == 50:
    # print("indices_firingse_tmp")
    # print(np.shape(indices_firingse_tmp))
    # print("neurons_firingse_tmp")
    # print(np.shape(neurons_firingse_tmp))
    # print(firingse_tmp)
    # print(firingse_tmp)

    indices_firingsi_tmp = (t + np.dot(0, firedi))[:, np.newaxis]
    neurons_firingsi_tmp = firedi[:, np.newaxis]
    firingsi_tmp = np.append(indices_firingsi_tmp, neurons_firingsi_tmp, axis=1)

    indices_firingsq_tmp = (t + np.dot(0, firedq))[:, np.newaxis]
    neurons_firingsq_tmp = firedq[:, np.newaxis]
    firingsq_tmp = np.append(indices_firingsq_tmp, neurons_firingsq_tmp, axis=1)

    # firingse_tmp = np.append(t + np.dot(0,firede), firede,axis=1)
    # firingsi_tmp = np.append(t + np.dot(0,firedi), firedi,axis=1)
    # firingsq_tmp = np.append(t + np.dot(0,firedq), firedq,axis=1)

    # Why do we have errors here...?
    if firingse_tmp.size != 0:
        firingse = np.append(firingse, firingse_tmp, axis=0)
        # firingse = np.append(firingse, firingse_tmp, axis=0)[0]
        # firingse = np.concatenate([[firingse],[t + np.dot(0,firedi), firede]])
    # print(np.shape(firingse))
    if firingsi_tmp.size != 0:
        firingsi = np.append(firingsi, firingsi_tmp, axis=0)
        # firingsi = np.append(firingsi, firingsi_tmp, axis=0)[0]
        # firingsi = np.concatenate([[firingsi], [t + np.dot(0, firedi), firedi]])
    if firingsq_tmp.size != 0:
        # firingsq = np.append(firingsq, firingsq_tmp, axis=0)[0]
        firingsq = np.append(firingsq, firingsq_tmp, axis=0)
        # firingsq = np.concatenate([[firingsq], [t + np.dot(0, firedq), firedq]])

    # firingse=np.concatenate([[firingse],[t + np.dot(0,firedi), firede]])
    # firingsi=np.concatenate([[firingsi],[t + np.dot(0,firedi), firedi]])
    # firingsq=np.concatenate([[firingsq],[t + np.dot(0,firedq), firedq]])

    ve_tmp[firede] = V_reset
    vi_tmp[firedi] = V_reset
    vq_tmp[firedq] = V_reset
    # start_time = time.time()
    if (
            firede.size != 0):  # If firede is not empty #!!! ERROR: updates here might be wrong, causing long running time and wrong data

        for i in np.arange(1, np.size(firede)):
            # Calculate Input Currents
            n = firede[i]

            for j in np.arange(1, Ne):
                # 100 is used because that is the length of the synaptic
                # exponential matrix
                z = ax_delaye[j, n]
                Ie[j, t + z: t + z + S_E.size] = Ie[j, t + z: t + z + S_E.size] + np.dot(S_E, See[j, n])

                # if t == 1:
                # print(np.dot(S_E, See[j, n]))
                # print('for loop 1 ' + str(time.time() - start_time))
            # inhibitory neurons
            for j in np.arange(1, Ni):
                z = ax_delaye[j, n]
                Ii[j, t + z:t + z + S_I.size] = Ii[j, t + z:t + z + S_I.size] + np.dot(S_I, Sei[j, n])

                # print('for loop 2 ' + str(time.time() - start_time))
            # downstream second population
            for j in np.arange(1, Nq):
                Iq[j, t + z:t + z + S_E.size] = Iq[j, t + z:t + z + S_E.size] + np.dot(S_E, Seq[j, n])
                # print('for loop 3 ' + str(time.time() - start_time))

    # print("--- %s seconds ---" % (time.time() - start_time))

    # for every fired inhibitory neuron, add the IPSPs to the input variables
    if (firedi.size != 0):
        for i in np.arange(1, np.size(firedi)):
            n = firedi[i]
            for j in np.arange(1, Ne):
                z = ax_delayi[j, n]
                Ie[j, t + z: t + z + S_I.size] = Ie[j, t + z: t + z + S_I.size] + np.dot(S_I, Sie[j, n])
                # print(np.dot(S_I, Sie[j, n]))

            for j in np.arange(1, Nq):
                Iq[j, t + z:t + z + S_I.size] = Iq[j, t + z:t + z + S_I.size] + np.dot(S_I, Siq[j, n])

    # !!! ERROR: np.dot(Ie[:,int(t)], R_m) is different in MATLAB and in Python, but changing it to a significantly larger value
    # does not change the results
    # V_infi is the same as V_infe, so is V_infq. So there's something wrong with the update rules in here...
    V_infe = E_L + np.dot(Ie[:, int(t)], R_m)
    # !!! CHECK THE INFI AND INFQ AS WELL?
    V_infi = E_L + np.dot(Ii[:, int(t)], R_m)
    V_infq = E_L + np.dot(Iq[:, int(t)], R_m)

    firingse_update_mean = np.append(firingse_update_mean, np.mean(np.dot(Ie[:, int(t)], R_m)))

    # print(ve)
    if t == 1:
        ve_tmp = V_infe + np.dot((ve - V_infe), np.exp(- dt / tau))
        vi_tmp = V_infi + np.dot((ve - V_infi), np.exp(- dt / tau))
        vq_tmp = V_infq + np.dot((ve - V_infq), np.exp(- dt / tau))
    else:
        # print(t)
        ve_tmp = V_infe + np.dot((ve_tmp - V_infe), np.exp(- dt / tau))
        vi_tmp = V_infi + np.dot((vi_tmp - V_infi), np.exp(- dt / tau))
        vq_tmp = V_infq + np.dot((vq_tmp - V_infq), np.exp(- dt / tau))
    # ve_tmp = V_infe + np.dot((ve_tmp - V_infe),np.exp(- dt / tau))
    # vi_tmp = V_infi + np.dot((vi_tmp - V_infi),np.exp(- dt / tau))
    # vq_tmp = V_infq + np.dot((vq_tmp - V_infq),np.exp(- dt / tau))

    # print(np.mean(ve_tmp))

    ve_view[:, int(t)] = ve_tmp
    vi_view[:, int(t)] = vi_tmp
    vq_view[:, int(t)] = vq_tmp
    # !!! PROBLEM: the energy transferred to the downstream population is not correct, because the firing times are not correct (see above)
    # !!! PROBLEM： the energy transferred to the downstream population is not correct, too small, not making the other
    # !!! PROBLEM: the neurons can't fire when they reach -70. Need to find the reason why their weights go down.
    # neurons have enough energy to fire.


## Plot Output

plt.plot(S_E, linewidth=3)
plt.xlim(0, 100)
plt.title('Postsynaptic Potential')
plt.xlabel('Time(ms)')
plt.ylabel('Synaptic Output')
plt.show()

figure, (ax1, ax2, ax3, ax4) = plt.subplots(4)
ax1.plot(firingse[:, 0], firingse[:, 1], '.')
print(np.shape(firingse[:, 0]), np.shape(firingse[:, 1]))

# ax1.ylabel('Neuron index')

ax1.set(ylabel='Neuron index')
ax1.set_title('Excitatory Neurons Ordered')

ax2.plot(firingse[np.random.permutation(len(firingse)), 0], firingse[:, 1], '.')
ax2.set_title('Excitatory Neurons Shuffled')

ax3.plot(firingsi[:, 0], firingsi[:, 1], '.r')
ax3.set_title('Inhibitory Neurons Ordered')
print(np.shape(firingsi[:, 0]), np.shape(firingsi[:, 1]))

ax4.plot(firingsi[np.random.permutation(len(firingsi)), 0], firingsi[:, 1], '.r')
ax4.set_title('Inhibitory Neurons Shuffled')
ax4.set(xlabel='Time (ms)')
plt.show()

# subplot(3,1,1);
# plot(Iq(1,1:simtime/dt)); title('Current(mA)');
figure, (ax1, ax2) = plt.subplots(2)

ax1.plot(vq_view.T)
ax1.set_title('Effect of Synaptic Strength')
ax1.set(xlabel='Time(ms)', ylabel='Membrane Potential (mV)')

ax2.plot(firingsq[:, 0], firingsq[:, 1], '.k')
ax2.set_title('Downstream Neurons')
ax2.set_xlim((0, 1000))
ax2.set_ylim((0, Nq + 1))
ax1.set(xlabel='Time(ms)', ylabel='Neuron Index')
plt.show()

# avgdelay = [sum(sum(ax_delaye))/prod(size(ax_delaye)),sum(sum(ax_delayi))/prod(size(ax_delayi))]



