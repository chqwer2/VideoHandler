# coding: utf-8
import numpy as np
import nengo
import matplotlib.pyplot as plt
from numpy import loadtxt,savetxt,zeros
import nengo_gui

# 讀取 Input data
word='coffee';angle='0'
def getinput(ch):
    dataMSO = np.loadtxt('MSO'+'/MSO_'+word+'_'+str(angle)+'_ch'+str(ch)+'.txt')
    dataLSO = np.loadtxt('LSO'+'/LSO_'+word+'_'+str(angle)+'_ch'+str(ch)+'.txt')
    lengthMSO = len(dataMSO[0]);lengthLSO = len(dataLSO[0])
    return dataMSO, dataLSO, lengthMSO, lengthLSO

T = 1000.00
MSO_W = np.append(10*np.ones(18), 5*np.ones(22))
LSO_W = np.append(np.zeros(18), np.ones(22))

for ch in range(1,41):
    MSOSignal, LSOSignal, lengthMSO, lengthLSO = getinput(ch)

    model = nengo.Network(label="IC_Model")

    # 定義 Input Functions
    def funcMSO(t, angle):
        index = int(np.floor(t*T))
        if index > lengthMSO-1:
            return MSOSignal[angle-1][lengthMSO-1]
        else:
            return MSOSignal[angle-1][index]

    def funcLSO(t, angle):
        index = int(np.floor(t*T))
        if index > lengthLSO-1:
            return LSOSignal[angle-1][lengthLSO-1]
        else:
            return LSOSignal[angle-1][index]

    # 定義 Weight Functions
    def funcWeight(x):
        inp_MSO = x[0]*MSO_W[ch-1]
        inp_LSO = x[1]*LSO_W[ch-1]
        if LSO_W[ch-1]==0:
            return inp_MSO
        else:
            return inp_MSO*inp_LSO


    # 建立輸入 MSO/LSO Node
    with model:
        MSO270 = nengo.Node(lambda t: funcMSO(t, 1))
        MSO300 = nengo.Node(lambda t: funcMSO(t, 2))
        MSO330 = nengo.Node(lambda t: funcMSO(t, 3))
        MSO0 = nengo.Node(lambda t: funcMSO(t, 4))
        MSO30 = nengo.Node(lambda t: funcMSO(t, 5))
        MSO60 = nengo.Node(lambda t: funcMSO(t, 6))
        MSO90 = nengo.Node(lambda t: funcMSO(t, 7))

        LSO270 = nengo.Node(lambda t: funcLSO(t, 1))
        LSO300 = nengo.Node(lambda t: funcLSO(t, 2))
        LSO330 = nengo.Node(lambda t: funcLSO(t, 3))
        LSO0 = nengo.Node(lambda t: funcLSO(t, 4))
        LSO30 = nengo.Node(lambda t: funcLSO(t, 5))
        LSO60 = nengo.Node(lambda t: funcLSO(t, 6))
        LSO90 = nengo.Node(lambda t: funcLSO(t, 7))

        synapse1 = None
        synapse2 = 0
        synapse3 = None


    # 建立 Ensemble
    with model:
        lifRate_model = nengo.LIFRate(tau_rc=0.002, tau_ref=0.0002)

        seed = np.random.seed()
        max_rates=nengo.dists.Uniform(2000, 2000)

        radius1=3
        radius2=10

        inp270_ens = nengo.Ensemble(1000, dimensions=2, radius=radius1, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        inp300_ens = nengo.Ensemble(1000, dimensions=2, radius=radius1, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        inp330_ens = nengo.Ensemble(1000, dimensions=2, radius=radius1, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        inp0_ens = nengo.Ensemble(1000, dimensions=2, radius=radius1, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        inp30_ens = nengo.Ensemble(1000, dimensions=2, radius=radius1, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        inp60_ens = nengo.Ensemble(1000, dimensions=2, radius=radius1, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        inp90_ens = nengo.Ensemble(1000, dimensions=2, radius=radius1, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)

        IC270_ens = nengo.Ensemble(1000, dimensions=1, radius=radius2, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        IC300_ens = nengo.Ensemble(1000, dimensions=1, radius=radius2, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        IC330_ens = nengo.Ensemble(1000, dimensions=1, radius=radius2, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        IC0_ens = nengo.Ensemble(1000, dimensions=1, radius=radius2, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        IC30_ens = nengo.Ensemble(1000, dimensions=1, radius=radius2, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        IC60_ens = nengo.Ensemble(1000, dimensions=1, radius=radius2, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        IC90_ens = nengo.Ensemble(1000, dimensions=1, radius=radius2, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)

    # 連接 Node, Ensemble
    with model:
        nengo.Connection(MSO270, inp270_ens[0], synapse=synapse1)
        nengo.Connection(MSO300, inp300_ens[0], synapse=synapse1)
        nengo.Connection(MSO330, inp330_ens[0], synapse=synapse1)
        nengo.Connection(MSO0, inp0_ens[0], synapse=synapse1)
        nengo.Connection(MSO30, inp30_ens[0], synapse=synapse1)
        nengo.Connection(MSO60, inp60_ens[0], synapse=synapse1)
        nengo.Connection(MSO90, inp90_ens[0], synapse=synapse1)

        nengo.Connection(LSO270, inp270_ens[1], synapse=synapse1)
        nengo.Connection(LSO300, inp300_ens[1], synapse=synapse1)
        nengo.Connection(LSO330, inp330_ens[1], synapse=synapse1)
        nengo.Connection(LSO0, inp0_ens[1], synapse=synapse1)
        nengo.Connection(LSO30, inp30_ens[1], synapse=synapse1)
        nengo.Connection(LSO60, inp60_ens[1], synapse=synapse1)
        nengo.Connection(LSO90, inp90_ens[1], synapse=synapse1)

        nengo.Connection(inp270_ens, IC270_ens, function=funcWeight, synapse=synapse2)
        nengo.Connection(inp300_ens, IC300_ens, function=funcWeight, synapse=synapse2)
        nengo.Connection(inp330_ens, IC330_ens, function=funcWeight, synapse=synapse2)
        nengo.Connection(inp0_ens, IC0_ens, function=funcWeight, synapse=synapse2)
        nengo.Connection(inp30_ens, IC30_ens, function=funcWeight, synapse=synapse2)
        nengo.Connection(inp60_ens, IC60_ens, function=funcWeight, synapse=synapse2)
        nengo.Connection(inp90_ens, IC90_ens, function=funcWeight, synapse=synapse2)

    # Add Probes
    with model:
        '''
        MSO270_probe = nengo.Probe(MSO270)
        MSO300_probe = nengo.Probe(MSO300)
        MSO330_probe = nengo.Probe(MSO330)
        MSO0_probe = nengo.Probe(MSO0)
        MSO30_probe = nengo.Probe(MSO30)
        MSO60_probe = nengo.Probe(MSO60)
        MSO90_probe = nengo.Probe(MSO90)
        LSO270_probe = nengo.Probe(LSO270)
        LSO300_probe = nengo.Probe(LSO300)
        LSO330_probe = nengo.Probe(LSO330)
        LSO0_probe = nengo.Probe(LSO0)
        LSO30_probe = nengo.Probe(LSO30)
        LSO60_probe = nengo.Probe(LSO60)
        LSO90_probe = nengo.Probe(LSO90)
        '''

        IC270_probe = nengo.Probe(IC270_ens, synapse=synapse3)
        IC300_probe = nengo.Probe(IC300_ens, synapse=synapse3)
        IC330_probe = nengo.Probe(IC330_ens, synapse=synapse3)
        IC0_probe = nengo.Probe(IC0_ens, synapse=synapse3)
        IC30_probe = nengo.Probe(IC30_ens, synapse=synapse3)
        IC60_probe = nengo.Probe(IC60_ens, synapse=synapse3)
        IC90_probe = nengo.Probe(IC90_ens, synapse=synapse3)


    # Run the Model
    with model:
        dt = 0.001
        sim = nengo.Simulator(model, dt = dt)
        sim.run(max(lengthMSO,lengthLSO)/T)

    output_data = np.array([sim.data[IC270_probe], sim.data[IC300_probe], \
                            sim.data[IC330_probe], sim.data[IC0_probe], \
                            sim.data[IC30_probe], sim.data[IC60_probe], \
                            sim.data[IC90_probe]])
    savetxt('IC_'+word+'_'+str(angle)+'_ch'+str(ch)+'.txt', output_data, fmt='%.10f')


# Plot the Results
    '''
    plt.figure()
    plt.plot(sim.trange(), sim.data[MSO270_probe], label="270")
    plt.plot(sim.trange(), sim.data[MSO300_probe], label="300")
    plt.plot(sim.trange(), sim.data[MSO330_probe], label="330")
    plt.plot(sim.trange(), sim.data[MSO0_probe], label="0")
    plt.plot(sim.trange(), sim.data[MSO30_probe], label="30")
    plt.plot(sim.trange(), sim.data[MSO60_probe], label="60")
    plt.plot(sim.trange(), sim.data[MSO90_probe], label="90")
    plt.legend(loc='best');plt.xlabel('time');plt.title('MSO')
    plt.figure()
    plt.plot(sim.trange(), sim.data[LSO270_probe], label="270")
    plt.plot(sim.trange(), sim.data[LSO300_probe], label="300")
    plt.plot(sim.trange(), sim.data[LSO330_probe], label="330")
    plt.plot(sim.trange(), sim.data[LSO0_probe], label="0")
    plt.plot(sim.trange(), sim.data[LSO30_probe], label="30")
    plt.plot(sim.trange(), sim.data[LSO60_probe], label="60")
    plt.plot(sim.trange(), sim.data[LSO90_probe], label="90")
    plt.legend(loc='best');plt.xlabel('time');plt.title('LSO')
    plt.figure()
    plt.plot(sim.trange(), sim.data[IC270_probe], label="270")
    plt.plot(sim.trange(), sim.data[IC300_probe], label="300")
    plt.plot(sim.trange(), sim.data[IC330_probe], label="330")
    plt.plot(sim.trange(), sim.data[IC0_probe], label="0")
    plt.plot(sim.trange(), sim.data[IC30_probe], label="30")
    plt.plot(sim.trange(), sim.data[IC60_probe], label="60")
    plt.plot(sim.trange(), sim.data[IC90_probe], label="90")
    plt.legend(loc='best');plt.xlabel('time');plt.title('From '+str(angle)+'-sound '+'in Ch.'+str(ch))
    plt.show()
    '''

# Nengo GUI
# nengo_gui.Viz(__file__).start()