# coding: utf-8
import numpy as np
import nengo
import matplotlib
import matplotlib.pyplot as plt
from numpy import loadtxt,savetxt,zeros
import nengo_gui


# 分割frame
def myenframe(y, frameSize, overlap):
    # Calculate step size and get the number of frames
    step = frameSize-overlap
    frameCount = int((len(y)-overlap)/step)
    # Splits given signals into frames
    out = np.zeros((frameSize, frameCount))
    for i in range(0,frameCount):
        startIndex = i*step
        out[:,i] = y[startIndex:(startIndex+frameSize)]
    return out

# 讀取資料
word='coffee';angle='0'
ipem_L=np.loadtxt(word+'_'+str(angle)+'L.txt');ipem_R=np.loadtxt(word+'_'+str(angle)+'R.txt')

# Sliding Window
swL=[];swR=[]
for i in range(0,40):
    chL = ipem_L[:,i];chR = ipem_R[:,i]
    frameL = myenframe(chL,400,200);frameR = myenframe(chR,400,200)
    swL=np.append(swL, np.max(frameL,axis=0));swR=np.append(swR, np.max(frameR,axis=0))



# 讀取 Input data
def getSignal():
    dataL=np.reshape(swL, (40, -1))
    dataR=np.reshape(swR, (40, -1))
    length = len(dataL[0])
    return dataL, dataR, length
LeftSignal, RightSignal, length = getSignal()

T = 22050.00/200

for ch in range(1,41):
    model = nengo.Network(label="LSO_Model")

    # 定義 Input Functions
    def funcL(t, ch):
        index = int(np.floor(t*T))
        if index > length-1:
            return LeftSignal[ch-1][length-1]
        else:
            return LeftSignal[ch-1][index]

    def funcR(t, ch):
        index = int(np.floor(t*T))
        if index > length-1:
            return RightSignal[ch-1][length-1]
        else:
            return RightSignal[ch-1][index]

    # 定義 Decay Functions
    def dbL1(x):
        if x < 5e-2:
            return -2.6
        else:
            return 2*np.log10(x)
    def dbL2(x):
        if x < 5e-2:
            return -2.6-0.1
        else:
            return 2*np.log10(x)-0.1
    def dbL3(x):
        if x < 5e-2:
            return -2.6-0.1
        else:
            return 2*np.log10(x)-0.1
    def dbL4(x):
        if x < 5e-2:
            return -2.6-0.25
        else:
            return 2*np.log10(x)-0.25
    def dbL5(x):
        if x < 5e-2:
            return -2.6-0.4
        else:
            return 2*np.log10(x)-0.35
    def dbL6(x):
        if x < 5e-2:
            return -2.6-0.4
        else:
            return 2*np.log10(x)-0.4
    def dbL7(x):
        if x < 5e-2:
            return -2.6-0.3
        else:
            return 2*np.log10(x)-0.45
    def dbR1(x):
        if x < 5e-2:
            return 2.6
        else:
            return -2*np.log10(x)
    def dbR2(x):
        if x < 5e-2:
            return 2.6+0.1
        else:
            return -2*np.log10(x)+0.1
    def dbR3(x):
        if x < 5e-2:
            return 2.6+0.1
        else:
            return -2*np.log10(x)+0.1
    def dbR4(x):
        if x < 5e-2:
            return 2.6+0.25
        else:
            return -2*np.log10(x)+0.25
    def dbR5(x):
        if x < 5e-2:
            return 2.6+0.4
        else:
            return -2*np.log10(x)+0.35
    def dbR6(x):
        if x < 5e-2:
            return 2.6+0.4
        else:
            return -2*np.log10(x)+0.4
    def dbR7(x):
        if x < 5e-2:
            return 2.6+0.3
        else:
            return -2*np.log10(x)+0.45

    # 定義 LSO output Function
    def LSO_output(x):
        if np.abs(x) <= 0.5:
            return 1.0
        else:
            return 0


    # 建立輸入左右耳 Node
    with model:
        inpL = nengo.Node(lambda t: funcL(t, ch))
        inpR = nengo.Node(lambda t: funcR(t, ch))

        synapse1 = None
        synapse2 = 0
        synapse3 = 0.001
        synapse4 = None


    # 建立 Ensemble
    with model:
        lifRate_model = nengo.LIFRate(tau_rc=0.002, tau_ref=0.0002)

        seed = np.random.seed()
        max_rates=nengo.dists.Uniform(2000, 2000)

        inpL_ens = nengo.Ensemble(300, dimensions=1, radius=5, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        inpR_ens = nengo.Ensemble(300, dimensions=1, radius=5, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)

        ILD1 = nengo.Ensemble(300, dimensions=1, radius=5, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        ILD2 = nengo.Ensemble(300, dimensions=1, radius=5, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        ILD3 = nengo.Ensemble(300, dimensions=1, radius=5, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        ILD4 = nengo.Ensemble(300, dimensions=1, radius=5, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        ILD5 = nengo.Ensemble(300, dimensions=1, radius=5, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        ILD6 = nengo.Ensemble(300, dimensions=1, radius=5, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        ILD7 = nengo.Ensemble(300, dimensions=1, radius=5, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)

        LSO1 = nengo.Ensemble(300, dimensions=1, radius=1, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        LSO2 = nengo.Ensemble(300, dimensions=1, radius=1, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        LSO3 = nengo.Ensemble(300, dimensions=1, radius=1, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        LSO4 = nengo.Ensemble(300, dimensions=1, radius=1, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        LSO5 = nengo.Ensemble(300, dimensions=1, radius=1, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        LSO6 = nengo.Ensemble(300, dimensions=1, radius=1, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        LSO7 = nengo.Ensemble(300, dimensions=1, radius=1, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)


    # 連接 Node, Ensemble
    with model:
        nengo.Connection(inpR, inpR_ens, synapse=synapse1)
        nengo.Connection(inpL, inpL_ens, synapse=synapse1)

        nengo.Connection(inpL_ens, ILD1, function=dbL1, synapse=synapse2)
        nengo.Connection(inpL_ens, ILD2, function=dbL2, synapse=synapse2)
        nengo.Connection(inpL_ens, ILD3, function=dbL3, synapse=synapse2)
        nengo.Connection(inpL_ens, ILD4, function=dbL4, synapse=synapse2)
        nengo.Connection(inpL_ens, ILD5, function=dbL5, synapse=synapse2)
        nengo.Connection(inpL_ens, ILD6, function=dbL6, synapse=synapse2)
        nengo.Connection(inpL_ens, ILD7, function=dbL7, synapse=synapse2)

        nengo.Connection(inpR_ens, ILD1, function=dbR7, synapse=synapse2)
        nengo.Connection(inpR_ens, ILD2, function=dbR6, synapse=synapse2)
        nengo.Connection(inpR_ens, ILD3, function=dbR5, synapse=synapse2)
        nengo.Connection(inpR_ens, ILD4, function=dbR4, synapse=synapse2)
        nengo.Connection(inpR_ens, ILD5, function=dbR3, synapse=synapse2)
        nengo.Connection(inpR_ens, ILD6, function=dbR2, synapse=synapse2)
        nengo.Connection(inpR_ens, ILD7, function=dbR1, synapse=synapse2)

        nengo.Connection(ILD1, LSO1, function=LSO_output, synapse=synapse3)
        nengo.Connection(ILD2, LSO2, function=LSO_output, synapse=synapse3)
        nengo.Connection(ILD3, LSO3, function=LSO_output, synapse=synapse3)
        nengo.Connection(ILD4, LSO4, function=LSO_output, synapse=synapse3)
        nengo.Connection(ILD5, LSO5, function=LSO_output, synapse=synapse3)
        nengo.Connection(ILD6, LSO6, function=LSO_output, synapse=synapse3)
        nengo.Connection(ILD7, LSO7, function=LSO_output, synapse=synapse3)


    # Add Probes
    with model:
        inpL_probe = nengo.Probe(inpL)
        inpR_probe = nengo.Probe(inpR)
        inpL_ens_probe = nengo.Probe(inpL_ens, synapse=synapse4)
        inpR_ens_probe = nengo.Probe(inpR_ens, synapse=synapse4)

        ILD1_probe = nengo.Probe(ILD1, synapse=synapse4)
        ILD2_probe = nengo.Probe(ILD2, synapse=synapse4)
        ILD3_probe = nengo.Probe(ILD3, synapse=synapse4)
        ILD4_probe = nengo.Probe(ILD4, synapse=synapse4)
        ILD5_probe = nengo.Probe(ILD5, synapse=synapse4)
        ILD6_probe = nengo.Probe(ILD6, synapse=synapse4)
        ILD7_probe = nengo.Probe(ILD7, synapse=synapse4)

        LSO1_probe = nengo.Probe(LSO1, synapse=synapse4)
        LSO2_probe = nengo.Probe(LSO2, synapse=synapse4)
        LSO3_probe = nengo.Probe(LSO3, synapse=synapse4)
        LSO4_probe = nengo.Probe(LSO4, synapse=synapse4)
        LSO5_probe = nengo.Probe(LSO5, synapse=synapse4)
        LSO6_probe = nengo.Probe(LSO6, synapse=synapse4)
        LSO7_probe = nengo.Probe(LSO7, synapse=synapse4)

    # Run the Model
    with model:
        dt = 0.001
        sim = nengo.Simulator(model, dt = dt)
        sim.run(length/T)


# Plot the Results
    '''
    matplotlib.rc('axes', labelsize=18)
    matplotlib.rc('legend', fontsize=18)
    maxILD = max(max(sim.data[ILD1_probe]),max(sim.data[ILD7_probe]))+0.25
    minILD = min(min(sim.data[ILD1_probe]),min(sim.data[ILD7_probe]))-0.25
    maxINPUT = max(max(sim.data[inpL_probe]),max(sim.data[inpR_probe]))+0.25

    plt.figure(1)
    plt.plot(sim.trange(), sim.data[inpL_probe], label="Left(inp)")
    plt.plot(sim.trange(), sim.data[inpR_probe], label="Right(inp)")
    plt.ylim([0, maxINPUT]);plt.legend(loc='best')
    plt.xlabel('time');plt.ylabel('spiking rate')
    plt.figure(2)
    plt.subplot(211)
    plt.plot(sim.trange(), sim.data[inpL_ens_probe], label="Left(ens)")
    plt.plot(sim.trange(), sim.data[inpR_ens_probe], label="Right(ens)", color='r')
    plt.ylim([0, maxINPUT])
    plt.legend(loc='best');plt.ylabel('spiking rate')
    plt.subplot(212)
    plt.plot(sim.trange(), sim.data[ILD4_probe], label="ILD4", color='r')
    plt.xlabel('time');plt.legend(loc='best')

    plt.figure(3)
    plt.subplot(711)
    plt.plot(sim.trange(), sim.data[ILD1_probe], label="ILD1")
    plt.ylim([minILD, maxILD]);plt.axhline(0, c='k');plt.legend(loc='best')
    plt.subplot(712)
    plt.plot(sim.trange(), sim.data[ILD2_probe], label="ILD2")
    plt.ylim([minILD, maxILD]);plt.axhline(0, c='k');plt.legend(loc='best')
    plt.subplot(713)
    plt.plot(sim.trange(), sim.data[ILD3_probe], label="ILD3")
    plt.ylim([minILD, maxILD]);plt.axhline(0, c='k');plt.legend(loc='best')
    plt.subplot(714)
    plt.plot(sim.trange(), sim.data[ILD4_probe], label="ILD4")
    plt.ylim([minILD, maxILD]);plt.axhline(0, c='k');plt.legend(loc='best')
    plt.subplot(715)
    plt.plot(sim.trange(), sim.data[ILD5_probe], label="ILD5")
    plt.ylim([minILD, maxILD]);plt.axhline(0, c='k');plt.legend(loc='best')
    plt.subplot(716)
    plt.plot(sim.trange(), sim.data[ILD6_probe], label="ILD6")
    plt.ylim([minILD, maxILD]);plt.axhline(0, c='k');plt.legend(loc='best')
    plt.subplot(717)
    plt.plot(sim.trange(), sim.data[ILD7_probe], label="ILD7")
    plt.ylim([minILD, maxILD]);plt.axhline(0, c='k');plt.legend(loc='best')
    plt.xlabel('time')
    plt.figure(4)
    plt.plot(sim.trange(), sim.data[LSO1_probe], label="LSO1")
    plt.plot(sim.trange(), sim.data[LSO2_probe], label="LSO2")
    plt.plot(sim.trange(), sim.data[LSO3_probe], label="LSO3")
    plt.plot(sim.trange(), sim.data[LSO4_probe], label="LSO4")
    plt.plot(sim.trange(), sim.data[LSO5_probe], label="LSO5")
    plt.plot(sim.trange(), sim.data[LSO6_probe], label="LSO6")
    plt.plot(sim.trange(), sim.data[LSO7_probe], label="LSO7")
    plt.legend(loc='best');plt.xlabel('time');plt.show()
    '''

# 輸出結果
    output_data = np.array([sim.data[LSO1_probe], sim.data[LSO2_probe],\
                            sim.data[LSO3_probe], sim.data[LSO4_probe],\
                            sim.data[LSO5_probe], sim.data[LSO6_probe],\
                            sim.data[LSO7_probe]])
    savetxt('LSO_'+word+'_'+str(angle)+'_ch'+str(ch)+'.txt', output_data, fmt='%.10f')

# Nengo GUI
# nengo_gui.Viz(__file__).start()
