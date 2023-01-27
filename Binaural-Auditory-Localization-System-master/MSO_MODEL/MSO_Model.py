# coding: utf-8
import numpy as np
import nengo
import matplotlib
import matplotlib.pyplot as plt
from numpy import loadtxt,savetxt,zeros
import nengo_gui

# 讀取 Input data
word='coffee';angle='0'
def getSignal():
    dataL = np.loadtxt(word+'_'+str(angle)+'L.txt')
    dataR = np.loadtxt(word+'_'+str(angle)+'R.txt')
    dataL = dataL.T;dataR = dataR.T
    length = len(dataL[0])
    return dataL, dataR, length
LeftSignal, RightSignal, length = getSignal()

T = 22050.00

for ch in range(1,41):
    model = nengo.Network(label="MSO_Jeffress_Model")

    # 定義 Input Functions
    def funcL(t, ch):
        index = int(np.floor(t*T))
        if index > length-1:
            return 0
        return LeftSignal[ch-1][index]

    def funcR(t, ch):
        index = int(np.floor(t*T))
        if index > length-1:
            return 0
        return RightSignal[ch-1][index]



    # 建立輸入左右耳 Node
    with model:
        inpL = nengo.Node(lambda t: funcL(t, ch))
        inpR = nengo.Node(lambda t: funcR(t, ch))

        synapse1 = None
        synapse2 = 0
        synapse3 = 0
        synapse4 = None

    # Delay
    class Delay(object):
        def __init__(self, dimensions, timesteps=50):
            self.history = 0.05*np.ones((timesteps, dimensions))
        def step(self, t, x):
            self.history = np.roll(self.history, -1)
            self.history[-1] = x
            return self.history[0]

    # 建立 Dalay Node
    with model:
        delay = Delay(1, timesteps=int(9))
        delaynodeLA = nengo.Node(delay.step, size_in=1, size_out=1)
        delay = Delay(1, timesteps=int(17))
        delaynodeLB = nengo.Node(delay.step, size_in=1, size_out=1)
        delay = Delay(1, timesteps=int(15))
        delaynodeLC = nengo.Node(delay.step, size_in=1, size_out=1)
        delay = Delay(1, timesteps=int(15))
        delaynodeLD = nengo.Node(delay.step, size_in=1, size_out=1)
        delay = Delay(1, timesteps=int(17))
        delaynodeLE = nengo.Node(delay.step, size_in=1, size_out=1)
        delay = Delay(1, timesteps=int(9))
        delaynodeLF = nengo.Node(delay.step, size_in=1, size_out=1)

        delay = Delay(1, timesteps=int(9))
        delaynodeRA = nengo.Node(delay.step, size_in=1, size_out=1)
        delay = Delay(1, timesteps=int(17))
        delaynodeRB = nengo.Node(delay.step, size_in=1, size_out=1)
        delay = Delay(1, timesteps=int(15))
        delaynodeRC = nengo.Node(delay.step, size_in=1, size_out=1)
        delay = Delay(1, timesteps=int(15))
        delaynodeRD = nengo.Node(delay.step, size_in=1, size_out=1)
        delay = Delay(1, timesteps=int(17))
        delaynodeRE = nengo.Node(delay.step, size_in=1, size_out=1)
        delay = Delay(1, timesteps=int(9))
        delaynodeRF = nengo.Node(delay.step, size_in=1, size_out=1)

    # 連接 Dalay Node
    with model:
        nengo.Connection(inpL, delaynodeLA, synapse=synapse1)
        nengo.Connection(delaynodeLA, delaynodeLB, synapse=synapse1)
        nengo.Connection(delaynodeLB, delaynodeLC, synapse=synapse1)
        nengo.Connection(delaynodeLC, delaynodeLD, synapse=synapse1)
        nengo.Connection(delaynodeLD, delaynodeLE, synapse=synapse1)
        nengo.Connection(delaynodeLE, delaynodeLF, synapse=synapse1)

        nengo.Connection(inpR, delaynodeRA, synapse=synapse1)
        nengo.Connection(delaynodeRA, delaynodeRB, synapse=synapse1)
        nengo.Connection(delaynodeRB, delaynodeRC, synapse=synapse1)
        nengo.Connection(delaynodeRC, delaynodeRD, synapse=synapse1)
        nengo.Connection(delaynodeRD, delaynodeRE, synapse=synapse1)
        nengo.Connection(delaynodeRE, delaynodeRF, synapse=synapse1)

    # 建立 Ensemble
    with model:
        radius = 1
        radius2 = 2
        radius3 = 5

        lifRate_model = nengo.LIFRate(tau_rc=0.002, tau_ref=0.0002)

        seed = np.random.seed()
        max_rates=nengo.dists.Uniform(2000, 2000)

        DelayL_ens1 = nengo.Ensemble(300, dimensions=1, radius=radius, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        DelayL_ens2 = nengo.Ensemble(300, dimensions=1, radius=radius, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        DelayL_ens3 = nengo.Ensemble(300, dimensions=1, radius=radius, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        DelayL_ens4 = nengo.Ensemble(300, dimensions=1, radius=radius, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        DelayL_ens5 = nengo.Ensemble(300, dimensions=1, radius=radius, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        DelayL_ens6 = nengo.Ensemble(300, dimensions=1, radius=radius, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        DelayL_ens7 = nengo.Ensemble(300, dimensions=1, radius=radius, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)

        DelayR_ens1 = nengo.Ensemble(300, dimensions=1, radius=radius, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        DelayR_ens2 = nengo.Ensemble(300, dimensions=1, radius=radius, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        DelayR_ens3 = nengo.Ensemble(300, dimensions=1, radius=radius, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        DelayR_ens4 = nengo.Ensemble(300, dimensions=1, radius=radius, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        DelayR_ens5 = nengo.Ensemble(300, dimensions=1, radius=radius, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        DelayR_ens6 = nengo.Ensemble(300, dimensions=1, radius=radius, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        DelayR_ens7 = nengo.Ensemble(300, dimensions=1, radius=radius, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)

        adder1 = nengo.Ensemble(300, dimensions=1, radius=radius2, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        adder2 = nengo.Ensemble(300, dimensions=1, radius=radius2, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        adder3 = nengo.Ensemble(300, dimensions=1, radius=radius2, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        adder4 = nengo.Ensemble(300, dimensions=1, radius=radius2, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        adder5 = nengo.Ensemble(300, dimensions=1, radius=radius2, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        adder6 = nengo.Ensemble(300, dimensions=1, radius=radius2, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)
        adder7 = nengo.Ensemble(300, dimensions=1, radius=radius2, neuron_type = lifRate_model, max_rates=max_rates, seed=seed)


    # 連接 Node, Ensemble
    with model:
        nengo.Connection(inpL, DelayL_ens1, synapse=synapse2)
        nengo.Connection(delaynodeLA, DelayL_ens2, synapse=synapse2)
        nengo.Connection(delaynodeLB, DelayL_ens3, synapse=synapse2)
        nengo.Connection(delaynodeLC, DelayL_ens4, synapse=synapse2)
        nengo.Connection(delaynodeLD, DelayL_ens5, synapse=synapse2)
        nengo.Connection(delaynodeLE, DelayL_ens6, synapse=synapse2)
        nengo.Connection(delaynodeLF, DelayL_ens7, synapse=synapse2)

        nengo.Connection(delaynodeRF, DelayR_ens7, synapse=synapse2)
        nengo.Connection(delaynodeRE, DelayR_ens6, synapse=synapse2)
        nengo.Connection(delaynodeRD, DelayR_ens5, synapse=synapse2)
        nengo.Connection(delaynodeRC, DelayR_ens4, synapse=synapse2)
        nengo.Connection(delaynodeRB, DelayR_ens3, synapse=synapse2)
        nengo.Connection(delaynodeRA, DelayR_ens2, synapse=synapse2)
        nengo.Connection(inpR, DelayR_ens1, synapse=synapse2)

    with model:
        nengo.Connection(DelayL_ens1, adder1, synapse=synapse3)
        nengo.Connection(DelayL_ens2, adder2, synapse=synapse3)
        nengo.Connection(DelayL_ens3, adder3, synapse=synapse3)
        nengo.Connection(DelayL_ens4, adder4, synapse=synapse3)
        nengo.Connection(DelayL_ens5, adder5, synapse=synapse3)
        nengo.Connection(DelayL_ens6, adder6, synapse=synapse3)
        nengo.Connection(DelayL_ens7, adder7, synapse=synapse3)

        nengo.Connection(DelayR_ens7, adder1, synapse=synapse3)
        nengo.Connection(DelayR_ens6, adder2, synapse=synapse3)
        nengo.Connection(DelayR_ens5, adder3, synapse=synapse3)
        nengo.Connection(DelayR_ens4, adder4, synapse=synapse3)
        nengo.Connection(DelayR_ens3, adder5, synapse=synapse3)
        nengo.Connection(DelayR_ens2, adder6, synapse=synapse3)
        nengo.Connection(DelayR_ens1, adder7, synapse=synapse3)

    # Add Probes
    with model:
        '''
        inpL_probe = nengo.Probe(inpL)
        LA_probe = nengo.Probe(delaynodeLA)
        LB_probe = nengo.Probe(delaynodeLB)
        LC_probe = nengo.Probe(delaynodeLC)
        LD_probe = nengo.Probe(delaynodeLD)
        LE_probe = nengo.Probe(delaynodeLE)
        LF_probe = nengo.Probe(delaynodeLF)

        inpR_probe = nengo.Probe(inpR)
        RA_probe = nengo.Probe(delaynodeRA)
        RB_probe = nengo.Probe(delaynodeRB)
        RC_probe = nengo.Probe(delaynodeRC)
        RD_probe = nengo.Probe(delaynodeRD)
        RE_probe = nengo.Probe(delaynodeRE)
        RF_probe = nengo.Probe(delaynodeRF)
        '''

        adder1_probe = nengo.Probe(adder1, synapse=synapse4)
        adder2_probe = nengo.Probe(adder2, synapse=synapse4)
        adder3_probe = nengo.Probe(adder3, synapse=synapse4)
        adder4_probe = nengo.Probe(adder4, synapse=synapse4)
        adder5_probe = nengo.Probe(adder5, synapse=synapse4)
        adder6_probe = nengo.Probe(adder6, synapse=synapse4)
        adder7_probe = nengo.Probe(adder7, synapse=synapse4)


    # Run the Model
    with model:
        dt = 0.00001
        sim = nengo.Simulator(model, dt = dt)
        sim.run(length/T)


# Plot the Results
    '''
    plt.figure(1)
    plt.subplot(7, 1, 1)
    plt.plot(sim.trange(), sim.data[inpL_probe], label="Left")
    plt.plot(sim.trange(), sim.data[RF_probe], label="Right")
    plt.axhline(0, c='k');plt.title("Delayed1 input");plt.legend(loc='best')
    plt.subplot(7, 1, 2)
    plt.plot(sim.trange(), sim.data[LA_probe], label="Left")
    plt.plot(sim.trange(), sim.data[RE_probe], label="Right")
    plt.axhline(0, c='k');plt.title("Delayed1 input");plt.legend(loc='best')
    plt.subplot(7, 1, 3)
    plt.plot(sim.trange(), sim.data[LB_probe], label="Left")
    plt.plot(sim.trange(), sim.data[RD_probe], label="Right")
    plt.axhline(0, c='k');plt.title("Delayed1 input");plt.legend(loc='best')
    plt.subplot(7, 1, 4)
    plt.plot(sim.trange(), sim.data[LC_probe], label="Left")
    plt.plot(sim.trange(), sim.data[RC_probe], label="Right")
    plt.axhline(0, c='k');plt.title("Delayed1 input");plt.legend(loc='best')
    plt.subplot(7, 1, 5)
    plt.plot(sim.trange(), sim.data[LD_probe], label="Left")
    plt.plot(sim.trange(), sim.data[RB_probe], label="Right")
    plt.axhline(0, c='k');plt.title("Delayed1 input");plt.legend(loc='best')
    plt.subplot(7, 1, 6)
    plt.plot(sim.trange(), sim.data[LE_probe], label="Left")
    plt.plot(sim.trange(), sim.data[RA_probe], label="Right")
    plt.axhline(0, c='k');plt.title("Delayed1 input");plt.legend(loc='best')
    plt.subplot(7, 1, 7)
    plt.plot(sim.trange(), sim.data[LF_probe], label="Left")
    plt.plot(sim.trange(), sim.data[inpR_probe], label="Right")
    plt.axhline(0, c='k');plt.title("Delayed1 input");plt.legend(loc='best')
    plt.xlabel('time');

    plt.figure(2)
    plt.plot(sim.trange(), sim.data[adder1_probe], label="270")
    plt.plot(sim.trange(), sim.data[adder2_probe], label="300")
    plt.plot(sim.trange(), sim.data[adder3_probe], label="330")
    plt.plot(sim.trange(), sim.data[adder4_probe], label="0")
    plt.plot(sim.trange(), sim.data[adder5_probe], label="30")
    plt.plot(sim.trange(), sim.data[adder6_probe], label="60")
    plt.plot(sim.trange(), sim.data[adder7_probe], label="90")
    plt.xlabel('time');plt.legend(loc='best')
    plt.show()
    '''


# 擷取 Output Envelope (Coincidence Counter)
    def getEnvelope (inputSignal):
        # Taking the absolute value
        absoluteSignal = []
        for sample in inputSignal:
            absoluteSignal.append (abs (sample))
        # Peak detection
        intervalLength = 300 # Experiment with this number, it depends on your sample frequency and highest "whistle" frequency
        outputSignal = []
        output = []
        for baseIndex in range (intervalLength, len (absoluteSignal)):
            maximum = 0
            for lookbackIndex in range (intervalLength):
                maximum = max (absoluteSignal[baseIndex - lookbackIndex], maximum)
            outputSignal.append (maximum)
            output=outputSignal
        return output[0::10]

    coincidence_signal_270 = getEnvelope(sim.data[adder1_probe][0::10])
    coincidence_signal_300 = getEnvelope(sim.data[adder2_probe][0::10])
    coincidence_signal_330 = getEnvelope(sim.data[adder3_probe][0::10])
    coincidence_signal_0 = getEnvelope(sim.data[adder4_probe][0::10])
    coincidence_signal_30 = getEnvelope(sim.data[adder5_probe][0::10])
    coincidence_signal_60 = getEnvelope(sim.data[adder6_probe][0::10])
    coincidence_signal_90 = getEnvelope(sim.data[adder7_probe][0::10])


# 輸出結果

    output_data = np.array([coincidence_signal_270, coincidence_signal_300, \
                            coincidence_signal_330, coincidence_signal_0, \
                            coincidence_signal_30, coincidence_signal_60, \
                            coincidence_signal_90])
    savetxt('MSO_'+word+'_'+str(angle)+'_ch'+str(ch)+'.txt', output_data, fmt='%.10f')


    '''
    plt.figure()
    plt.plot(coincidence_signal_270, label='270')
    plt.plot(coincidence_signal_300, label='300')
    plt.plot(coincidence_signal_330, label='330')
    plt.plot(coincidence_signal_0, label='0')
    plt.plot(coincidence_signal_30, label='30')
    plt.plot(coincidence_signal_60, label='60')
    plt.plot(coincidence_signal_90, label='90')
    plt.xlabel('time');plt.legend(loc='best')
    plt.show()
    '''


# Nengo GUI
# nengo_gui.Viz(__file__).start()
