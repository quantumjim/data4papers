from qiskit import QuantumCircuit, execute, Aer, transpile
from archiver4qiskit import get_backend, submit_job, get_archive, get_job

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

from qiskit import transpile
from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
from qiskit.visualization.timeline import draw

def get_noise(prob):
    
    if prob:
        noise_model = NoiseModel()   

        noise_model.add_all_qubit_quantum_error(pauli_error([('X',prob), ('I', 1 - prob)]), ["reset"])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(prob, 1), ["id"])
        noise_model.add_readout_error([[1-prob, prob],[prob, 1-prob]], [1])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(prob, 2), ["cx"])

        return noise_model
    else:
        return None

def get_params(backend_name,processor=None):
    '''
    Gets information about qubits, plaquettes and links for a given backend.
    Aer backends return Falcon info unless a different processor type is supplied.
    '''
    
    if 'simulator' in backend_name:
        if processor==None:
            processor = 'Falcon'
    else:
        processor = get_backend(backend_name).configuration().processor_type['family']

        
    if processor=='Falcon':

        N = 27

        plaquettes = [

            [[('x',1,4,7), ('y',3,5,8), ('z',12,13,14)],
             [('x',8,11,14), ('y',7,10,12), ('z',1,2,3)]],

            [[('x',12,15,18), ('y',14,16,19), ('z',23,24,25)],
             [('x',19,22,25), ('y',18,21,23), ('z',12,13,14)]]

        ]

        colors = [[0],[],[1]]

    elif processor=='Hummingbird':
        
        N = 65

        plaquettes = [

            [[('x',0,1,2), ('y',13,14,15), ('z',4,11,17)],
             [('x',15,16,17), ('y',2,3,4), ('z',0,10,13)]],

            [[('x',4,5,6), ('y',17,18,19), ('z',8,12,21)],
             [('x',19,20,21), ('y',6,7,8), ('z',4,11,17)]],

            [[('x',15,16,17), ('y',29,30,31), ('z',19,25,33)],
             [('x',31,32,33), ('y',17,18,19), ('z',15,24,29)]],

            [[('x',19,20,21), ('y',33,34,35), ('z',23,26,37)],
             [('x',35,36,37), ('y',21,22,23), ('z',19,25,33)]],

            [[('x',27,28,29), ('y',41,42,43), ('z',31,39,45)],
             [('x',43,44,45), ('y',29,30,31), ('z',27,38,41)]],

            [[('x',31,32,33), ('y',45,46,47), ('z',35,40,49)],
             [('x',47,48,49), ('y',33,34,35), ('z',31,39,45)]],

            [[('x',43,44,45), ('y',56,57,58), ('z',47,53,60)],
             [('x',58,59,60), ('y',45,46,47), ('z',43,52,56)]],

            [[('x',47,48,49), ('y',60,61,62), ('z',51,54,64)],
             [('x',62,63,64), ('y',49,50,51), ('z',47,53,60)]],

        ]

        colors = [[0,3,4,7], [2,6], [1,5]]
        
    else:
        
        N = 127
        
        plaquettes = [

            [[('x',0,1,2), ('y',18,19,20), ('z',4,15,22)],
             [('x',20,21,22), ('y',2,3,4), ('z',0,14,18)]],

            [[('x',4,5,6), ('y',22,23,24), ('z',8,16,26)],
             [('x',24,25,26), ('y',6,7,8), ('z',4,15,22)]],

            [[('x',20,21,22), ('y',39,40,41), ('z',24,34,43)],
             [('x',41,42,43), ('y',22,23,24), ('z',20,33,39)]],

            [[('x',24,25,26), ('y',43,44,45), ('z',28,35,47)],
             [('x',45,46,47), ('y',26,27,28), ('z',24,34,43)]],

            [[('x',28,29,30), ('y',47,48,49), ('z',32,36,51)],
             [('x',49,50,51), ('y',30,31,32), ('z',28,35,47)]],

            [[('x',37,38,39), ('y',56,57,58), ('z',41,53,60)],
             [('x',58,59,60), ('y',39,40,41), ('z',37,52,56)]],

            [[('x',41,42,43), ('y',60,61,62), ('z',45,54,64)],
             [('x',62,63,64), ('y',43,44,45), ('z',41,53,60)]],

            [[('x',45,46,47), ('y',64,65,66), ('z',49,55,68)],
             [('x',66,67,68), ('y',47,48,49), ('z',45,54,64)]],

            [[('x',58,59,60), ('y',77,78,79), ('z',62,72,81)],
             [('x',79,80,81), ('y',60,61,62), ('z',58,71,77)]],

            [[('x',62,63,64), ('y',81,82,83), ('z',66,73,85)],
             [('x',83,84,85), ('y',64,65,66), ('z',62,72,81)]],

            [[('x',66,67,68), ('y',85,86,87), ('z',70,74,89)],
             [('x',87,88,89), ('y',68,69,70), ('z',66,73,85)]],

            [[('x',75,76,77), ('y',94,95,96), ('z',79,91,98)],
             [('x',96,97,98), ('y',77,78,79), ('z',75,90,94)]],

            [[('x',79,80,81), ('y',98,99,100), ('z',83,92,102)],
             [('x',100,101,102), ('y',81,82,83), ('z',79,91,98)]],

            [[('x',83,84,85), ('y',102,103,104), ('z',87,93,106)],
             [('x',104,105,106), ('y',85,86,87), ('z',83,92,102)]],

            [[('x',100,101,102), ('y',118,119,120), ('z',104,111,122)],
             [('x',120,121,122), ('y',102,103,104), ('z',100,110,118)]],

            [[('x',104,105,106), ('y',122,123,124), ('z',108,112,126)],
             [('x',124,125,126), ('y',106,107,108), ('z',104,111,122)]]

        ]

        colors = [[0,3,5,9,11,14],[2,7,8,13],[1,4,6,10,12,15]]

    links = {color:[] for color in ['red','green','blue']}
    for p in colors[1]:
        links['red'] += plaquettes[p][1]
    for p in colors[2]:
        links['red'] += plaquettes[p][0]
    for p in colors[0]:
        links['green'] += plaquettes[p][0]
    for p in colors[2]:
        links['green'] += plaquettes[p][1] 
    for p in colors[0]:
        links['blue'] += plaquettes[p][1]
    for p in colors[1]:
        links['blue'] += plaquettes[p][0]
    for color, color_links in links.items():
        links[color] = list(set(color_links))
        
    return N, plaquettes, colors, links


def noise_details(archive_id):
    
    def net_prob(p,n):
        return (1-(1-2*p)**n)/2
    
    if '@' in archive_id:
        noise_backend = get_archive(archive_id).backend()
        properties = noise_backend.properties()
    else:
        noise_backend = get_backend(archive_id)
        properties = noise_backend.properties()

    probs_meas = [ properties.readout_error(j) for j in range(noise_backend.configuration().num_qubits) ]

    probs_prep = [ qubit[6].value for qubit in properties.qubits]
    
    probs_cx = {}
    times_cx = {}
    for j,k in noise_backend.configuration().coupling_map:
        if j<k:
            p = properties.gate_error('cx', [j,k])
            if p<0.5:
                probs_cx[j,k] = properties.gate_error('cx', [j,k])
                times_cx[j,k] = properties.gate_length('cx', [j,k])

    probs_idle = []
    for j in range(noise_backend.configuration().num_qubits):
        t_m = properties.readout_length(j)
        t_cx = max(times_cx.values())
        t_id = properties.gate_length('id',j)
        p_id = properties.gate_error('id', [j])
        probs_idle.append( net_prob(p_id, (t_m+t_cx)/t_id) )

        
    all_probs = probs_meas + probs_prep + list(probs_cx.values()) +  probs_idle
    p_0 = np.mean(all_probs)
    p_0_std = np.std(all_probs)
        
    return probs_meas, probs_prep, probs_cx, probs_idle, p_0, p_0_std


def get_circuit(rounds, backend_name, N, plaquettes, colors, links, order, dd):
    
    def measure_link(qc,r,v0,a,v1,b):
        if r=='z':
            qc.cx(v0,a)
            qc.cx(v1,a)
        else:
            if r=='y':
                qc.sdg([v0,v1])
            qc.h(a)
            qc.cx(a,v0)
            qc.cx(a,v1)
            qc.h(a)
            if r=='y':
                qc.s([v0,v1])
        to_measure.append((a,b))
        
    def finalize_measurements(to_measure):
        qc.barrier()
        unmeasured = list(range(N))
        for a,b in to_measure:
            qc.measure(a,b)
            unmeasured.remove(a)
        qc.barrier()
        for a,b in to_measure:
            qc.delay(ringdown[a],a)
        if 'simulator' in backend_name:
            qc.id(unmeasured)
        return [] 
    
    qc = QuantumCircuit(N,N*rounds)
    
    if 'simulator' not in backend_name:
        backend = get_backend(backend_name)
        rate = backend.configuration().sample_rate
        ringdown = [16*int((backend.properties().gate_length('reset',q)-backend.properties().readout_length(q))*rate/16) for q in range(N)]    
    else:
        ringdown = [0 for q in range(N)]

    # when simulating, begin with resets with which we can associate prep noise
    if 'simulator' in backend_name:
        qc.reset(qc.qubits)

    output = {}
    b = 0
    to_measure = []
    for j in range(rounds):
        if 'simulator' in backend_name:
            if j%3==0:
                qc.id(qc.qubits)
        color_links = links[order[j%3]]  
        for r,v0,a,v1 in color_links:
            measure_link(qc,r,v0,a,v1,b)                     
            output[j,r,v0,a,v1] = b # keep track of where results are written in the classical register
            b += 1
        if j%3==2:
            to_measure = finalize_measurements(to_measure)
    to_measure = finalize_measurements(to_measure)
            
        
    if 'simulator' not in backend_name:
        # transpile
        tqc = transpile(qc,backend)
        assert tqc.num_nonlocal_gates()==qc.num_nonlocal_gates(), 'Transpilation changed the number of non-local gates!'
        durations = InstructionDurations().from_backend(backend)
        # schedule
        pm = PassManager([ALAPSchedule(durations)])
        qc = pm.run(tqc)
        # add dd if needed
        if dd:
            dd_sequence = [XGate()]*2
            pm = PassManager([DynamicalDecoupling(durations, dd_sequence)])
            qc = pm.run(qc)
            # make sure delays are a multiple of 16 samples, while keeping the barriers
            # as aligned as possible
            total_delay = [{q:0 for q in qc.qubits} for _ in range(2)]
            for gate in qc.data:
                if gate[0].name=='delay':
                    q = gate[1][0]
                    t = gate[0].params[0]
                    total_delay[0][q] += t
                    new_t = 16*np.ceil((total_delay[0][q]-total_delay[1][q])/16)
                    total_delay[1][q] += new_t
                    gate[0].params[0] = new_t
        
    return qc, output


def run_experiment(backend_name,rounds,prob=None,num_copies=None,processor=None,dd=True,order=None,provider=None):
    
    if num_copies==None:
        if 'simulator' in backend_name or backend_name=='ibm_washington':
            num_copies = 1
        else:
            num_copies = 100
            
    if order==None:
        order = ['red', 'green', 'blue']
    
    N, plaquettes, colors, links = get_params(backend_name,processor=processor)
    qc, output = get_circuit(rounds, backend_name, N, plaquettes, colors, links, order, dd)

    backend = get_backend(backend_name,provider)
    if 'simulator' in backend_name:
        shots = 8192
    else:
        shots = 1024

    if 'simulator' in backend_name:
        if prob!=None:
            archive_id = submit_job(qc, backend, shots=shots, noise_model=get_noise(prob))
        else:
            archive_id = submit_job(qc, backend, shots=shots)
    else:
        rep_delay = None
        #if num_copies>1:
        #    rep_delay = 0.99*backend.configuration().rep_delay_range[1]

        archive_id = submit_job([qc.copy() for _ in range(num_copies)], backend, shots=shots, rep_delay=rep_delay)
            
    
    archive = get_archive(archive_id)
    archive.output = output
    archive.rounds = rounds
    archive.prob = prob
    archive.processor = processor
    archive.order = order
    archive.save()
        
    return archive_id

def get_results(archive_id):
    
    job_id, backend_name = archive_id.split('@')
    
    archive = get_archive(archive_id)
    output = archive.output
    rounds = archive.rounds
    processor = archive.processor
    shots = archive.qobj().to_dict()['config']['shots']
    order = archive.order
    
    N, plaquettes, colors, links = get_params(backend_name,processor=processor)

    counts = archive.result().get_counts()

    if type(counts)==list:
        num_copies = len(counts)
    else:
        num_copies = 1
        counts = [counts]

    T = int(rounds/3)

    link_bits = {}
    for color, color_links in links.items():
        for r,v0,a,v1 in color_links:
            link_bits[r,v0,a,v1] = []
            for j in range(rounds):
                if (j,r,v0,a,v1) in output:
                    bit = output[j,r,v0,a,v1]
                    if (r,v0,a,v1) in link_bits:
                        link_bits[r,v0,a,v1] += [bit]
                    else:
                        link_bits[r,v0,a,v1] = [bit]

    w_av = [[[0 for _ in range(num_copies)] for _ in range(T-1)] for p in range(len(plaquettes))]
    w_samples = [[[0 for _ in range(num_copies)] for _ in range(T-1)] for p in range(len(plaquettes))]
    for copy in range(num_copies):
        for string in counts[copy]:
            for c,color in enumerate(colors):

                for t in range(1,T):
                    t0 = t-1 # only this rounds resets needed, due to lack of resets
                    for p in color:
                        plaquette = plaquettes[p]

                        w = ''
                        for link in plaquette[0]+plaquette[1]:
                            if link in links[order[0]] and c==1: # the second color is a bit annoying
                                for dt in [+1,-1]: 
                                    b = link_bits[link][t+dt]
                                    w += string[-b-1]
                            else:
                                b = link_bits[link][t]
                                w += string[-b-1]

                        if w.count('1')%2==1:
                            w_av[p][t0][copy] += counts[copy][string]
                        w_samples[p][t0][copy] += counts[copy][string]

        for t in range(T-1):
            for p in range(len(plaquettes)):
                w_av[p][t][copy] /= w_samples[p][t][copy]
                
    for t in range(T-1):
        for p in range(len(plaquettes)):
            w_av[p][t] = np.mean(w_av[p][t])
                
    return w_av