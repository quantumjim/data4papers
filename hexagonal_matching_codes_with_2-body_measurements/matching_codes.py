from qiskit import QuantumCircuit, execute, Aer, transpile
from archiver4qiskit import get_backend, submit_job, get_archive, get_job

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

def get_noise(prob):
    
    if prob:
        noise_model = NoiseModel()   

        noise_model.add_all_qubit_quantum_error(pauli_error([('X',prob), ('I', 1 - prob)]), ["reset", "id"])
        noise_model.add_readout_error([[1-prob, prob],[prob, 1-prob]], [1])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(prob, 2), ["cx"])

        return noise_model
    else:
        return None
    
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
    for j,k in noise_backend.configuration().coupling_map:
        if j<k:
            probs_cx[j,k] = properties.gate_error('cx', [j,k])

    probs_idle = []
    for j in range(noise_backend.configuration().num_qubits):
        t_m = properties.readout_length(j)
        t_r = properties.gate_length('reset',j)
        t_id = properties.gate_length('id',j)
        p_id = properties.gate_error('id', [j])
        
        probs_idle.append( net_prob(p_id,(t_m+t_r)/t_id) )

        
    all_probs = probs_meas + probs_prep + list(probs_cx.values()) +  probs_idle
    p_0 = np.mean(all_probs)
    p_0_std = np.std(all_probs)
        
    return probs_meas, probs_prep, probs_cx, probs_idle, p_0, p_0_std


def get_params(backend_name):

    if 'aer' in backend_name:
        processor = 'Simulator'
    else:
        processor = get_backend(backend_name).configuration().processor_type['family']

    if processor in ['Falcon', 'Simulator']:

        N = 27

        plaquettes = [

            [[('x',1,4,7), ('y',3,5,8), ('z',12,13,14)],
             [('x',8,11,14), ('y',7,10,12), ('z',1,2,3)],
             [('z',1,2,3), ('z',12,13,14), ('z',None,7,None), ('z',None,8,None)]],

            [[('x',12,15,18), ('y',14,16,19), ('z',23,24,25)],
             [('x',19,22,25), ('y',18,21,23), ('z',12,13,14)],
             [('z',12,13,14), ('z',23,24,25), ('z',None,18,None), ('z',None,19,None)]]

        ]

        colors = [[0],[1]]

    else:

        N = 65

        plaquettes = [

            [[('x',0,1,2), ('y',13,14,15), ('z',4,11,17)],
             [('x',15,16,17), ('y',2,3,4), ('z',0,10,13)],
             [('z',0,10,13), ('z',4,11,17), ('z',None,2,None), ('z',15,24,29)]],

            [[('x',4,5,6), ('y',17,18,19), ('z',8,12,21)],
             [('x',19,20,21), ('y',6,7,8), ('z',4,11,17)],
             [('z',4,11,17), ('z',8,12,21), ('z',None,6,None), ('z',19,25,33)]],

            [[('x',15,16,17), ('y',29,30,31), ('z',19,25,33)],
             [('x',31,32,33), ('y',17,18,19), ('z',15,24,29)],
             [('z',15,24,29), ('z',19,25,33), ('z',4,11,17), ('z',31,39,45)]],

            [[('x',19,20,21), ('y',33,34,35), ('z',23,26,37)],
             [('x',35,36,37), ('y',21,22,23), ('z',19,25,33)],
             [('z',19,25,33), ('z',23,26,37), ('z',8,12,21), ('z',35,40,49)]],

            [[('x',27,28,29), ('y',41,42,43), ('z',31,39,45)],
             [('x',43,44,45), ('y',29,30,31), ('z',27,38,41)],
             [('z',27,38,41), ('z',31,39,45), ('z',15,24,29), ('z',43,52,56)]],

            [[('x',31,32,33), ('y',45,46,47), ('z',35,40,49)],
             [('x',47,48,49), ('y',33,34,35), ('z',31,39,45)],
             [('z',31,39,45), ('z',35,40,49), ('z',19,25,33), ('z',47,53,60)]],

            [[('x',43,44,45), ('y',56,57,58), ('z',47,53,60)],
             [('x',58,59,60), ('y',45,46,47), ('z',43,52,56)],
             [('z',43,52,56), ('z',47,53,60), ('z',31,39,45), ('z',None,58,None)]],

            [[('x',47,48,49), ('y',60,61,62), ('z',51,54,64)],
             [('x',62,63,64), ('y',49,50,51), ('z',47,53,60)],
             [('z',47,53,60), ('z',51,54,64), ('z',35,40,49), ('z',None,62,None)]],

        ]

        colors = [[0,5], [1,6], [2,7], [3,4]]

    z_links = []
    for plaquette in plaquettes:
        z_links += plaquette[2]
    z_links = list(set(z_links))


    return N, plaquettes, colors, z_links




def get_circuit(T, backend_name, N, plaquettes, colors, z_links):
    
    def measure_link(qc,r,v0,a,v1,b):
        if v0!=None:
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

            qc.measure(a,b)
            qc.reset(a)
        else:
            qc.measure(a,b)

    def idle(cxed, measured):
        # when simulating
        if 'aer' in backend_name:
            # add an id for all qubits not involved in the cx
            qc.id(list(set(range(N)).difference(cxed)))
            # and another for the measurement
            qc.id(list(set(range(N)).difference(measured)))
        
    qc = QuantumCircuit(N,N*T*len(colors))

    # when simulating, begin with resets with which we can associate prep noise
    if 'aer' in backend_name:
        qc.reset(qc.qubits)

    # the following are to keep track of where results are written in the classical register.
    output = {}
    b = 0

    for t in range(T):
        for c,color in enumerate(colors):
            for p in color:
                plaquette = plaquettes[p]
                for g in range(2):
                    cxed = []
                    measured = []
                    group = plaquette[g]
                    for r,v0,a,v1 in group:

                        measure_link(qc,r,v0,a,v1,b)                     
                        output[t,c,g,r,v0,a,v1] = b
                        b += 1

                        # keep track of which qubits have been affected
                        cxed += [v0,a,v1]
                        measured += [a]

                    # add ids on idle qubits
                    idle(cxed, measured)

            g = 2
            cxed = []
            measured = []
            for r,v0,a,v1 in z_links:

                measure_link(qc,r,v0,a,v1,b)                     
                output[t,c,g,r,v0,a,v1] = b
                b += 1

                cxed += [v0,a,v1]
                measured += [a]

            idle(cxed, measured)
        
    return qc, output


def get_results(archive_id,T,prob=None,num_copies=None):
    
    job_id, backend_name = archive_id.split('@')
    
    if num_copies==None:
        if 'aer' in backend_name:
            num_copies = 1
        else:
            num_copies = 100
    
    N, plaquettes, colors, z_links = get_params(backend_name)
    
    if not job_id:
        
        qc, output = get_circuit(T, backend_name, N, plaquettes, colors, z_links)
    
        backend = get_backend(backend_name)

        if 'aer' not in backend_name:
            tqc = transpile(qc,backend)
            assert tqc.num_nonlocal_gates()==qc.num_nonlocal_gates(), 'Transpilation changed the number of non-local gates!'

            
    shots = 8192

    if not job_id:

        note = 'For T='+str(T)
        if 'aer' in backend_name:
            if prob!=None:
                note += ' and prob='+str(prob)
                archive_id = submit_job(qc, backend, note=note, shots=shots, noise_model=get_noise(prob))
                print('\n'+note+':\narchive_id =', archive_id+'\n')
            else:
                archive_id = submit_job(qc, backend, note=note, shots=shots)
                print(note+':\narchive_id =', archive_id)
        else:
            rep_delay = None
            if num_copies>1:
                rep_delay = 0.99*backend.configuration().rep_delay_range[1]

            archive_id = submit_job([tqc.copy() for _ in range(num_copies)], backend, note=note, shots=shots, rep_delay=rep_delay)
            print(note+':\narchive_id =', archive_id)      
            
    
        archive = get_archive(archive_id)
        archive.output = output
        archive.save()
        
    else:
        
        archive = get_archive(archive_id)
        output = archive.output
    
    counts = archive.result().get_counts()

    if type(counts)==list:
        num_copies = len(counts)
    else:
        num_copies = 1
        counts = [counts]

    w_av = [[[0 for p in range(len(plaquettes))] for _ in range(T-1)] for _ in range(num_copies) ]
    w_samples = [[[0 for p in range(len(plaquettes))] for _ in range(T-1)] for _ in range(num_copies) ]
    z_av = [[[0 for p in range(len(plaquettes))] for _ in range(T-1)] for _ in range(num_copies) ]
    z_samples = [[[0 for p in range(len(plaquettes))] for _ in range(T-1)] for _ in range(num_copies) ]

    for copy in range(num_copies):

        for string in counts[copy]:

            for c,color in enumerate(colors):
                for p in color:
                    plaquette = plaquettes[p]

                    for t0 in range(T-1):
                        w = ''
                        for dt in range(2):
                            t = t0+dt
                            for g in range(2):
                                group = plaquette[g]
                                for r,v0,a,v1 in group:
                                    b = output[t,c,g,r,v0,a,v1]
                                    w += string[-b-1]

                        z = ''
                        g = 2
                        for r,v0,a,v1 in plaquette[g]:
                            b = output[t0+1,c,g,r,v0,a,v1]
                            if c>0:
                                pre_b = output[t0+1,c-1,g,r,v0,a,v1]
                            else:
                                pre_b = output[t0,len(colors)-1,g,r,v0,a,v1]
                            z += string[-b-1] + string[-pre_b-1]


                        if w.count('1')%2==1:
                            w_av[copy][t0][p] += counts[copy][string]
                        w_samples[copy][t0][p] += counts[copy][string] 

                        if z.count('1')%2==1:
                            z_av[copy][t0][p] += counts[copy][string]
                        z_samples[copy][t0][p] += counts[copy][string]


        for t in range(T-1):
            for p in range(len(plaquettes)):
                w_av[copy][t][p] /= w_samples[copy][t][p]
                z_av[copy][t][p] /= z_samples[copy][t][p]
                
    return w_av, z_av, archive_id

def get_average_min(w_av, z_av):
    
    av_w = np.mean([np.min(np.array(w_av_copy).flatten()) for w_av_copy in w_av])
    av_z = np.mean([np.min(np.array(z_av_copy).flatten()) for z_av_copy in z_av])
    std_w = np.std([np.min(np.array(w_av_copy).flatten()) for w_av_copy in w_av])
    std_z = np.std([np.min(np.array(z_av_copy).flatten()) for z_av_copy in z_av])
    
    return av_w, std_w, av_z, std_z
    
def get_average_mean(w_av, z_av):
    
    av_w = np.mean([np.mean(np.array(w_av_copy).flatten()) for w_av_copy in w_av])
    av_z = np.mean([np.mean(np.array(z_av_copy).flatten()) for z_av_copy in z_av])
    std_w = np.std([np.mean(np.array(w_av_copy).flatten()) for w_av_copy in w_av])
    std_z = np.std([np.mean(np.array(z_av_copy).flatten()) for z_av_copy in z_av])
    
    return av_w, std_w, av_z, std_z
