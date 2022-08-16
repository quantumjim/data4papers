from archiver4qiskit2 import get_backend, submit_job, get_archive

from topological_codes import RepetitionCode, GraphDecoder

from qiskit import transpile
from qiskit.circuit.library import RXGate, XGate, RZGate
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import thermal_relaxation_error, depolarizing_error, pauli_error

from qiskit import QuantumCircuit, Aer

from qiskit.visualization.timeline import draw

import networkx as nx
import numpy as np
import time

from matplotlib import pyplot as plt

import pickle

class RepetitionBenchmark():
    
    def __init__(
        self,
        backend_name,
        d=3,
        T=2,
        max_cx_error=0.5,
        max_shots=1e5,
        archive_ids=None,
        run=True,
        use_old=False,
        mid_ts=None,
        simulate=False,
        dds=[True,False],
        provider=None,
        echo_num=[2,0],
        dd_qubits=[None,None]):
        
        '''
        
        echo_num = [2,0]:
            Dynamical decoupling inserts echo_num[0] X gates into any gap
            Then echo_num[1] Y-X pairs are inserted into the resulting gaps
            The default argument [2,0] simply does XX
            
        dd_qubits = [None, None]
            dd_qubits[j] is the list of qubits on which the echo_num[j] dd
            sequence is applied. If `None`, it is applied to all.
        
        '''
        
        if archive_ids:
            archive_id = list(archive_ids.values())[0]
            if 'simulator' not in archive_id:
                self.backend = get_archive(archive_id).backend()
            else:
                self.backend = get_backend(backend_name)
        else:
            if type(backend_name)==str:
                self.backend = get_backend(backend_name, provider=provider)
            else:
                self.backend = backend_name
        self._durations = InstructionDurations().from_backend(self.backend)
            
        self.d = d
        self.T = T
        self.use_old = use_old
        self.run = run
        self.mid_ts = mid_ts
        
        self.echo_num = echo_num
        self.dd_qubits = dd_qubits
        
        self.shots = min(self.backend.configuration().max_shots,max_shots)
                
        self.length = 2*d-1
        self.max_cx_error = max_cx_error
        
        self.get_lines()
        
        self.t1 = {q: self.backend.properties().t1(q) for q in self.mid_qubits}
        self.t2 = {q: self.backend.properties().t2(q) for q in self.mid_qubits}
        
        self.rate = self.backend.configuration().sample_rate
        
        if archive_ids:
            dds = set([dd for _,dd in archive_ids])
            self.params = [(dtype, dd) for dtype in ['min', 'max'] for dd in dds]
            delays = [eval(delay) for delay,dd in archive_ids]
            self.delay = {'min':delays[0], 'max':delays[-1]}
        else:
            self.delay = {}
            self.delay['min'] = {
                q:[16*int((self.backend.properties().gate_length('reset',q)\
                          -self.backend.properties().readout_length(q))*self.rate/16)]*2 for q in self.mid_qubits
            }
            self.delay['max'] = {q:[16*int(t*self.rate/8 /16) for t in [self.t1[q], self.t2[q]] ] for q in self.mid_qubits}
            self.params = [(dtype, dd) for dtype in ['min', 'max'] for dd in dds]
                    
        self.bit_code = {}
        self.phase_code = {}
        for q in self.mid_qubits:
            for dtype,dd in self.params:
                delay = self.delay[dtype][q]
                self.bit_code[delay[0]] = RepetitionCode(d,T,resets=False,delay=delay[0],barriers=True)
                self.phase_code[delay[1]] = RepetitionCode(d,T,resets=False,delay=delay[1],xbasis=True,barriers=True)

                if len(self.mid_qubits)*3<self.backend.configuration().max_experiments:
                    self.logicals = ['0','1','+']
                else:
                    self.logicals = ['1','+']
        
        self.circuits = {}
        self.archive_ids = {}   
        for dtype,dd in self.params:
            self.get_circuits(dtype, dd)
            if archive_ids:
                self.archive_ids[str(self.delay[dtype]), dd] = archive_ids[str(self.delay[dtype]), dd]
            else:
                if run:
                    if not simulate:
                        circuits = self.circuits[dtype, dd]
                        backend = self.backend
                        noise_model = None
                    else:
                        circuits, noise_model = self.noisify(self.circuits[dtype, dd])
                        backend = Aer.get_backend('aer_simulator')
                    self.archive_ids[str(self.delay[dtype]), dd] = submit_job(circuits, backend, shots=self.shots, noise_model=noise_model)
        
        # decoder defined with a vanilla code
        self.decoder = GraphDecoder(RepetitionCode(d,T,resets=False,barriers=True))
            
        self._error_probs = None
        self._between_probs = None
        
        self.save()
        
    def measure_t1(self, dd=False, postselected=False):
        if postselected:
            between_probs = self.postselected_between_probs()
        else:
            between_probs = self.between_probs()
        t1 = {}
        for q in self.mid_qubits:
            t1[q] = [[] for _ in range(3)]
            for j in range(1,self.T):
                probs = []
                for dtype in ['min', 'max']:
                    probs.append( np.sum([between_probs[(dtype,dd)][q][bit][j] for bit in ['0', '1']]) )
                probs.append( (probs[1]-probs[0])/(1-2*probs[0]) )
                delays = [self.delay[dtype][q][0] for dtype in ['min', 'max']]
                delays += [delays[1]-delays[0]]
                for j in range(3):
                    t1[q][j].append(-delays[j]/self.rate/np.log(1-probs[j]))
        return t1
        
    def measure_t2(self, dd=False, postselected=False):
        if postselected:
            between_probs = self.postselected_between_probs()
        else:
            between_probs = self.between_probs()
        t2 = {}
        for q in self.mid_qubits:
            t2[q] = [[] for _ in range(3)]
            for j in range(1,self.T):
                probs = [between_probs[(dtype,dd)][q]['+'][j] for dtype in ['min', 'max']]
                probs.append( (probs[1]-probs[0])/(1-2*probs[0]) )
                delays = [self.delay[dtype][q][1] for dtype in ['min', 'max']]
                delays += [delays[1]-delays[0]]
                for j in range(3):
                    t2[q][j].append(-delays[j]/self.rate/np.log(1-2*probs[j]))
        return t2
            
    def _transpile(self,qc,dd,initial_layout):
        if self.run:
            # transpile to backend and schedule
            qc = transpile(qc,self.backend,initial_layout=initial_layout,scheduling_method='alap')

            # then dynamical decoupling if needed
            if dd:

                dd_sequences = []
                spacings = []
                for j,echo_num in enumerate(self.echo_num):  
                    if echo_num:
                        if j==0:
                            dd_sequences.append( [XGate()]*echo_num )
                            spacings.append(None)
                        elif j==1:
                            dd_sequences.append( [XGate(), RZGate(np.pi), XGate()]*echo_num )
                            d = 1.0/(2*echo_num-1+1)
                            spacing = [d/2]+([0,d,d]*echo_num)[:-1]+[d/2]
                            for _ in range(2):
                                spacing[0] += 1-sum(spacing)
                            spacings.append(spacing)
                    else:
                        dd_sequences.append(None)
                        spacings.append(None)

                for j,dd_sequence in enumerate(dd_sequences):
                    if dd_sequence:
                        if self.dd_qubits[j]:
                            qubits = [initial_layout[q] for q in self.dd_qubits[j]]
                        else:
                            qubits = None
                        pm = PassManager([DynamicalDecoupling(self._durations, dd_sequence, qubits=qubits, spacing=spacings[j])])
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

                # transpile to backend and schedule again
                qc = transpile(qc,self.backend,scheduling_method='alap')
   
        return qc

    def processed_results(self, dtype, dd):

        job = get_archive(self.archive_ids[str(self.delay[dtype]), dd])
        
        results = []
        for j,q in enumerate(self.mid_qubits):
            delay = self.delay[dtype][q]
            for k,logical in enumerate(self.logicals):
                log = '0'*(logical not in ['1', '-']) + '1'*(logical in ['1', '-'])
                raw_results = {log:job.result().get_counts(j*len(self.logicals)+k)}
                if logical in ['0','1']:
                    results.append(self.bit_code[delay[0]].process_results(raw_results))
                else:
                    results.append(self.phase_code[delay[1]].process_results(raw_results))
                    
        return results
    
    def _update_error_probs(self, dtype, dd):

        job = get_archive(self.archive_ids[str(self.delay[dtype]), dd])
        
        results = self.processed_results(dtype, dd)
        
        all_error_probs = []
        for j,q in enumerate(self.mid_qubits):
            delay = self.delay[dtype][q]
            for k,logical in enumerate(self.logicals):
                log = '0'*(logical not in ['1', '-']) + '1'*(logical in ['1', '-'])
                all_error_probs.append(self.decoder.get_error_probs(results[j*len(self.logicals)+k],logical=log,use_old=self.use_old))
        
        return all_error_probs
    
    def update_between_probs(self):
        
        if self._error_probs==None:
            self.update_error_probs()

        q_mid = int(self.length/2)
        s_mid = int(self.d/2)-1, int(self.d/2)

        between_probs = {}
        for dtype,dd in self.params:
            between_probs[dtype,dd] = {q: {log:[None for _ in range(self.T+1)] for log in self.logicals} for q in self.mid_qubits }
            j = 0
            for q in self.mid_qubits:
                for log in self.logicals:
                
                    error_probs = self._error_probs[dtype,dd][j]
                    for e0,e1 in error_probs:
                        (s,t,x) = e0
                        (ss,tt,xx) = e1
                        prob = error_probs[e0,e1]

                        if x==s_mid[0] and xx==s_mid[1]:
                            assert t==tt
                            if prob or prob==0:
                                between_probs[dtype,dd][q][log][t] = prob
                            else:
                                between_probs[dtype,dd][q][log][t] = np.nan
                    j += 1

        self._between_probs = between_probs
        
    def postselected_between_probs(self):

        assert self.d==3 and self.T==2, 'Post-selected method is only valid for d=3, T=2 codes.'
        
        between_probs = {}
        for dtype,dd in self.params:
            
            results = self.processed_results(dtype,dd)
            
            between_probs[dtype,dd] = {}
            #q: {log:[None for _ in range(self.T+1)] for log in self.logicals} for q in self.mid_qubits }

            for j,q in enumerate(self.mid_qubits):
                between_probs[dtype,dd][q] = {}
                for dj,logical in enumerate(self.logicals):
                    between_probs[dtype,dd][q][logical] = []
                    for t in range(3):
                        if logical=='+':
                            log = '0'
                        else:
                            log = logical
                        string1 = (log+' '+log+'  '+'00 '*t+'11 '+'00 '*(2-t))[0:-1]
                        string0 = (log+' '+log+'  '+3*'00 ')[0:-1]
                        ratio = results[3*j+dj][log][string1]/results[3*j+dj][log][string0]
                        between_probs[dtype,dd][q][logical].append( ratio/(ratio+1) )

        return between_probs
        
    def update_error_probs(self):
        self._error_probs = {}
        for dtype,dd in self.params:
            self._error_probs[dtype,dd] = self._update_error_probs(dtype,dd)
    
    def error_probs(self):
        if self._error_probs==None:
            self.update_error_probs()
        return self._error_probs
    
    def between_probs(self):
        if self._between_probs==None:
            self.update_between_probs()
        return self._between_probs
            
    def get_lines(self):
        '''
        For the coupling map of the given backend, find a line of qubits of the given length,
        centered on `mid`, such that no cx has a greater error than `max_cx_error`.
        '''

        backend = self.backend
        length = self.length
        max_cx_error = self.max_cx_error

        # get the coupling graph
        coupling_map = nx.Graph()
        for pair in backend.configuration().coupling_map:
            error = backend.properties().gate_error('cx',pair)
            if error<max_cx_error:
                coupling_map.add_edge(pair[0],pair[1],error=error)
            else:
                coupling_map.add_node(pair[0])
                coupling_map.add_node(pair[1])

        self.mid_qubits = []
        self.lines = []
        self._errors = []
        for mid in coupling_map.nodes:

            # find all the neighbours of the original point at all relevant distances
            depth = int((length-1)/2)
            neighbors = {mid}
            for j in range(depth):
                for q in neighbors:
                    neighbors = neighbors.union(set(coupling_map.neighbors(q)))
            # and put them in a graph
            neighbourhood = nx.Graph()
            for q0 in neighbors:
                for q1 in neighbors:
                    if [q0,q1] in coupling_map.edges():
                        neighbourhood.add_edge(q0,q1)

            # find all lines of the correct length centered on the correct point
            lines = []
            for q0 in neighbors:
                for q1 in neighbors:
                    if q0!=q1:
                        paths = list(nx.algorithms.shortest_simple_paths(neighbourhood,q0,q1))
                        for path in paths:
                            if len(path)==length:
                                if path[depth]==mid:
                                    if path not in lines and path[::-1] not in lines:
                                        lines.append(path)

            # for each line, determine the max cx error for any cx that touches the mid point,
            # and the overall max (errors[0] and errors[1], respectively)
            errors = []                 
            for line in lines:
                error = []
                for j0,j1 in [(depth-1,depth),(0,length-1)]:
                    error.append(max([coupling_map[line[j]][line[j+1]]['error'] for j in range(j0,j1)]))
                errors.append(error)

            # find the line for which
            # * no error is greater than the allowed max
            # * of those that satisfy the above, the smallest local max
            # * of those that satisfy the above, the global local max
            min_e0 = np.inf
            min_e1 = {error[0]:np.inf for error in errors}
            for error in errors:
                min_e0 = min(error[0],min_e0)
                min_e1[error[0]] = min(error[1],min_e1[error[0]])

            if min_e1:
                best_error = [min(min_e1),min_e1[min(min_e1)]]                        
                best_line =  lines[errors.index(best_error)]

                # for consistency order this so that best_line[0]<best_line[1]
                if best_line[-1]<best_line[0]:
                    best_line = best_line[::-1]

                # and record all results
                self.mid_qubits.append(mid)
                self.lines.append(best_line)
                self._errors.append(best_error)
                
    def get_circuits(self, dtype, dd):
            
        # set initial layout given lines
        initial_layouts = []
        for line in self.lines:
            initial_layout = []
            for j in range (self.d-1):
                initial_layout.append(line[2*j+1])
            for j in range (self.d):
                initial_layout.append(line[2*j])
            initial_layouts.append(initial_layout)
            
        # collect and transpile circuits
        self.circuits[dtype, dd] = []
        for q, initial_layout in zip(self.mid_qubits,initial_layouts):
            circuits = []
            delay = self.delay[dtype][q]
            for log in self.logicals:
                if log in ['0', '1']:
                    circuits += [self.bit_code[delay[0]].circuit[log]]
                else:
                    circuits += [self.phase_code[delay[1]].circuit['0'*(log=='+') + '1'*(log=='-')]]
            self.circuits[dtype, dd] += [self._transpile(qc,dd,initial_layout) for qc in circuits]

        # test transpilation
        for qc in self.circuits[dtype, dd]:
            assert qc.num_nonlocal_gates()==qc.num_nonlocal_gates(), 'Non-trivial transpilation'
            
    def noisify(self, qcs, excited_state_population=1/3):

        properties = self.backend.properties()
        
        noise_model = NoiseModel()
        for j in range(len(qcs[0].qubits)):
            p_meas = properties.readout_error(j)
            noise_model.add_readout_error([[1-p_meas, p_meas],[p_meas, 1-p_meas]], [j])
        
        noisy_qcs = []
        for qc in qcs:
            noisy_qc = QuantumCircuit()
            for regs in [qc.qregs, qc.cregs]:
                for reg in regs:
                    noisy_qc.add_register(reg)
            for g,qs,cs in qc:
                js = [qc.qubits.index(q) for q in qs]
                if g.name=='delay':
                    for j,q in zip(js,qs):
                        if self.mid_ts:
                            t1,t2 = self.mid_ts[j]
                        else:
                            t1 = properties.t1(j)
                            t2 = min(t1,properties.t2(j))
                        relax = thermal_relaxation_error(t1, t2, g.duration/self.rate, excited_state_population=excited_state_population)
                        noisy_qc.append(relax,[j])
                elif g.name=='cx':
                    p_cx = properties.gate_error('cx', js)
                    noisy_qc.append(depolarizing_error(p_cx, 2), qs)
                if g.name!='delay':
                    noisy_qc.append(g,qs,cs)
            noisy_qcs.append(noisy_qc)
        return noisy_qcs, noise_model
            
    def save(self):
        self.name = self.backend.name() + '@' + str(time.time()).split('.')[0]
        with open('experiments/'+self.name, 'wb') as file:
            pickle.dump(self,file)
            
            
def minimal_circuits(backend_name, shots = 8192,background_plus=False):
    '''
    Create circuits to required for minimal calculations of t1 and t2,
    with the same parameters as a repetition benchmark.
    '''
    
    # we make a runless benchmark object to use the same delays and mid qubits
    bench = RepetitionBenchmark(backend_name,run=False)

    # logicals are labelled 0, 1 and 2 here, with 2 being +
    n = len(bench.backend.properties().qubits)
    qcs = []
    for dtype in ['min','max']:
        for q in bench.mid_qubits:
            for log in range(3):
                qc = QuantumCircuit(n,1)
                if background_plus:
                    qc.h([j for j in range(n) if j!=q])
                if log==1:
                    qc.x(q)
                if log==2:
                    qc.h(q)
                if background_plus:
                    qc.barrier()
                qc.delay(bench.delay[dtype][q][log==2], q)
                if log==2:
                    qc.h(q)
                qc.measure(q,0)
                qcs.append( bench._transpile(qc,log==2,None) ) # dd applied for +
                                
    return qcs, bench

def process_minimal(archive_id, bench):
    '''
    Peform minimal calculations of t1 and t2 for results from
    running the circuits of `minimal_circuits`.
    '''
    
    def diff(p0, p1):
        return max((p1-p0)/(1-p0),0)

    job = get_archive(archive_id)

    # get the probability of the incorrect output for each circuit
    probs = [ {q:{dtype:0 for dtype in ['min','max']} for q in bench.mid_qubits} for log in range(3)]
    j = 0
    for dtype in ['min','max']:
        for q in bench.mid_qubits:
            for log in range(3):
                counts = job.result().get_counts(j)
                shots = sum(counts.values())
                if str((log+1)%2) in counts:
                    probs[log][q][dtype] = counts[str((log+1)%2)]/shots
                j += 1

    # get the difference of probs between long and short delays
    # also add together the 0 and 1 probs to get the prob needed for t1
    # the remaining one is used for t2
    prob_diff = [{},{}]
    for q in bench.mid_qubits:

        prob_diff[0][q] = 0
        for log in range(2):
            prob_diff[0][q] += diff(probs[log][q]['min'], probs[log][q]['max'])

        prob_diff[1][q] = diff(probs[2][q]['min'], probs[2][q]['max'])
    
    
    # from these probabilities, determine t1 and t2 for each qubit
    t1 = {}
    t2 = {}
    for q in bench.mid_qubits:
        delay_diff = [bench.delay['max'][q][t] - bench.delay['min'][q][t] for t in range(2)]
        t1[q] = -delay_diff[0]/bench.rate/np.log(1-prob_diff[0][q])
        t2[q] = -delay_diff[1]/bench.rate/np.log(1-2*prob_diff[1][q])
        
    return probs, prob_diff, t1, t2