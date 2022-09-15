import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from numpy.random import binomial
from numpy.random import normal
import math

class Attack_round_sketch_width_dependency:
    def __init__(self, nof_repititions = 5, bnr_list = [1.0, 2.0], nof_sim_keys=3, tail_size=1000, b_list = [30, 60], l=100, bk_ratio = 3, seed=None):
        self.nof_repititions = nof_repititions
        self.nof_sim_keys    = nof_sim_keys
        self.bnr_list        = bnr_list
        self.tail_size       = tail_size
        self.l               = l
        self.b_list          = b_list
        self.bk_ratio        = bk_ratio
        self.seed            = seed
        self.rng             = default_rng(seed)
        self.results         = np.zeros((self.nof_repititions, len(self.b_list), len(self.bnr_list)), dtype=float)

    def __repr__(self):
        return "This test is sweeping b values. Test parameters:\n number of repititions = {0}; \n num of keys simulated = {1};\n attack tail size = {2};\n ell value = {3}; b/k = {7}; b values = {4}; BNR values = {6};\n seed = {5};" \
            .format(self.nof_repititions, self.nof_sim_keys, self.tail_size, self.l, self.b_list, self.seed, self.bnr_list, self.bk_ratio)

    def set_l(self, curr_l):        
        self.l   = curr_l
    def set_b(self, curr_b):        
        self.b   = curr_b
        self.k   = int(self.b / self.bk_ratio)
    #def set_k(self, curr_k):        self.k   = int(self.b / self.bk_ratio)
    def set_bnr(self, curr_bnr):    
        self.bnr = curr_bnr
    
    def draw_sketch(self):
        self.sim_keys_hash = self.rng.choice(self.b, size=(self.l, self.k)) 
        self.sim_keys_sign = self.rng.choice(2     , size=(self.l, self.k)) * 2 - 1
    
    def generate_v(self, mk_factor=10, lk_factor=20):
        tail_sd = math.sqrt(self.tail_size / self.b)
        lk_weight = int(tail_sd * lk_factor)
        mk_weight = int(tail_sd * mk_factor)
        v = np.ones(self.k, dtype=int)
        v[0: 2] = mk_weight
        v[2:  ] = lk_weight
        return v
    
    def encode_v(self, v):
        counters_v = np.zeros((self.l, self.b), dtype=int)
        for line in range(self.l):
            for key in range(self.k):
                counters_v[line, self.sim_keys_hash[line, key]] += v[key] * self.sim_keys_sign[line, key]
        return counters_v
    
    def decode_v(self, counters):
        weak_estimates = np.zeros((self.l, self.k), dtype=int)
        for line in range(self.l):
            for key in range(self.k):
                weak_estimates[line, key] = counters[line, self.sim_keys_hash[line, key]] * self.sim_keys_sign[line, key]
        # for even length axis: returns average of the two medians
        estimates = np.median(weak_estimates, axis=0)
        return estimates
    
    def check_parameters(self, nof_checks = 10):
        # pass if all sketch draws return exact estimates for v
        print("testing parameters l = {0}, b = {1}, k = {3}, b-k ratio = {2}.".format(self.l, self.b, self.bk_ratio, self.k))
        v = self.generate_v()
        results = np.zeros(nof_checks, dtype=int)
        for i in range(nof_checks):
            self.draw_sketch()
            estimates_v = self.decode_v(self.encode_v(v))
            #print(estimates_v)
            results[i] = np.absolute(estimates_v - v).sum()
        return results.sum() == 0
    
    def record_attack_round_dependency(self, file_pref = "sim_3"):
        self.results = np.zeros((self.nof_repititions, len(self.b_list), len(self.bnr_list)), dtype=float)
        for b_idx in range(len(self.b_list)):
            self.set_b(self.b_list[b_idx])
            # check sketch parameters
            if self.check_parameters() == False:
                print("Sketch paremeters sanity check failed for parameters ell = {0} b = {1}. Simulation stopped."\
                     .format(self.l, self.b))
                return None
            for bnr_idx in range(len(self.bnr_list)):
                self.set_bnr(self.bnr_list[bnr_idx])
                for rep in range(self.nof_repititions):
                    #print("simulating b = {0}, BNR = {1} repitition = {2}".format(self.b_list[b_idx], self.bnr_list[bnr_idx], rep))
                    self.draw_sketch()
                    self.results[rep, b_idx, bnr_idx] = self.simulate_median_attack()
            self.results[:, b_idx, :]            
            file_name = "./results/{0}_b_{1}.csv".format(file_pref, self.b_list[b_idx])
            print("saving file for b = {0} in {1}".format(self.b_list[b_idx], file_name))
            np.savetxt(file_name, self.results[:, b_idx, :], delimiter=',')
    
    def simulate_median_attack(self):
        counters_a    = np.zeros((self.l, self.nof_sim_keys), dtype=float)
        key_0_bias    = 0
        key_1_bias    = 0
        win_round     = 1
        nof_collected = 0
        while abs(key_0_bias) < self.bnr or abs(key_1_bias) < self.bnr:
            #  query
            counters_z = self.get_tail_contribution()
            counters_z_median = np.median(counters_z, axis=0)
            # collection desicion
            if counters_z_median[0] > counters_z_median[1]:
                counters_a = counters_a + counters_z
                nof_collected += 1
            elif counters_z_median[0] < counters_z_median[1]:
                counters_a = counters_a - counters_z
                nof_collected += 1
            # update the keys stopping signal
            factor = 1
            if nof_collected > 0:
                factor = math.sqrt(nof_collected * self.tail_size / self.b)
            estimates_a = np.median(counters_a, axis = 0)
            key_0_bias = estimates_a[0] / factor
            key_1_bias = estimates_a[1] / factor
            win_round += 1
        print("For l = {3}, BNR = {4}, attack wins after {2} rounds. Bias: key 0 = {0}, key 1 = {1}."
              .format(key_0_bias, key_1_bias, win_round, self.l, self.bnr))
        return win_round
    
    # BLRW distribution contribution
    def get_tail_contribution(self):
        contribution = np.zeros((self.l, self.nof_sim_keys), dtype=float)
        normal_mean = 0
        normal_sd   = 1
        for j in range(self.l):
            line_contribution    = np.zeros(3, dtype = float)
            line_contribution[0] = math.sqrt(binomial(self.tail_size, 1/self.b)) * normal(normal_mean, normal_sd)
            line_contribution[1] = math.sqrt(binomial(self.tail_size, 1/self.b)) * normal(normal_mean, normal_sd)
            line_contribution[2] = math.sqrt(binomial(self.tail_size, 1/self.b)) * normal(normal_mean, normal_sd)
            contribution[j] = line_contribution
        return contribution

# test 1: sweep b with bnr = 1, l = 100, for 40 repititions
test = Attack_round_sketch_width_dependency(nof_repititions = 40, bnr_list = [1.0], b_list = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300])
print(test)
test.record_attack_round_dependency(file_pref = "sim_3_run_bnr_1")

# test 1: sweep b with bnr = 2, l = 100, for 20 repititions
test = Attack_round_sketch_width_dependency(nof_repititions = 20, bnr_list = [2.0], b_list = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300])
print(test)
test.record_attack_round_dependency(file_pref = "sim_3_run_bnr_2")