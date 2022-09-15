# CLNSSS_AttackSim
This directory contains the code of simulations of the paper:

Edith Cohen, Xin Lyu, Jelani Nelson, Tamás Sarlós, Moshe Shechner, Uri Stemmer
["On the Robustness of CountSketch to Adaptive Inputs"](https://arxiv.org/abs/2202.13736), ICML 2022.

Three simulations are implemented:
1) Tracking 3 keys deviation w.r.t an attack vector per attack rounds
2) Sweeping parameter 'l' (the sketch "depth") vs the amount of rounds the attack reaches predetermined BNR (Bias to Noise Ratio).
3) Sweeping parameter 'b' (the sketch "width") vs the amount of rounds the attack reaches predetermined BNR (Bias to Noise Ratio).

Simulations are presented on Jupyter notebooks. In addition, their code is also in directory '/src/'

Plots are implemented for sim_1 and sim_2
