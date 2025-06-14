# ABM and EQL Training
These are code sets that were manipulated as a means of training 

# Read me from source code repository 
Code for "Learning differential equation models from stochastic agent-based model simulations," by John Nardini, Ruth Baker, Mat Simpson, and Kevin Flores. Article available at https://royalsocietypublishing.org/doi/full/10.1098/rsif.2020.0987 (Open-access version: https://arxiv.org/abs/2011.08255).

All code is implemented using Python (version 3.7.3). The folders listed in this repository implement code to run agent-based models (ABMs), perform equation learning (EQL), or store previously-computed data from ABM simulations.

The **ABM** folder contains the code used to implement the ABMs from our study, which include the birth-death-migration (BDM) process and the susceptible-infected-recovered (SIR) model. The jupyter notebook plot_ABM_DE_BDM.ipynb can be used to simulate the BDM ABM and compare its output to its mean-field model. The jupyter notebook plot_ABM_DE_SIR.ipynb can be used to simulate the SIR ABM and compare its output to its mean-field model. Both of these notebooks rely on ABM_package.py to implement the ABMs.

The **EQL** folder contains code used to implement the EQL methods. We provide a basic introduction in  EQL Tutorial.ipynb, and then the five case studies from our manuscript are available in Case Study 1 Varying parameters for BDM.ipynb, Case study 2 varying number of ABM Simulations.ipynb, Case Study 3a Varying data resolution.ipynb, Case study 3b predicting unobserved dynamics.ipynb, Case study 4 model selection with EQL.ipynb, and Case study 5 SIR Varying params.ipynb. Each of these files rely on the files PDE_FIND3.py, which conducts the EQL methods and PDEFind_class_online.py, which processes the data and calls the EQL methods.

The **data** folder contains pre-simulated and saved ABM output data. For the BDM process, the files are named as: "logistic_ABM_sim_rp_" + str(rp) + "_rd_" + str(rd) + "_real" + str(N) + ".npy", where rp = {0.001,0.01, 0.05, 0.1, 0.5, 1} is the agent proliferation rate, rd = {rp, rp/2, rp/4} is the agent death rate, and N = { 1,10,15,25, 50 } is the number of ABM simulations over which we computed the averaged ABM data. For the SIR model, the files are named as: "SIR_ABM_ri_" + str(ri) + "_ri_" + str(rr) + "_real" + str(N) + ".npy", where ri = {0.001, 0.005, 0.01, 0.05, 0.1, 0.25 } is the agent infection rate, rr = {ri, ri/4, ri/2 ri/10} is the agent recovery rate, and N = { 3,5,10,15,25, 50} is the number of ABM simulations over which we computed the averaged ABM data.

# Changes implemented
In **ABM_package.py**, function **local_neighborhood_mask** was added. This function creates a sparse matrix with 1s in the neighborhood of a point (loc), and 0s elsewhere. The resulting sparse matrix is then used to determine how 'crowded' the neighborhood surrounding the selected agent is. Within the BDM_ABM function, the local_neighborhood_mask function is called. A 'crowded' neighborhood results in the migration rate to be multiplied by a factor, f, and the proliferation rate to be multiplied by the factor, 1/f. A sparse neighborhood will result in the opposite effect. 

**data_generation.ipynb** created new saved data that can be run through the EQL algorithm. In order to do this, a derivative function was added to the ABM_package.py. This function calculates the forward, backward and centered differences of the ABM data/time points. They are stored in an array, such that the first value is determined by forward differences, the interior points are determined by centered differences and the final point is determined by backward difference. 


