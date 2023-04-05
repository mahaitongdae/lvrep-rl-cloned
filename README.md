# lvrep-rl modified by Zhaolin

The code for random feature + SAC is found in the cdc_2023 branch. For more details refer to the README in that branch. 

The code for iLQR is found in the branch iLQC. Please use test_main.py in that branch to train and evaluate iLQR on the pendulum swingup problem.

The code for Koopman control is found in the branch DeepKoopman. Please use the Learn_Koopman_with_KlinearEig.py file in the train folder of that branch to train a new Koopman dynamics model; for evaluation, please use the og_eval_mpc.py file in the control folder of that branch.
