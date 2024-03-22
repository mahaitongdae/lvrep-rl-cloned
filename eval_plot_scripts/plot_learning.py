import numpy as np
import matplotlib.pyplot as plt
import csv

# mean_cost_arr = np.empty(4)
euler = False
sigmas = [0.0,1.0,2.0,3.0]
seeds = [0,1,2,3]
# rfs = [256, 512, 1024,2048]

# eval_seed= 100

n_evals = 11
sdec = np.empty((len(sigmas),n_evals))
eval_pts = [150, 175, 200, 225, 250,275,300,325,350,375,400]

sigma_eval = np.empty((len(sigmas),n_evals))
sigma_eval_std = np.empty((len(sigmas),n_evals))
for i in np.arange(len(sigmas)):
	seeds_eval = np.empty((len(seeds), n_evals))
	for j in np.arange(len(seeds)):
		filename = f"sdec_learning_data/rf_num=512_learn_rf=False_sigma={sigmas[i]}_euler=False_summary_files ({seeds[j]}).csv"
		seed_eval = np.empty(n_evals)
		with open(filename, newline = '') as csvfile:
			seed_file = csv.reader(csvfile, delimiter = ' ', quotechar = '|')
			t = 0
			for row in seed_file:
				row_data = row[0].split(',')
				# print(t, row_data)
				if t > 0: #skip 1st row
					seed_eval[t-1] = float(row_data[2])
					# print(float(row_data[2]))
				t += 1
			# print(t)
			# print(seed_eval)
		seeds_eval[j,:] = seed_eval
	sigma_eval[i,:] = np.mean(seeds_eval,axis = 0)
	sigma_eval_std[i,:] = np.std(seeds_eval,axis = 0)



# make text in plot look LaTex-like
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['xtick.labelsize'] = "x-large"
matplotlib.rcParams['ytick.labelsize'] = "x-large"
plt.rcParams.update({
    "text.usetex": True
})



colors = ['green','blue', 'orange', 'red']
for i in np.arange(len(sigmas)):
	plt.plot(eval_pts, sigma_eval[i,:], label = f'$\sigma$={sigmas[i]}', color = colors[i])
	plt.fill_between(eval_pts, sigma_eval[i,:] - sigma_eval_std[i,:], 
		sigma_eval[i,:] + sigma_eval_std[i,:], color = colors[i], alpha = 0.1 )
# plt.plot(eval_pts, sigma_eval[1,:], label = f'$\sigma$={sigmas[1]}', color = 'r')
# plt.fill_between(eval_pts, sigma_eval[1,:] - sigma_eval_std[1,:], 
# 	sigma_eval[1,:] + sigma_eval_std[1,:], color = 'r', alpha = 0.1 )

# plt.plot(eval_pts, sigma_eval[2,:], label = f'$\sigma$={sigmas[2]}', color = 'orange')
# plt.fill_between(eval_pts, sigma_eval[2,:] - sigma_eval_std[2,:], 
# 	sigma_eval[2,:] + sigma_eval_std[2,:], color = 'orange', alpha = 0.1 )
plt.xlabel("learning episode", fontsize = "x-large")
plt.ylabel("evaluation episodic reward", fontsize = "x-large")
plt.legend(fontsize = "x-large", loc = "lower right")
plt.grid()
plt.savefig("learning_curve.pdf",bbox_inches = "tight")
plt.close()




# #Note for sdec, the costs are all rewards

# #compute sdec fixed
# for i in np.arange(len(rfs)):
# 	rf_cost = 0
# 	for j in np.arange(len(seeds)):
# 		seed_costs = np.load(f"sdec/cost_log/cost_rf_num={rfs[i]}_learn_rf=False_tr_seed={seeds[j]}_eval_seed={eval_seed}_euler={euler}_sigma={sigmas[0]}.npy")
# 		mean_cost = np.mean(seed_costs)
# 		rf_cost += mean_cost
# 	rf_cost /= len(seeds) #get average
# 	sdec_fixed[i] = rf_cost

# #compute sdec fixed noisy
# for i in np.arange(len(rfs)):
# 	rf_cost = 0
# 	for j in np.arange(len(seeds)):
# 		seed_costs = np.load(f"sdec/cost_log/cost_rf_num={rfs[i]}_learn_rf=False_tr_seed={seeds[j]}_eval_seed={eval_seed}_euler={euler}_sigma={sigmas[1]}.npy")
# 		mean_cost = np.mean(seed_costs)
# 		rf_cost += mean_cost
# 	rf_cost /= len(seeds) #get average
# 	sdec_fixed_noisy[i] = rf_cost

# # #compute sdec tunable
# # for i in np.arange(len(rfs)):
# # 	rf_cost = 0
# # 	for j in np.arange(len(seeds)):
# # 		seed_costs = np.load(f"sdec/cost_log/cost_rf_num={rfs[i]}_learn_rf=True_tr_seed={seeds[j]}_eval_seed={eval_seed}_euler={euler}_sigma={sigmas[0]}.npy")
# # 		mean_cost = np.mean(seed_costs)
# # 		rf_cost += mean_cost
# # 	rf_cost /= len(seeds) #get average
# # 	sdec_tunable[i] = rf_cost

# # #compute sdec tunable
# # for i in np.arange(len(rfs)):
# # 	rf_cost = 0
# # 	for j in np.arange(len(seeds)):
# # 		seed_costs = np.load(f"sdec/cost_log/cost_rf_num={rfs[i]}_learn_rf=True_tr_seed={seeds[j]}_eval_seed={eval_seed}_euler={euler}_sigma={sigmas[1]}.npy")
# # 		mean_cost = np.mean(seed_costs)
# # 		rf_cost += mean_cost
# # 	rf_cost /= len(seeds) #get average across seeds
# # 	sdec_tunable_noisy[i] = rf_cost

# # #compute koopman performance
# # koopman = 0
# # for i in np.arange(len(seeds)):
# # 	seed_costs = np.load(f"koopman/cost_log/samples=80000_koopman_tr_seed={seeds[i]}_eval_seed=0_sigma={sigmas[0]}_euler=False.npy")
# # 	mean_cost = np.mean(seed_costs)
# # 	koopman -= mean_cost #Koopman returns costs, not rewardss
# # koopman /= len(seeds)

# # koopman_noisy = 0
# # for i in np.arange(len(seeds)):
# # 	seed_costs = np.load(f"koopman/cost_log/samples=80000_koopman_tr_seed={seeds[i]}_eval_seed=0_sigma={sigmas[1]}_euler=False.npy")
# # 	mean_cost = np.mean(seed_costs)
# # 	koopman_noisy -= mean_cost
# # koopman_noisy /= len(seeds)


# # #compute ilqr performance

# # ilqr = -np.mean(np.load(f"ilqr/cost_log/seed=0_euler=False_sigma={sigmas[0]}.npy"))



# # ilqr_noisy = 0
# # for i in np.arange(len(seeds)):
# # 	seed_costs = np.load(f"ilqr/cost_log/seed={seeds[i]}_euler=False_sigma={sigmas[1]}.npy")
# # 	mean_cost = np.mean(seed_costs)
# # 	ilqr_noisy -= mean_cost
# # ilqr_noisy /= len(seeds)


# # plt.style.use("fivethirtyeight")

# plt.plot(rfs, sdec_fixed, color = 'green', marker = 'o', linestyle = 'dashed', linewidth = 2, markersize = 12, label = "SDEC")
# plt.plot(rfs, sdec_fixed_noisy, color = 'blue', marker = 'o', linestyle = 'dashed', linewidth = 2, markersize = 12, label = "SDEC (noisy)")

# # plt.plot(rfs, sdec_tunable, color = 'orange', marker = '^', linestyle = 'dashed', linewidth = 2, markersize = 12, label = "SDEC (tunable)")
# # plt.plot(rfs, sdec_tunable_noisy, color = 'red', marker = '^', linestyle = 'dashed', linewidth = 2, markersize = 12, label = "SDEC (tunable), noisy")

# # plt.axhline(y = koopman, color = "brown", linestyle = 'dashed', linewidth = 2, label = "Koopman")


# # plt.axhline(y = koopman, color = "black", linestyle = 'dashed', linewidth = 2, label = "Koopman, noisy")

# # plt.axhline(y = ilqr, color = "purple", linestyle = 'dashed', linewidth = 2, label = "iLQR")

# # plt.axhline(y = ilqr_noisy, color = "magenta", linestyle = 'dashed', linewidth = 2, label = "iLQR, noisy")

# plt.xlabel("Number of random features")
# plt.ylabel("Episodic reward")
# plt.legend()
# plt.grid()
# plt.savefig("comparison.pdf")

# # for i in seeds:
# 	for sigma in sigmas:
# 		seed_costs = np.load(f"samples=80000_koopman_tr_seed={i}_eval_seed=0_sigma={sigma}_euler={euler}.npy")
# 		mean_cost = np.mean(seed_costs)
# 		print(f"tr seed {i} cost, euler={euler}, sigma = {sigma} ", mean_cost)
# 		# mean_cost_arr[i] = mean_cost

# print("mean cost over 4 training seeds: ", np.mean(mean_cost_arr))
# print("stdev over 4 training seeds: ", np.std(mean_cost_arr))