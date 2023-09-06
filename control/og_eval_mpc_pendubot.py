#computes the performance of the learned koopman-U model for Pendulum-v1
import sys
sys.path.append("../utility")
sys.path.append("../train")
sys.path.append("../gym_env")
import do_mpc

import numpy as np
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import random
import argparse
from collections import OrderedDict
from copy import copy
import scipy
import scipy.linalg
from Utility import data_collecter

import os
from numpy.linalg import inv


import Learn_Koopman_with_KlinearEig as lka
from our_env.noisy_pend import noisyPendulumEnv
from pendulum import PendulumEnv
from envs.env_helper import *



def Prepare_Region_LQR(env_name, Nstate,NKoopman, thdot_weight=0.1, u_weight = 0.01):
	x_ref = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
	if env_name == "Quadrotor2D-v2":
		Q = np.zeros((NKoopman,NKoopman))
		Q[0, 0] = 1.
		Q[1, 1] = 1.
		Q[2, 2] = 0.01
		Q[3, 3] = 0.01
		Q[4, 4] = 1.
		Q[5, 5] = 0.01
		R = np.eye(1) * u_weight
	return Q,R, x_ref

def Psi_o(s,net,NKoopman): # Evaluates basis functions Psi(s(t_k))
	psi = np.zeros([NKoopman,1])
	print("s shape", s.shape)
	ds = net.encode(torch.DoubleTensor(s)).detach().cpu().numpy()
	psi[:NKoopman, 0] = ds
	return psi

def normalize(angle):
	return ((angle + np.pi) % (2 * np.pi) - np.pi)

def Cost(observations,u_list,Q,R,x_ref,gamma = 0.99):
	steps = observations.shape[1]
	loss = 0
	for s in range(steps):
		if s != steps - 1:
			print("u now", u_list[s])
		x_cost = (observations[0, s] - 1)** 2 + \
				 (observations[1, s]) ** 2 + \
				 0.01 * (observations[2, s]) ** 2 + \
				 0.01 * (observations[3, s] - 1) ** 2 + \
				 (observations[4, s]) ** 2 + \
				 0.01 * (observations[5, s]) ** 2
		u_cost = 0.01 * u_list[s] ** 2
		# loss += x_cost * gamma**s
		loss += x_cost + u_cost
	return loss

def main(tr_seed = 0, save_traj = False, samples = 80000, sigma = 0.0, euler = False):
	method = "KoopmanU"
	suffix = "Pendubot-v0"
	env_name = "Pendubot-v0"
	root_path = "../Data/" + suffix
	dt = 0.008
	print("euler", euler)
	print("sigma",sigma)
	model_path = 'KoopmanU_Quadrotor2D-v2layer3_edim70_eloss0_gamma0.8_aloss1_samples100000_dt0.008_seed0_return_norm_th=False_sigma=1.0_euler=False.pth'
	Data_collect = data_collecter(env_name)
	udim = Data_collect.udim
	Nstate = Data_collect.Nstates
	print("Nstate",Nstate)
	dicts = torch.load(root_path + "/" + model_path)
	state_dict = dicts["model"]

	#build net for koopmanU
	layer = dicts["layer"]
	NKoopman = layer[-1] + Nstate
	net = lka.Network(layer,NKoopman,udim)
	net.load_state_dict(state_dict)
	device = torch.device("cpu")
	net.cpu()
	net.double()

	#perform eval experiments
	gamma = 1.
	Ad = state_dict['lA.weight'].cpu().numpy()
	Bd = state_dict['lB.weight'].cpu().numpy()
	Q,R,x_ref = Prepare_Region_LQR(env_name,Nstate,NKoopman)
	Ad = np.matrix(Ad)
	Bd = np.matrix(Bd)
	print(f"Bd for seed={tr_seed}", Bd)
	
	#make model for mpc
	model_type = 'discrete'
	model = do_mpc.model.Model(model_type)
	_phi = model.set_variable(var_type = '_x', var_name = 'phi', shape = (NKoopman,1))
	_u = model.set_variable(var_type = '_u', var_name = 'u', shape = (1,1))
	phi_next = Ad@_phi + Bd@_u
	model.set_rhs('phi',phi_next)
	phi_cost = (_phi[0] - 1.0)**2 \
			   + _phi[1]**2 \
			   + 0.01 * _phi[2] ** 2 \
			   + 0.01 * (_phi[3] - 1.0)**2 \
			   + (_phi[4]) ** 2\
			   + 0.01 * _phi[5] ** 2  #should be (1,1) shape
	# phi_cost = _phi[1] **2 + 0.1 * _phi[2]**2
	u_cost = 0.01 * _u ** 2
	model.set_expression(expr_name = 'phi_cost',expr = phi_cost)
	model.set_expression(expr_name = 'u_cost',expr = u_cost)
	model.setup()

	#make controller for mpc
	mpc = do_mpc.controller.MPC(model)
	setup_mpc = {
		'n_robust': 0,
		'n_horizon': 20,
		't_step': 0.05,
		'state_discretization':'discrete',
    	'store_full_solution':True,		
	}
	mpc.set_param(**setup_mpc)

	#make objective for mpc
	mterm = model.aux['phi_cost'] #terminal cost
	lterm = model.aux['u_cost'] + model.aux['phi_cost']# stage cost
	mpc.set_objective(mterm = mterm, lterm =lterm)

	# lower bounds of the input
	mpc.bounds['lower','_u','u'] = -0.5

	# upper bounds of the input
	mpc.bounds['upper','_u','u'] =  0.5

	#finish setup of mpc
	mpc.setup()
	mpc.set_initial_guess()

	#estimator/simulator
	estimator = do_mpc.estimator.StateFeedback(model)
	simulator = do_mpc.simulator.Simulator(model)
	simulator.set_param(t_step = 0.05)
	simulator.setup()


	eval_seed = 0
	np.random.seed(eval_seed)
	max_steps = 200

	n_init_states = 1

	final_costs = np.empty(n_init_states)


	from envs.env_helper import env_creator_quad2d, ENV_CONFIG
	ENV_CONFIG.update({'reward_exponential': False,
					   'reward_scale': 1.0,
					   'eval': True,
					   'noise_scale': sigma})
	env = Gymnasium2GymWrapper(env_creator_quad2d(ENV_CONFIG))
	obs_dim = env.observation_space.shape[0]
	if save_traj == True:
		all_traj = np.empty((n_init_states,max_steps,obs_dim))

	for i in np.arange(n_init_states):
		# x_ref_lift = Psi_o(x_ref,net,NKoopman).reshape(-1,1)
		eps_cost = 0
		obs = np.array(env.reset(seed=i))
		# # mpc.x0 = x0 - x_ref_lift
		# obs = init_state
		for t in np.arange(max_steps):
			if save_traj == True:
				all_traj[i,t,:] = obs
			obs_lift = np.matrix(Psi_o(obs,net,NKoopman)).reshape(NKoopman,1)
			mpc.x0 = obs_lift
			u0 = mpc.make_step(obs_lift)
			u0 = np.array(u0[:, 0])
			obs,reward,done,info  = env.step(u0)
			print("u0", u0)
			print("obs", obs)
			eps_cost += -reward


		print("eps cost for this trial", eps_cost)
		final_costs[i] = eps_cost

	print(f"mean cost for tr_seed = {tr_seed}, sigma = {sigma}, euler={euler}", np.mean(final_costs), np.std(final_costs))
	if save_traj == True:
		filename = f"traj_log/samples={samples}_koopman_tr_seed={tr_seed}_eval_seed={eval_seed}_sigma={sigma}_euler={euler}.npy"
		os.makedirs(os.path.dirname(filename), exist_ok = True)
		with open(filename, 'wb') as f:
			np.save(f, all_traj)
	final_costs_file = f"cost_log/samples={samples}_koopman_tr_seed={tr_seed}_eval_seed={eval_seed}_sigma={sigma}_euler={euler}.npy"
	os.makedirs(os.path.dirname(final_costs_file), exist_ok = True)
	with open(final_costs_file, 'wb') as f:
		np.save(f, final_costs)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--tr_seed", default=0, type=int) # specifies training seed to evalute
	parser.add_argument("--save_traj", default = "False")
	parser.add_argument("--samples", default = 80000) #tr samples with which model trained
	parser.add_argument("--sigma", type = float, default = 1.0)
	parser.add_argument("--euler", default = "False")
	args = parser.parse_args()
	save_traj = True if args.save_traj == "True" else False
	sigma = args.sigma
	euler = True if args.euler == "True" else False
	tr_seed = args.tr_seed
	samples = args.samples

	main(tr_seed = tr_seed, save_traj = save_traj, samples = samples, sigma = sigma, euler = euler)