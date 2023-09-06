import torch 
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from envs.pendulum import Pendulum
from envs.quadrotor import Quadrotor
from envs.pendubot import Pendubot
from algorithms.run_min_algo import run_min_algo
import argparse
import os


def eval_traj(env, traj, cmd, horizon = 200):
    #returns the cost of the traj with associated action list 'cmd' in environment 'env'
    #with horizon steps per episode
    cost = 0.
    for i in np.arange(horizon):
        th1, th2, th1_dot, th2_dot = traj[i]
        cost += (- (np.sin(th1) - 1.) ** 2 - np.cos(th1) ** 2 - th1_dot ** 2 - 0.01 * np.sin(th2) ** 2
                 - 0.01 * (np.cos(th2) - 1.) ** 2 - 0.01 * th2_dot ** 2 - 0.01 * cmd[i] ** 2)
        print("cmd %d" % i, cmd[i])
    return cost



def main():
    seed = args.seed
    euler = True if args.euler == "True" else False
    sigma = args.sigma

    torch.set_default_tensor_type(torch.DoubleTensor)

    np.random.seed(seed)
    n_init_states = 10
    init_states = np.random.normal(scale=[0.1, 0.1, 0.1, 0.1], size=[10, 4]) + np.array([np.pi/2, 0., 0., 0.,])

    max_steps = 200
    final_opt_cost = np.empty(n_init_states)
    all_cmd_opt = np.empty((n_init_states, max_steps, ))
    all_traj = np.empty((n_init_states,max_steps, 4))


    #Create nonlinear control envs for different initial states
    for i in np.arange(n_init_states):
        # np.random.seed(i)
        env = Pendubot(horizon = max_steps, init_state = init_states[i,:], sigma = sigma,euler = euler)
        cmd_opt, _,metrics = run_min_algo(env, algo = 'ddp_linquad_reg', max_iter = 100)
        all_cmd_opt[i,:] = cmd_opt.reshape([-1])
        eval_env = Pendubot(horizon = max_steps, init_state = init_states[i,:], sigma=sigma, euler = euler)

        traj_opt,_ = eval_env.forward(cmd_opt)
        for j in np.arange(max_steps):
            all_traj[i,j,:] = traj_opt[j].numpy()

        # cost_i = metrics["cost"][-1] #take cost of final step
        cost_i = eval_traj(env,traj_opt, cmd_opt, horizon = max_steps)
        final_opt_cost[i] = cost_i
        print("initial state:", init_states[i,:])
        print("final cost", final_opt_cost[i])
        # Visualize the movement
        # env.visualize(cmd_opt)

    print(f"mean cost for seed={seed}, euler = {euler}, sigma = {sigma}", np.mean(final_opt_cost), np.std(final_opt_cost))

    traj_file = f"traj_log/ilqr_seed={seed}_euler={euler}_sigma={sigma}.npy"
    os.makedirs(os.path.dirname(traj_file), exist_ok = True)
    with open(traj_file, 'wb') as g:
        np.save(g, all_traj)

    costs_file = f"cost_log/seed={seed}_euler={euler}_sigma={sigma}.npy"
    os.makedirs(os.path.dirname(costs_file), exist_ok = True)
    with open(costs_file, 'wb') as f:
        np.save(f, final_opt_cost)

    # #Plot costs 

    # sns.lineplot(x = 'iteration', y = 'cost', data = metrics)
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--sigma", type = float, default = 0.0)
    parser.add_argument("--euler", default = False)
    args = parser.parse_args()
    main()