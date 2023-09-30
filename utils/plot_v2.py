import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
from matplotlib.ticker import ScalarFormatter
try:
    from tensorflow.python.training.summary_io import summary_iterator
    # we need tensorflow for extracting data from tensorboard events.
except:
    pass
try:
    from eval_v2 import eval
except:
    pass

labels = [ 'info/eval_ret',] # 'info/eval_ret', 'info/evaluation'

def extract_data_from_events(path, tags):
    data = {key : [] for key in tags+['training_iteration']}
    try:
        for e in summary_iterator(path):
            for v in e.summary.value:
                if v.tag in tags :
                    # try:
                    data.get('training_iteration').append(int(e.step))
                    data.get(v.tag).append(v.simple_value)
    except:
        pass

    return pd.DataFrame.from_dict(data)

def plot(data_source = 'events'):
    sns.set(style='darkgrid', font_scale=1.3)
    sns.set_palette([(0.0, 0.24705882352941178, 1.0),
                     (0.011764705882352941, 0.9294117647058824, 0.22745098039215686),
                     (0.9098039215686274, 0.0, 0.043137254901960784),
                     (0.5411764705882353, 0.16862745098039217, 0.8862745098039215),
                     (1.0, 0.7686274509803922, 0.0),
                     (0.0, 0.8431372549019608, 1.0)])


    # path_dict = {
    #     'Random_feature': '/home/mht/PycharmProjects/lvrep-rl-cloned/results/noisy_2d_drones/rfsac_nystrom_False_rf_num_2048',
    #     'Nystrom': '/home/mht/PycharmProjects/lvrep-rl-cloned/results/noisy_2d_drones/rfsac_nystrom_True_rf_num_4096_sample_dim_8192',
    #
    # }

    # title = 'Pendubot'
    # hue = 'Algorithm'
    # path_dict = {
    #     'random feature': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_False_rf_num_4096',
    #     # 'Nystrom': '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_4096',
    #     'noisy random feature': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_False_rf_num_4096',
    #     # 'noisy Nystrom': '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_4096'
    # }

    # title = '2D Drones'
    # hue = 'Algorithm'
    # path_dict = {
    #     'random feature': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_4096_learn_rf_True',
    #     'Nystrom': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_4096_learn_rf_False_sample_dim_4096',
    #     'noisy random feature': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_4096_learn_rf_True',
    #     'noisy Nystrom': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_4096_learn_rf_False_sample_dim_4096'
    # }

    # title = 'Pendubot Random Feature'
    #
    # title = '2D Drones Random Feature'
    # path_dict = {
    #     '2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_2048_learn_rf_True',
    #     '4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_4096_learn_rf_True',
    #     '8192': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_8192_learn_rf_True'
    # }
    #
    # title = 'Noisy 2D Drones Random Feature'
    # path_dict = {
    #     '2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_2048',
    #     '4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_4096_learn_rf_True',
    #     '8192': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_8192'
    # }
    #
    # title = '2D Drones Random Feature'
    #
    # path_dict = {
    #     'Top 2048 of 2048' : '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_2048_sample_dim_2048',
    #     # TODO: outliers
    #     'Top 2048 of 4096' : '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_2048',
    #     'Top 2048 of 8192' : '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_4096'
    # }


    title = 'Pendubot'
    hue = 'nystrom dim'
    path_dict = {
        'Top 1024 of 1024' : '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_1024',
        # TODO: outliers
        'Top 1024 of 2048' : '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_2048',
        'Top 1024 of 4096' : '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_4096'
    }

    # title = 'Noisy Pendubot'
    # path_dict = {
    #     'Top 1024 of 1024': '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_1024',
    #     'Top 1024 of 2048': '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_2048',
    #     'Top 1024 of 4096': '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_4096'
    # }

    dfs = []
    for key, path in path_dict.items():
        # rfdim = int(rfdim)
        for dir in os.listdir(path):
            if not os.path.isdir(os.path.join(path, dir)):
                continue
            if dir.startswith('skip'):
                continue
            abs_path = os.path.join(os.path.join(path, dir), 'summary_files')
            if data_source == 'csv':
                df = pd.read_csv(os.path.join(path, 'progress.csv'))
            elif data_source == 'events':
                for fname in os.listdir(abs_path):
                    if fname.startswith('events'):
                        break
                df = extract_data_from_events(os.path.join(abs_path, fname), labels)


            df[hue] = key
            df['training_iteration'] = df['training_iteration'] / 1e4
            # df['episode_reward_evaluated'] = np.log(df['episode_reward_mean'] / 200.) / 10. * 200
            # if rfdim.startswith('SAC'):
            #     df['exp_setup'] = rfdim
            # elif rfdim.endswith('thexp'):
            #     df['exp_setup'] = 'thexp'
            # elif rfdim.endswith('exp'):
            #     df['exp_setup'] = 'sinthexp'
            # elif rfdim.endswith('th'):
            #     df['exp_setup'] = 'th'
            # else:
            #     df['exp_setup'] = 'sinth'
            dfs.append(df)

    total_df = pd.concat(dfs, ignore_index=True)
    for y in labels: #  'episode_reward_max',
        plt.figure(figsize=[6, 4])
        sns.lineplot(total_df, x='training_iteration', y=y, hue=hue, palette='muted')
        # plt.tight_layout()
        # title = ' Pendubot'
        plt.title(title)
        plt.ylabel('episodic return')
        # plt.xlim(0, 300000)
        if title == '2D Drones':
            plt.ylim(-200, 0)
        plt.xlabel(r'training iterations ($\times 10^4$)')
        # plt.ticklabel_format(axis='x', style='scientific')
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_scientific(True)
        # plt.gca().xaxis.set_major_formatter(formatter)
        plt.tight_layout()
        # plt.show()
        figpath = '/home/mht/PycharmProjects/lvrep-rl-cloned/fig/' + title + y.split('/')[1] + '.pdf'
        plt.savefig(figpath)

def plot_bar():
    sns.set(style='darkgrid', font_scale=1)
    sns.set_palette([(0.0, 0.24705882352941178, 1.0),
                     (0.011764705882352941, 0.9294117647058824, 0.22745098039215686),
                     (0.9098039215686274, 0.0, 0.043137254901960784),
                     (0.5411764705882353, 0.16862745098039217, 0.8862745098039215),
                     (1.0, 0.7686274509803922, 0.0),
                     (0.0, 0.8431372549019608, 1.0)])
    # title = '2D Drones Random Feature'
    # path_dict = {
    #     '2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_2048_learn_rf_True',
    #     '4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_4096_learn_rf_True',
    #     '8192': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_8192_learn_rf_True'
    # }
    title = '2D Drones Nystrom'
    path_dict = {
        '2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_2048_learn_rf_True',
        '4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_4096_learn_rf_True',
        '8192': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_8192_learn_rf_True'
    }
    # title = '2D Drones Random Feature'
    # path_dict = {
    #     '2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_2048_sample_dim_8192',
    #     '4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_2048_sample_dim_4096',
    #     '8192': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_4096_sample_dim_4096',
    # }

    # title = 'Pendubot Random Feature'
    # path_dict = {
    #     '2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_2048_learn_rf_True',
    #     '4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_4096_learn_rf_True',
    #     '8192': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_8192_learn_rf_True'
    # }

    # title = 'Pendubot Random Feature'
    # path_dict = {
    #     '2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_2048_learn_rf_False',
    #     '4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_4096_learn_rf_False',
    #     '8192': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_8192_learn_rf_False'
    # }

    # title = 'Noisy Pendubot Random Feature'
    # path_dict = {
    #     '2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_2048_learn_rf_False',
    #     '4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_4096_learn_rf_False',
    #     '8192': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_8192_learn_rf_False'
    # }


    # title = 'Pendubot'
    # path_dict = {
    #     'Top 1024 of 1024' : '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_1024',
    #     # TODO: outliers
    #     'Top 1024 of 2048' : '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_2048',
    #     'Top 1024 of 4096' : '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_4096'
    # }
    #
    # title = 'Noisy Pendubot'
    # path_dict = {
    #     'Top 1024 of 1024': '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_1024',
    #     'Top 1024 of 2048': '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_2048',
    #     'Top 1024 of 4096': '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_4096'
    # }

    dfs = []
    for key, path in path_dict.items():
        best_mean = -1e6
        best_ep_rets = None
        for dir in os.listdir(path):
            if not os.path.isdir(os.path.join(path, dir)):
                continue
            try:
                ep_rets = eval(os.path.join(path, dir))
            except:
                continue
            mean = np.mean(ep_rets)
            if mean > best_mean:
                best_mean = mean
                best_ep_rets = ep_rets
        df = pd.DataFrame.from_dict({'ep_rets': [best_ep_ret for best_ep_ret in best_ep_rets]}) # / 1.8
        df['Algorithm'] = key
        dfs.append(df)

    total_df = pd.concat(dfs, ignore_index=True)
    plt.figure(figsize=[6, 4])
    sns.barplot(total_df, x='Algorithm', y='ep_rets', palette='muted')
    plt.title(title)
    plt.ylabel('')
    plt.xlabel('training iterations')
    plt.tight_layout()
    plt.show()
    # figpath = '/home/mht/PycharmProjects/rllib-random-feature/fig/' + title + '.pdf'
    # plt.savefig(figpath)



def plot_pendulum():
    sns.set(style='darkgrid', font_scale=1)
    sns.set_palette([(0.0, 0.24705882352941178, 1.0),
                     (0.011764705882352941, 0.9294117647058824, 0.22745098039215686),
                     (0.9098039215686274, 0.0, 0.043137254901960784),
                     (0.5411764705882353, 0.16862745098039217, 0.8862745098039215),
                     (1.0, 0.7686274509803922, 0.0),
                     (0.0, 0.8431372549019608, 1.0)])
    path_dict = {
        'Random Feature SAC': '/home/mht/ray_results/SAC_Pendulum-v1_2023-04-24_09-36-31jrzl7hdp',
        'SAC' : '/home/mht/ray_results/SAC_Pendulum-v1_2023-04-23_19-18-33qzefa_7_',
        # '16384': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_09-18-428jl_v2ly',
        # '32768': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_18-58-048lea_yvt'
    }

    dfs = []
    for rfdim, path in path_dict.items():
        # rfdim = int(rfdim)
        df = pd.read_csv(os.path.join(path, 'progress.csv'))
        df['algorithm'] = rfdim
        a = 0
        dfs.append(df)

    total_df = pd.concat(dfs, ignore_index=True)
    for y in ['episode_reward_mean', ]: # 'episode_reward_min', 'episode_reward_max', 'episode_len_mean'
        plt.figure(figsize=[6, 4])
        sns.lineplot(total_df, x='training_iteration', y=y, hue='algorithm', palette='muted')
        plt.tight_layout()
        plt.xlim([-2, 500])
        plt.title('Mean episodic return')
        plt.ylabel('')
        plt.xlabel('training iterations')
        plt.tight_layout()
        # plt.show()
        figpath = '/home/mht/PycharmProjects/rllib_random_feature/fig/pen_' + y + '.png'
        plt.savefig(figpath)

if __name__ == '__main__':
    plot_bar()
