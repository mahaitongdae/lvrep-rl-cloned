import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
from tensorflow.python.training.summary_io import summary_iterator

def extract_data_from_events(path, tags):
    data = {key : [] for key in tags+['training_iteration']}
    for e in summary_iterator(path):
        for v in e.summary.value:
            if v.tag in tags :
                data.get('training_iteration').append(int(e.step))
                data.get(v.tag).append(v.simple_value)

    return pd.DataFrame.from_dict(data)

def plot(data_source = 'events'):
    sns.set(style='darkgrid', font_scale=1)
    sns.set_palette([(0.0, 0.24705882352941178, 1.0),
                     (0.011764705882352941, 0.9294117647058824, 0.22745098039215686),
                     (0.9098039215686274, 0.0, 0.043137254901960784),
                     (0.5411764705882353, 0.16862745098039217, 0.8862745098039215),
                     (1.0, 0.7686274509803922, 0.0),
                     (0.0, 0.8431372549019608, 1.0)])

    # noisy Pendulum
    # path_dict = {
    #     'Nystrom_1024'          : '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/good_results/rfsac_nystrom_True_rf_num_1024_sample_dim_1024/',
    #     'Nystrom_2048_top_1024' : '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/good_results/rfsac_nystrom_True_rf_num_1024_sample_dim_2048/',
    #     'Nystrom_4096_top_1024' : '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/good_results/rfsac_nystrom_True_rf_num_1024_sample_dim_4096/',
    #     'Random_feature_2048'   : '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_False_rf_num_2048',
    #     'Random_feature_4096'   : '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_False_rf_num_4096',
    #     # 'SAC'                   : '/home/mht/ray_results/Pendulum-v1/SAC/_2023-08-25_02-56-515vwpn54u'
    # }
    #
    # title = 'Pendubot'

    # path_dict = {
    #     'Nystrom_1024': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/good_results/rfsac_nystrom_True_rf_num_1024_sample_dim_1024/',
    #     'Nystrom_2048_top_1024': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/good_results/rfsac_nystrom_True_rf_num_1024_sample_dim_2048/',
    #     'Nystrom_4096_top_1024': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/good_results/rfsac_nystrom_True_rf_num_1024_sample_dim_4096/',
    #     'Random_feature_2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_False_rf_num_2048',
    #     'Random_feature_4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_False_rf_num_4096',
    #     # 'SAC'                   : '/home/mht/ray_results/Pendulum-v1/SAC/_2023-08-25_02-56-515vwpn54u'
    # }
    #
    # title = 'Noisy Pendubot'

    path_dict = {
        'Random_feature_2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_0.5/good_results/rfsac_nystrom_False_rf_num_2048_sample_dim_1024',
        'Random_feature_4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_0.5/good_results/rfsac_nystrom_False_rf_num_4096_sample_dim_1024',

    }

    title = 'Drones'

    dfs = []
    for key, path in path_dict.items():
        # rfdim = int(rfdim)
        for dir in os.listdir(path):
            if not os.path.isdir(os.path.join(path, dir)):
                continue
            abs_path = os.path.join(os.path.join(path, dir), 'summary_files')
            if data_source == 'csv':
                df = pd.read_csv(os.path.join(path, 'progress.csv'))
            elif data_source == 'events':
                for fname in os.listdir(abs_path):
                    if fname.startswith('events'):
                        break
                df = extract_data_from_events(os.path.join(abs_path, fname), ['info/evaluation'])

            df['Algorithm'] = key
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
    for y in ['info/evaluation',]: #  'episode_reward_max',
        plt.figure(figsize=[6, 4])
        sns.lineplot(total_df, x='training_iteration', y=y, hue='Algorithm', palette='muted')
        # plt.tight_layout()
        # title = ' Pendubot'
        plt.title(title)
        plt.ylabel('')
        # plt.xlim(0, 300000)
        plt.ylim(-500, 0)
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
    plot()
