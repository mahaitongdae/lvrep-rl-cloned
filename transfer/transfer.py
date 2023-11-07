import torch
import gymnasium
import numpy as np
from differentiable_representation import NystromFeatureExtractor
from differentiable_dynamics import Pendulum3D
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BATCH_SIZE = 512

if __name__ == '__main__':
    env = gymnasium.make('Pendulum-v1')
    obs = np.vstack([env.observation_space.sample() for i in range(BATCH_SIZE)])
    action = np.vstack([env.action_space.sample() for i in range(BATCH_SIZE)])
    pendulum_3d = Pendulum3D()
    pendulum_3d_2 = Pendulum3D( m=2.0) # g=12.0,, l=2.0
    fsa1 = pendulum_3d.rollout(torch.from_numpy(obs), torch.from_numpy(action))
    fsa2 = pendulum_3d_2.rollout(torch.from_numpy(obs), torch.from_numpy(action))

    # env = gymnasium.make('Pendulum-v1')
    with torch.no_grad():  # for the ground truth
        feature_extractor = NystromFeatureExtractor(env)
        feature_extractor.set_dynamics(pendulum_3d)
        feature_extractor.sample_and_decompose()
        phi = feature_extractor.get_nystrom_feature(fsa1)

    feature_extractor2 = NystromFeatureExtractor(env)
    feature_extractor2.set_dynamics(pendulum_3d_2)
    feature_extractor2.sample_and_decompose()
    phi2 = feature_extractor.get_nystrom_feature(fsa2)

    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.Adam([
                                  # feature_extractor2.rollout_dynamics.g,
                                  # feature_extractor2.rollout_dynamics.l,
                                  feature_extractor2.rollout_dynamics.m])
    loss = torch.nn.MSELoss()
    data = dict(iter=[], m=[]) # g=[], l=[],
    for i in range(10000):
        obs = np.vstack([env.observation_space.sample() for i in range(BATCH_SIZE)])
        action = np.vstack([env.action_space.sample() for i in range(BATCH_SIZE)])
        fsa2 = pendulum_3d_2.rollout(torch.from_numpy(obs), torch.from_numpy(action))
        phi2 = feature_extractor.get_nystrom_feature(fsa2)
        supervised_loss = loss(phi.detach_(), phi2)
        optimizer.zero_grad()
        supervised_loss.backward()
        optimizer.step()
        data['iter'].append(i)
        # data['g'].append(feature_extractor2.rollout_dynamics.g.detach().item())
        # data['l'].append(feature_extractor2.rollout_dynamics.l.detach().item())
        data['m'].append(feature_extractor2.rollout_dynamics.m.detach().item())

    pendulum_3d_sysid = Pendulum3D(m=2.0) # g=12.0, l=2.0
    optimizer_sysid = torch.optim.Adam([
                                        # pendulum_3d_sysid.g,
                                        # pendulum_3d_sysid.l,
                                        pendulum_3d_sysid.m])
    data_sysid = dict(iter=[], m=[]) # g=[], l=[],
    for i in range(10000):
        obs = np.vstack([env.observation_space.sample() for i in range(BATCH_SIZE)])
        action = np.vstack([env.action_space.sample() for i in range(BATCH_SIZE)])
        fsa2 = pendulum_3d_sysid.rollout(torch.from_numpy(obs), torch.from_numpy(action))
        sysid_loss = loss(fsa2, fsa1.detach_())
        optimizer_sysid.zero_grad()
        sysid_loss.backward()
        optimizer_sysid.step()
        data_sysid['iter'].append(i)
        # data_sysid['g'].append(pendulum_3d_sysid.g.detach().item())
        # data_sysid['l'].append(pendulum_3d_sysid.l.detach().item())
        data_sysid['m'].append(pendulum_3d_sysid.m.detach().item())

    df1 = pd.DataFrame.from_dict(data)
    df1['algo'] = 'features'
    df2 = pd.DataFrame.from_dict(data_sysid)
    df2['algo'] = 'sysid'
    df = pd.concat([df1, df2], ignore_index=True)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[10, 4])
    fig, ax = plt.subplots(1, 1, figsize=[8, 8])
    # sns.lineplot(df, x='iter', y='l', hue='algo', ax=ax1)
    sns.lineplot(df, x='iter', y='m', hue='algo', ax=ax)
    # sns.lineplot(df, x='iter', y='m', hue='algo', ax=ax3)
    plt.show()