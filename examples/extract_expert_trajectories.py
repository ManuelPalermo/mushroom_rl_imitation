import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.utils.callbacks import PlotDataset
from mushroom_rl.utils.preprocessors import MinMaxPreprocessor, StandardizationPreprocessor

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.environments import Gym
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, parse_dataset


class CriticNetwork(nn.Module):
    # For sac agent
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features[0])
        self._h2 = nn.Linear(n_features[0], n_features[1])
        self._h3 = nn.Linear(n_features[1], n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)
        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features[0])
        self._h1 = nn.Linear(n_features[0], n_features[1])
        self._out = nn.Linear(n_features[1], n_output)

        nn.init.xavier_uniform_(self._in.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._out.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, x, **kwargs):
        x = torch.squeeze(x, 1).float()
        x = F.relu(self._in(x))
        x = F.relu(self._h1(x))
        out = self._out(x)
        return out


def _create_sac_agent(mdp):
    # Settings
    initial_replay_size = 500
    max_replay_size = 50000
    batch_size = 128
    n_features = (128, 64)
    warmup_transitions = 1000
    tau = 0.005
    lr_alpha = 3e-4
    target_entropy = 10 * (-np.prod(mdp.info.action_space.shape).astype(np.float32))

    # most simple environments run faster on CPU than with cuda
    use_cuda = torch.cuda.is_available()

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=mdp.info.action_space.shape,
                           use_cuda=use_cuda)
    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape,
                              use_cuda=use_cuda)

    actor_optimizer = {'class':  optim.Adam,
                       'params': {'lr': 3e-4}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class':  optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    # Agent
    agent = SAC(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None, target_entropy=target_entropy)
    return agent


def _create_mdp(env_id, horizon=1000, gamma=0.99):
    mdp = Gym(env_id, horizon=horizon, gamma=gamma)
    return mdp


def train_expert_from_scratch_and_save_trajectories(env_id, n_trajectories,
                                                    expert_train_epochs=10,
                                                    horizon=1000):
    # train expert from scratch
    mdp = _create_mdp(env_id=env_id, horizon=horizon)
    expert_agent = _create_sac_agent(mdp)

    # normalization callback(try to use limited range if mdp
    # observation limits are available)
    try:
        normalizer = MinMaxPreprocessor(mdp_info=mdp.info)
    except:
        normalizer = StandardizationPreprocessor(mdp_info=mdp.info)

    # plotting callback
    plotter = PlotDataset(mdp.info, obs_normalized=True)

    # Algorithm(with normalizing and plotting)
    core_expert = Core(expert_agent, mdp, callback_step=plotter, preprocessors=[normalizer])

    for it in range(expert_train_epochs):
        core_expert.learn(n_steps=10000, n_steps_per_fit=1)
        dataset = core_expert.evaluate(n_episodes=10, render=False)
        J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
        print('Epoch: {}, J: {}'.format(it, J_mean))

    # visualize and save expert trajectories
    input()
    core_expert.evaluate(n_episodes=5, render=True)
    expert_dataset = core_expert.evaluate(n_episodes=n_trajectories, render=False)
    states, actions, rewards, next_states, absorbing, last = \
        parse_dataset(expert_dataset)

    os.makedirs("./expert_data/", exist_ok=True)
    np.savez(file="./expert_data/expert_dataset_{}_{}.npz".format(env_id,
                                                                  round(float(np.mean(compute_J(expert_dataset))), 2)),
             obs=states,
             actions=actions,
             episode_starts=np.roll(last, -1),
             episode_returns=rewards)


if __name__ == "__main__":
    train_expert_from_scratch_and_save_trajectories(
            env_id='Hopper-v2', n_trajectories=100,
            expert_train_epochs=20, horizon=1000)