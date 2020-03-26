from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.callbacks import PlotDataset
from mushroom_rl.utils.preprocessors import MinMaxPreprocessor

from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.environments import Gym
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features[0])
        self._h1 = nn.Linear(n_features[0], n_features[1])
        self._out = nn.Linear(n_features[1], n_output)

        nn.init.xavier_uniform_(self._in.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self._out.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, x, **kwargs):
        x = torch.squeeze(x, 1).float()
        x = F.leaky_relu(self._in(x), 0.2)
        x = F.leaky_relu(self._h1(x), 0.2)
        out = self._out(x)
        return out



def _create_ppo_agent(mdp):
    use_cuda = False
    mdp_info = deepcopy(mdp.info)

    # Settings
    network_layers_actor = (128, 64)
    network_layers_critic = (128, 64)
    lr_actor = 3e-4
    lr_critic = 3e-4

    weight_decay_actor = 0.0
    weight_decay_critic = 0.0
    n_epochs_policy = 3
    batch_size_policy = 128
    clip_eps_ppo = .2
    gae_lambda = .95
    policy_std_0 = 0.3

    policy_params = dict(network=ActorNetwork,
                         input_shape=mdp_info.observation_space.shape,
                         output_shape=mdp_info.action_space.shape,
                         std_0=policy_std_0,
                         n_features=network_layers_actor,
                         use_cuda=use_cuda,
                         )

    critic_params = dict(network=ActorNetwork,
                         optimizer={'class':  optim.Adam,
                                    'params': {'lr':           lr_critic,
                                               'weight_decay': weight_decay_critic}},
                         loss=F.mse_loss,
                         input_shape=mdp_info.observation_space.shape,
                         output_shape=(1,),
                         n_features=network_layers_critic,
                         use_cuda=use_cuda,
                         )

    alg_params = dict(actor_optimizer={'class':  optim.Adam,
                                       'params': {'lr':           lr_actor,
                                                  'weight_decay': weight_decay_actor}},
                      n_epochs_policy=n_epochs_policy,
                      batch_size=batch_size_policy,
                      eps_ppo=clip_eps_ppo,
                      lam=gae_lambda,
                      quiet=True
                      )

    # TorchApproximator parameters (used for behaviour cloning)
    torch_approx_params = dict(batch_size=128,
                               optimizer={'class':  optim.Adam,
                                          'params': {'lr':           1e-3,
                                                     'weight_decay': 1e-5}},
                               loss=torch.nn.MSELoss())
    policy_params = {**policy_params, **torch_approx_params}

    policy = GaussianTorchPolicy(**policy_params)

    agent = PPO(mdp_info=mdp_info, policy=policy,
                critic_params=critic_params, **alg_params)

    return agent



def experiment():
    mdp = Gym(name='Pendulum-v0', horizon=200, gamma=0.99)
    agent = _create_ppo_agent(mdp)

    # normalization callback
    normalizer = MinMaxPreprocessor(mdp_info=mdp.info)

    # dataset ploter callback
    plotter = PlotDataset(mdp.info, obs_normalized=True)

    # Algorithm(with normalizing and plotting)
    core = Core(agent, mdp,
                callback_step=plotter,
                preprocessors=[normalizer])

    # test before BC
    dataset = core.evaluate(n_episodes=25, render=False)
    R_mean = np.mean(compute_J(dataset))
    J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
    print('Before BC -> J: {}, R: {}'.format(J_mean, R_mean))

    # test after BC
    # load expert training data
    expert_files = np.load("expert_data/expert_dataset_pendulum_SAC_120.npz")
    inputs = expert_files["obs"]
    outputs = expert_files["actions"]

    # train policy mu network through behaviour cloning
    agent.policy._mu.fit(inputs, outputs, n_epochs=100, patience=10)

    dataset = core.evaluate(n_episodes=25, render=False)
    R_mean = np.mean(compute_J(dataset))
    J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
    print('After BC -> J: {}, R: {}'.format(J_mean, R_mean))


experiment()