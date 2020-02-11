from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ImitationLearning.vail import VAIL
from ImitationLearning.gail import GAIL

from mushroom_rl.environments.mujoco_envs import HumanoidGait
from mushroom_rl.utils.preprocessors import NormalizationBoxedPreprocessor
from mushroom_rl.utils.callbacks.plot_dataset import PlotDataset

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, episodes_length




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


class DiscriminatorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(DiscriminatorNetwork, self).__init__()

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
                                gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, x, **kwargs):
        x = F.leaky_relu(self._in(x), 0.2)
        x = F.leaky_relu(self._h1(x), 0.2)
        out = torch.sigmoid(self._out(x))
        return out


def _create_gail_agent(mdp, **kwargs):
    use_cuda = torch.cuda.is_available()

    mdp_info = deepcopy(mdp.info)

    # Settings
    network_layers_actor = (512, 256)
    network_layers_critic = (512, 256)
    network_layers_discriminator = 512

    lr_actor = 1e-4
    lr_critic = 1e-4
    lr_discriminator = 1e-4

    weight_decay_actor = 0.0
    weight_decay_critic = 0.0
    weight_decay_discriminator = 0.0

    n_epochs_policy = 3
    batch_size_policy = 256
    clip_eps_ppo = .2
    gae_lambda = .95
    policy_std_0 = 0.1

    batch_size_discriminator = 256

    discrim_obs_mask = np.arange(mdp_info.observation_space.shape[0])
    discrim_act_mask = []
    discrim_input_shape = (len(discrim_obs_mask) + len(discrim_act_mask),)

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

    discriminator_params = dict(optimizer={'class':  optim.Adam,
                                           'params': {'lr':           lr_discriminator,
                                                      'weight_decay': weight_decay_discriminator}},
                                batch_size=batch_size_discriminator,

                                network=DiscriminatorNetwork,
                                input_shape=discrim_input_shape,
                                n_features=network_layers_discriminator,
                                use_cuda=use_cuda,
                                )

    alg_params = dict(actor_optimizer={'class':  optim.Adam,
                                       'params': {'lr':           lr_actor,
                                                  'weight_decay': weight_decay_actor}},

                      n_epochs_policy=n_epochs_policy,
                      batch_size_policy=batch_size_policy,
                      eps_ppo=clip_eps_ppo,
                      lam=gae_lambda,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask,
                      quiet=True
                      )

    # load expert training data -> (select only joints from trajectories to compare(qpos/qvel))
    expert_files = np.load("expert_data/humanoid_gait_trajectory.npz")
    states = expert_files["trajectory_data"].T[:, 2:29]

    from mushroom_rl.utils.spaces import Box
    norm_info = deepcopy(mdp.info)
    norm_info.observation_space = Box(low=norm_info.observation_space._low[2:29],
                                      high=norm_info.observation_space._high[2:29])
    normalizer = NormalizationBoxedPreprocessor(mdp_info=norm_info)

    normalizer.set_state(dict(mean=states.mean(axis=0),
                              std=states.std(axis=0),
                              count=states.shape[1]))

    demonstrations = (dict(states=normalizer(states)))

    agent = GAIL(mdp_info=mdp_info, policy_class=GaussianTorchPolicy, policy_params=policy_params,
                 discriminator_params=discriminator_params, critic_params=critic_params,
                 demonstrations=demonstrations, **alg_params)
    return agent


def _create_vail_agent(mdp, **kwargs):
    use_cuda = torch.cuda.is_available()

    mdp_info = deepcopy(mdp.info)

    # Settings
    network_layers_actor = (512, 256)
    network_layers_critic = (512, 256)
    network_layers_discriminator = 512

    lr_actor = 5e-5
    lr_critic = 1e-4
    lr_discriminator = 5e-5

    weight_decay_actor = 0.0
    weight_decay_critic = 0.0
    weight_decay_discriminator = 0.0

    n_epochs_policy = 3
    batch_size_policy = 256
    clip_eps_ppo = .2
    gae_lambda = .95
    policy_std_0 = 0.3

    batch_size_discriminator = 256
    lr_beta = 5e-6
    d_noise_vector_size = 32
    info_constraint = 0.5

    discrim_obs_mask = np.arange(0, 27) # select observations to compare(qpos/qvel)
    discrim_act_mask = []
    discrim_input_shape = (len(discrim_obs_mask) + len(discrim_act_mask),)

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

    discriminator_params = dict(optimizer={'class':  optim.Adam,
                                           'params': {'lr':           lr_discriminator,
                                                      'weight_decay': weight_decay_discriminator}},
                                batch_size=batch_size_discriminator,
                                z_size=d_noise_vector_size,

                                input_shape=discrim_input_shape,
                                output_shape=(1,),
                                n_features=network_layers_discriminator,
                                use_cuda=use_cuda,
                                )

    alg_params = dict(actor_optimizer={'class':  optim.Adam,
                                       'params': {'lr':           lr_actor,
                                                  'weight_decay': weight_decay_actor}},


                      n_epochs_policy=n_epochs_policy,
                      batch_size_policy=batch_size_policy,
                      eps_ppo=clip_eps_ppo,
                      lam=gae_lambda,
                      info_constraint=info_constraint,
                      lr_beta=lr_beta,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask,
                      quiet=True
                      )


    # load expert training data -> (select only joints from trajectories to compare(qpos/qvel))
    expert_files = np.load("expert_data/humanoid_gait_trajectory.npz")
    states = expert_files["trajectory_data"].T[:, 2:29]

    from mushroom_rl.utils.spaces import Box
    norm_info = deepcopy(mdp.info)
    norm_info.observation_space = Box(low=norm_info.observation_space._low[2:29],
                                      high=norm_info.observation_space._high[2:29])
    normalizer = NormalizationBoxedPreprocessor(mdp_info=norm_info)

    normalizer.set_state(dict(mean=states.mean(axis=0),
                              std=states.std(axis=0),
                              count=states.shape[1]))

    demonstrations = (dict(states=normalizer(states)))

    agent = VAIL(mdp_info=mdp_info, policy_class=GaussianTorchPolicy, policy_params=policy_params,
                 discriminator_params=discriminator_params, critic_params=critic_params,
                 demonstrations=demonstrations, **alg_params)
    return agent


def _create_env():
    mdp_class = HumanoidGait
    mdp_params = dict(gamma=0.99, horizon=2000, nmidsteps=10,
                      goal_reward="trajectory",
                      goal_reward_params=dict(use_error_terminate=True),
                      use_muscles=False,
                      obs_avg_window=1, act_avg_window=1)
    return mdp_class, mdp_params


def experiment(algorithm):
    from _wrapping_envs.PlottingEnv import PlottingEnv
    mdp_class, mdp_params = _create_env()
    mdp = PlottingEnv(env_class=mdp_class, env_kwargs=mdp_params)

    if algorithm == "GAIL":
        agent = _create_gail_agent(mdp)
    elif algorithm == "VAIL":
        # train new agent with gail, following expert trajectories
        agent = _create_vail_agent(mdp)
    else:
        raise NotImplementedError
    # normalization callback
    normalizer = NormalizationBoxedPreprocessor(mdp_info=mdp.info)

    # Algorithm(with normalizing and plotting)
    core = Core(agent, mdp, preprocessors=[normalizer])
    #core = Core(agent, mdp)    # without normalization

    # evaluate untrained policy
    dataset = core.evaluate(n_episodes=10)
    J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
    ep_len = np.mean(episodes_length(dataset))
    print('Before Gail ->  J: {},  Len_ep: {},  Entropy: {}'.format(J_mean, ep_len, agent.policy.entropy()))

    epoch_js = []
    # gail train loop
    for it in range(100):
        agent._env_reward_frac = np.exp(-it/3)
        core.learn(n_steps=10241, n_steps_per_fit=2048, render=False)
        dataset = core.evaluate(n_episodes=5, render=False)
        J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
        ep_len = np.mean(episodes_length(dataset))
        epoch_js.append(J_mean)
        print('Epoch: {}  ->  J: {},  Len_ep: {},  Entropy: {}'.format(str(it), J_mean, ep_len, agent.policy.entropy()))


    print("--- The train has finished ---")
    import matplotlib.pyplot as plt
    plt.plot(epoch_js)
    plt.show()

    input()

    # evaluate trained policy
    dataset = core.evaluate(n_episodes=10)
    J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
    print('After Gail -> J: {}, Entropy: {}'.format(J_mean, agent.policy.entropy()))


if __name__ == "__main__":
    # not working yet
    algorithm = ["GAIL", "VAIL"]
    experiment(algorithm=algorithm[1])