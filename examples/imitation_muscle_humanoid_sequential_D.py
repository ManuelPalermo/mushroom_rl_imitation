from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.environments.mujoco_envs import HumanoidGait
from mushroom_rl.utils.preprocessors import MinMaxPreprocessor
from mushroom_rl.utils.callbacks.plot_dataset import PlotDataset

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, episodes_length

from mushroom_rl_imitation.imitation.gail_sequential_D import GAIL
from mushroom_rl_imitation.imitation.vail_sequential_D import VAIL


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
    def __init__(self, input_shape, output_shape, n_features, seq_size, **kwargs):
        super(DiscriminatorNetwork, self).__init__()

        n_input = input_shape[-1] * seq_size
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
        x = x.reshape(x.shape[0], -1)  # flatten time dimension
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
    network_layers_discriminator = (512, 256)

    lr_actor = 5e-5
    lr_critic = 2e-4
    lr_discriminator = 1e-4

    weight_decay_actor = 0.0
    weight_decay_critic = 0.0
    weight_decay_discriminator = 0.0

    n_epochs_policy = 3
    batch_size_policy = 256
    batch_size_critic = 256
    clip_eps_ppo = .2
    gae_lambda = .95
    policy_std_0 = 0.3

    n_epochs_discriminator = 2
    batch_size_discriminator = 256
    seq_size_discriminator = 2

    discrim_obs_mask = np.arange(0, 27)
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
                         batch_size=batch_size_critic,
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
                                seq_size=seq_size_discriminator,
                                input_shape=discrim_input_shape,
                                output_shape=(1,),
                                n_features=network_layers_discriminator,
                                use_cuda=use_cuda,
                                )

    alg_params = dict(actor_optimizer={'class':  optim.Adam,
                                       'params': {'lr':           lr_actor,
                                                  'weight_decay': weight_decay_actor}},
                      n_epochs_policy=n_epochs_policy,
                      n_epochs_discriminator=n_epochs_discriminator,
                      batch_size_policy=batch_size_policy,
                      disc_seq_size=seq_size_discriminator,
                      eps_ppo=clip_eps_ppo,
                      lam=gae_lambda,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask,
                      quiet=True
                      )

    # TorchApproximator parameters (used for behaviour cloning)
    torch_approx_params = dict(batch_size=128,
                               optimizer={'class':  optim.Adam,
                                          'params': {'lr':           1e-3,
                                                     'weight_decay': 1e-5}},
                               loss=torch.nn.MSELoss(),
                               )
    policy_params = {**policy_params, **torch_approx_params}

    # load expert training data -> (select only joints from trajectories to compare(qpos/qvel))
    demonstrations = _load_demos_selected_discriminator(mdp.info)

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
    network_layers_discriminator = (512, 256)

    lr_actor = 5e-5
    lr_critic = 2e-4
    lr_discriminator = 1e-4

    weight_decay_actor = 0.0
    weight_decay_critic = 0.0
    weight_decay_discriminator = 0.0

    n_epochs_policy = 3
    batch_size_policy = 256
    batch_size_critic = 256
    clip_eps_ppo = .2
    gae_lambda = .95
    policy_std_0 = 0.3

    n_epochs_discriminator = 2
    batch_size_discriminator = 256
    lr_beta = 1e-5
    d_noise_vector_size = 32
    info_constraint = 0.5
    seq_size_discriminator = 2

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
                         batch_size=batch_size_critic,
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
                                seq_size=seq_size_discriminator,
                                input_shape=discrim_input_shape,
                                output_shape=(1,),
                                n_features=network_layers_discriminator,
                                use_cuda=use_cuda,
                                )

    alg_params = dict(actor_optimizer={'class':  optim.Adam,
                                       'params': {'lr':           lr_actor,
                                                  'weight_decay': weight_decay_actor}},
                      n_epochs_policy=n_epochs_policy,
                      n_epochs_discriminator=n_epochs_discriminator,
                      batch_size_policy=batch_size_policy,
                      disc_seq_size=seq_size_discriminator,
                      eps_ppo=clip_eps_ppo,
                      lam=gae_lambda,
                      info_constraint=info_constraint,
                      lr_beta=lr_beta,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask,
                      quiet=True
                      )

    # TorchApproximator parameters (used for behaviour cloning)
    torch_approx_params = dict(batch_size=128,
                               optimizer={'class':  optim.Adam,
                                          'params': {'lr':           1e-3,
                                                     'weight_decay': 1e-5}},
                               loss=torch.nn.MSELoss(),
                               )
    policy_params = {**policy_params, **torch_approx_params}

    # load expert training data -> (select only joints from trajectories to compare(qpos/qvel))
    demonstrations = _load_demos_selected_discriminator(mdp.info)

    agent = VAIL(mdp_info=mdp_info, policy_class=GaussianTorchPolicy, policy_params=policy_params,
                 discriminator_params=discriminator_params, critic_params=critic_params,
                 demonstrations=demonstrations, **alg_params)
    return agent


def _load_demos_selected_discriminator(mdp_info):
    # load expert training data -> (select only joints from trajectories to compare(qpos/qvel))
    expert_files = np.load("expert_data/humanoid_mocap_trajectory.npz")
    states = expert_files["trajectory_data"].T[:, 2:29]

    from mushroom_rl.utils.spaces import Box
    norm_info = deepcopy(mdp_info)
    norm_info.observation_space = Box(low=norm_info.observation_space._low[2:29],
                                      high=norm_info.observation_space._high[2:29])
    normalizer = MinMaxPreprocessor(mdp_info=norm_info)

    normalizer.set_state(dict(mean=np.mean(states, axis=0),
                              var=1*(np.std(states, axis=0)**2),
                              count=1))

    norm_states = np.array([normalizer(st) for st in states])
    demonstrations = dict(states=norm_states)
    return demonstrations


def init_policy_with_bc(agent):
    # load expert training data -> (select only joints from trajectories to compare(qpos/qvel))
    expert_files = np.load("expert_data/expert_dataset_humanoid_muscles.npz")
    states = expert_files["obs"]
    actions = expert_files["actions"]

    # initialize policy mu network through behaviour cloning
    agent.policy._mu.fit(states, actions,
                         n_epochs=100, patience=10)


def _create_env():
    mdp = HumanoidGait(gamma=0.99, horizon=1000, n_intermediate_steps=10,
                       goal_reward="trajectory",
                       goal_reward_params=dict(use_error_terminate=False),
                       use_muscles=True,
                       obs_avg_window=1, act_avg_window=1)
    return mdp


def evaluate_dataset(dataset, mdp_info):
    J_mean = np.mean(compute_J(dataset, mdp_info.gamma))
    R_mean = np.mean(compute_J(dataset))
    ep_len = np.mean(episodes_length(dataset))
    return J_mean, R_mean, ep_len


def experiment(algorithm, init_bc=False):
    mdp = _create_env()

    if algorithm == "GAIL":
        agent = _create_gail_agent(mdp)
    elif algorithm == "VAIL":
        # train new agent with gail, following expert trajectories
        agent = _create_vail_agent(mdp)
    else:
        raise NotImplementedError

    # normalization callback
    normalizer = MinMaxPreprocessor(mdp_info=mdp.info)

    # dataset plotter callback
    plotter = PlotDataset(mdp.info, obs_normalized=True)

    # Algorithm(with normalizing and plotting)
    core = Core(agent, mdp, callback_step=plotter, preprocessors=[normalizer])

    # evaluate untrained policy
    dataset = core.evaluate(n_episodes=10)
    print('Before {} ->  J: {},  R: {},  Len_ep: {},  Entropy: {}'
          .format(algorithm, *evaluate_dataset(dataset, mdp.info), agent.policy.entropy()))

    if init_bc:
        # initialize policy with bc
        init_policy_with_bc(agent)
        dataset = core.evaluate(n_episodes=10)
        print('After BC ->  J: {},  R: {},  Len_ep: {},  Entropy: {}'
              .format(*evaluate_dataset(dataset, mdp.info), agent.policy.entropy()))

    epoch_js = []
    # gail train loop
    for it in range(100):
        core.learn(n_steps=20481, n_steps_per_fit=2048)
        dataset = core.evaluate(n_episodes=10)
        J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
        R_mean = np.mean(compute_J(dataset))
        ep_len = np.mean(episodes_length(dataset))
        print('Epoch: {}  ->  J: {},  R: {},  Len_ep: {},  Entropy: {}'
              .format(str(it), J_mean,R_mean, ep_len,agent.policy.entropy()))
        epoch_js.append(J_mean)

    print("--- The train has finished ---")
    import matplotlib.pyplot as plt
    plt.plot(epoch_js)
    plt.show()
    input()

    # evaluate trained policy
    dataset = core.evaluate(n_episodes=25)
    print('After {} ->  J: {},  R: {},  Len_ep: {},  Entropy: {}'
          .format(algorithm, *evaluate_dataset(dataset, mdp.info), agent.policy.entropy()))


if __name__ == "__main__":
    # not tested yet(training but no results so far)
    algorithm = ["GAIL", "VAIL"]
    experiment(algorithm=algorithm[0], init_bc=False)