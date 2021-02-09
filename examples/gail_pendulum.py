from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mushroom_rl.utils.callbacks import PlotDataset

from mushroom_rl.utils.preprocessors import MinMaxPreprocessor

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.environments import Gym
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl_imitation.imitation import GAIL, VAIL


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


def _create_gail_agent(mdp, expert_data, disc_only_state=False, **kwargs):
    use_cuda = False    # torch.cuda.is_available()

    mdp_info = deepcopy(mdp.info)

    # Settings
    network_layers_actor = (128, 64)
    network_layers_critic = (128, 64)
    network_layers_discriminator = (128, 64)

    lr_actor = 3e-4
    lr_critic = 3e-4
    lr_discriminator = 3e-4

    weight_decay_actor = 0.0
    weight_decay_critic = 0.0
    weight_decay_discriminator = 0.0

    n_epochs_policy = 3
    batch_size_policy = 128
    batch_size_critic = 128
    clip_eps_ppo = .2
    gae_lambda = .95
    policy_std_0 = 0.3

    n_epochs_discriminator = 1
    batch_size_discriminator = 128

    discrim_obs_mask = np.arange(mdp_info.observation_space.shape[0])
    discrim_act_mask = [] if disc_only_state else np.arange(mdp_info.action_space.shape[0])
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
                                input_shape=discrim_input_shape,
                                n_features=network_layers_discriminator,
                                use_cuda=use_cuda,
                                )

    alg_params = dict(actor_optimizer={'class':  optim.Adam,
                                       'params': {'lr':           lr_actor,
                                                  'weight_decay': weight_decay_actor}},
                      n_epochs_policy=n_epochs_policy,
                      n_epochs_discriminator=n_epochs_discriminator,
                      batch_size_policy=batch_size_policy,
                      eps_ppo=clip_eps_ppo,
                      lam=gae_lambda,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask
                      )


    # TorchApproximator parameters (used for behaviour cloning)
    torch_approx_params = dict(batch_size=128,
                               optimizer={'class':  optim.Adam,
                                          'params': {'lr':           1e-3,
                                                     'weight_decay': 1e-5}},
                               loss=torch.nn.MSELoss(),
                               )
    policy_params = {**policy_params, **torch_approx_params}

    agent = GAIL(mdp_info=mdp_info, policy_class=GaussianTorchPolicy, policy_params=policy_params,
                 discriminator_params=discriminator_params, critic_params=critic_params,
                 demonstrations=expert_data, **alg_params)
    return agent


def _create_vail_agent(mdp, expert_data, disc_only_state=False, n_expert_samples=1e32, **kwargs):
    use_cuda = False    # torch.cuda.is_available()

    mdp_info = deepcopy(mdp.info)

    # Settings
    network_layers_actor = (128, 64)
    network_layers_critic = (128, 64)
    network_layers_discriminator = 128

    lr_actor = 3e-4
    lr_critic = 3e-4
    lr_discriminator = 3e-4

    weight_decay_actor = 0.0
    weight_decay_critic = 0.0
    weight_decay_discriminator = 0.0

    n_epochs_policy = 3
    batch_size_policy = 128
    batch_size_critic = 128
    clip_eps_ppo = .2
    gae_lambda = .95
    policy_std_0 = 0.3

    n_epochs_discriminator = 1
    d_noise_vector_size = 8
    batch_size_discriminator = 128
    info_constraint = 0.5
    lr_beta = 3e-5

    discrim_obs_mask = np.arange(mdp_info.observation_space.shape[0])
    discrim_act_mask = [] if disc_only_state else np.arange(mdp_info.action_space.shape[0])
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
                      eps_ppo=clip_eps_ppo,
                      lam=gae_lambda,
                      info_constraint=info_constraint,
                      lr_beta=lr_beta,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask
                      )

    # TorchApproximator parameters (used for behaviour cloning)
    torch_approx_params = dict(batch_size=128,
                               optimizer={'class':  optim.Adam,
                                          'params': {'lr':           1e-3,
                                                     'weight_decay': 1e-5}},
                               loss=torch.nn.MSELoss(),
                               )
    policy_params = {**policy_params, **torch_approx_params}

    agent = VAIL(mdp_info=mdp_info, policy_class=GaussianTorchPolicy, policy_params=policy_params,
                 discriminator_params=discriminator_params, critic_params=critic_params,
                 demonstrations=expert_data, **alg_params)
    return agent


def prepare_expert_data(data_path, n_samples, normalizer=None):
    if normalizer is None:
        normalizer = lambda x: x

    # load expert training data
    expert_files = np.load(data_path)
    inputs = normalizer(expert_files["obs"])[:int(n_samples)]
    outputs = expert_files["actions"][:int(n_samples)]
    return dict(states=inputs, actions=outputs)


def init_policy_with_bc(agent, expert_data):
    # initialize policy mu network through behaviour cloning
    agent.policy._mu.fit(expert_data["states"], expert_data["actions"],
                         n_epochs=100, patience=10)


def experiment(algorithm, init_bc=False, discr_only_state=False):
    mdp = Gym(name='Pendulum-v0', horizon=200, gamma=0.99)
    horizon = mdp.info.horizon

    # prepare expert samples(no need to normalize the data as
    # a Normalizer was used when extracting it)
    n_trajectories = 10
    expert_data = prepare_expert_data(
            data_path="expert_data/expert_dataset_Pendulum-v0.npz",
            n_samples=horizon*n_trajectories, normalizer=None,
    )

    if algorithm == "GAIL":
        agent = _create_gail_agent(mdp, expert_data, discr_only_state)
    elif algorithm == "VAIL":
        # train new agent with gail, following expert trajectories
        agent = _create_vail_agent(mdp, expert_data, discr_only_state)
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
    J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
    print('Before {} -> J: {}, Entropy: {}'.format(algorithm, J_mean, agent.policy.entropy()))

    if init_bc:
        init_policy_with_bc(agent, expert_data=expert_data)
        # evaluate untrained policy
        dataset = core.evaluate(n_episodes=10)
        J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
        print('After BC -> J: {}, Entropy: {}'.format(J_mean, agent.policy.entropy()))


    epoch_js = []
    # gail train loop
    for it in range(100):
        core.learn(n_steps=10000, n_steps_per_fit=1024, render=False)
        dataset = core.evaluate(n_episodes=10, render=False)
        J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
        epoch_js.append(J_mean)
        print('Epoch: {}  ->  J: {}, Entropy: {}'.format(str(it), J_mean, agent.policy.entropy()))


    print("--- The train has finished ---")
    import matplotlib.pyplot as plt
    plt.plot(epoch_js)
    plt.show()
    input()

    # evaluate trained policy
    dataset = core.evaluate(n_episodes=25)
    J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
    print('After {} -> J: {}, Entropy: {}'.format(algorithm, J_mean, agent.policy.entropy()))


if __name__ == "__main__":
    algorithm = ["GAIL", "VAIL"][1]
    experiment(algorithm=algorithm, init_bc=True, discr_only_state=True)