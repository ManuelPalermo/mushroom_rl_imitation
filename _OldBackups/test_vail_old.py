from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.utils.preprocessors import NormalizationBoxedPreprocessor

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.environments import Gym
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl_imitation._OldBackups.vail_old import VAIL
from mushroom_rl_imitation._OldBackups.dataset_from_baselines import ExpertDataset


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


def _create_vail_agent(mdp):
    use_cuda = False

    mdp_info = deepcopy(mdp.info)

    # Settings
    network_layers_actor = (128, 64)
    network_layers_critic = (128, 64)
    network_layers_discriminator = 128

    lr_actor = 3e-4
    lr_critic = 3e-4
    lr_discriminator = 3e-4
    lr_beta = 1e-4

    weight_decay_actor = 0.0
    weight_decay_critic = 0.0
    weight_decay_discriminator = 0.0

    n_epochs_discriminator = 1
    n_epochs_policy = 3
    clip_eps_ppo = .2
    gae_lambda = .95
    policy_std_0 = 0.3

    d_noise_vector_size = 4
    info_constraint = 0.5

    batch_size_policy = 128
    batch_size_discriminator = 128

    state_act_shape = (mdp_info.observation_space.shape[0] + mdp_info.action_space.shape[0],)

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

    discriminator_params = dict(input_shape=state_act_shape[-1],
                                output_shape=1,
                                n_features=network_layers_discriminator,
                                z_size=d_noise_vector_size,
                                )

    alg_params = dict(actor_optimizer={'class':  optim.Adam,
                                       'params': {'lr':           lr_actor,
                                                  'weight_decay': weight_decay_actor}},
                      discriminator_optimizer={'class':  optim.Adam,
                                               'params': {'lr':           lr_discriminator,
                                                          'weight_decay': weight_decay_discriminator}},

                      n_epochs_policy=n_epochs_policy,
                      n_epochs_discriminator=n_epochs_discriminator,
                      batch_size_policy=batch_size_policy,
                      batch_size_discriminator=batch_size_discriminator,
                      eps_ppo=clip_eps_ppo,
                      lam=gae_lambda,
                      info_constraint=info_constraint,
                      lr_beta=lr_beta,
                      quiet=True
                      )

    # TorchApproximator parameters (used for behaviour cloning)
    torch_approx_params = dict(batch_size=128,
                               optimizer={'class':  optim.Adam,
                                          'params': {'lr':           1e-3,
                                                     'weight_decay': 1e-5}},
                               loss=torch.nn.MSELoss())
    policy_params = {**policy_params, **torch_approx_params}

    # batch size should have same size as n_steps_per_fit
    dataset = ExpertDataset(expert_path='../expert_data/expert_dataset_pendulum_SAC_120.npz',
                            train_fraction=0.99, batch_size=1024,
                            randomize=True, verbose=False, traj_limitation=10,
                            sequential_preprocessing=True)

    agent = VAIL(mdp_info=mdp_info, policy_class=GaussianTorchPolicy, policy_params=policy_params,
                 discriminator_params=discriminator_params, critic_params=critic_params,
                 demonstrations=dataset, **alg_params)
    return agent


def init_policy_with_bc(agent, normalizer=None):
    if normalizer is None:
        normalizer = lambda x: x

    # load expert training data
    expert_files = np.load("../expert_data/expert_dataset_pendulum_SAC_120.npz")
    inputs = normalizer(expert_files["obs"])[:200*10]
    outputs = expert_files["actions"][:200*10]

    # initialize policy mu network through behaviour cloning
    agent.policy._mu.fit(inputs, outputs, n_epochs=100, patience=10)


def test_vail(init_bc=False):
    # could use callback plotting but is working
    # bad with algorithms which fit full trajectories

    # train new expert
    # train_expert_from_scratch_and_save_trajectories()
    # input()

    from mushroom_rl_imitation.wrapping_envs.PlottingEnv import PlottingEnv
    mdp = PlottingEnv(env_class=Gym, env_kwargs=dict(name='Pendulum-v0',
                                                     horizon=200, gamma=0.99))

    #mdp.seed(1)
    #np.random.seed(1)
    #torch.manual_seed(1)
    #torch.cuda.manual_seed(1)

    # train new agent with gail, following expert trajectories
    agent = _create_vail_agent(mdp)

    # normalization callback
    normalizer = NormalizationBoxedPreprocessor(mdp_info=mdp.info)

    # Algorithm(with normalizing and plotting)
    core = Core(agent, mdp, preprocessors=[normalizer])
    #core = Core(agent, mdp)    # without normalization

    # evaluate untrained policy
    dataset = core.evaluate(n_episodes=10)
    J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
    print('Before Vail -> J: {}, Entropy: {}'.format(J_mean, agent.policy.entropy()))

    if init_bc:
        init_policy_with_bc(agent, normalizer=normalizer)
        # evaluate untrained policy
        dataset = core.evaluate(n_episodes=10)
        J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
        print('After BC -> J: {}, Entropy: {}'.format(J_mean, agent.policy.entropy()))

    epoch_js = []
    # gail train loop
    for it in range(100):
        core.learn(n_steps=10000, n_steps_per_fit=1024, render=False)
        dataset = core.evaluate(n_episodes=3, render=False)
        J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
        epoch_js.append(J_mean)
        print('Epoch: {}  ->  J: {}, Entropy: {}'.format(str(it), J_mean, agent.policy.entropy()))

    print("--- The train has finished ---")
    import matplotlib.pyplot as plt
    plt.plot(epoch_js)
    plt.show()
    input()

    # evaluate trained policy
    dataset = core.evaluate(n_episodes=10)
    J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
    print('After Gail -> J: {}, Entropy: {}'.format(J_mean, agent.policy.entropy()))


test_vail(init_bc=True)