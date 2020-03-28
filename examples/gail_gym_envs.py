
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mushroom_rl.utils.callbacks import PlotDataset

from ImitationLearning.vail import VAIL
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor, MinMaxPreprocessor

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.environments import Gym
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from ImitationLearning.gail import GAIL

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
    network_layers_critic = (128, 128)
    network_layers_discriminator = (128, 64)

    lr_actor = 1e-3
    lr_critic = 1e-3
    lr_discriminator = 1e-3

    weight_decay_actor = 0.0
    weight_decay_critic = 0.0
    weight_decay_discriminator = 1e-5

    n_epochs_policy = 3
    batch_size_policy = 256
    clip_eps_ppo = .2
    gae_lambda = .95
    policy_std_0 = 0.5

    batch_size_discriminator = 256

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


    # TorchApproximator parameters (used for behaviour cloning)
    torch_approx_params = dict(batch_size=256,
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


def _create_vail_agent(mdp, expert_data, disc_only_state=False, **kwargs):
    use_cuda = False    # torch.cuda.is_available()

    mdp_info = deepcopy(mdp.info)

    # Settings
    network_layers_actor = (128, 64)
    network_layers_critic = (128, 64)
    network_layers_discriminator = 128

    lr_actor = 1e-3
    lr_critic = 1e-3
    lr_discriminator = 1e-3

    weight_decay_actor = 0.0
    weight_decay_critic = 0.0
    weight_decay_discriminator = 1e-5

    n_epochs_policy = 3
    batch_size_policy = 256
    clip_eps_ppo = .2
    gae_lambda = .95
    policy_std_0 = 0.5

    d_noise_vector_size = 16
    batch_size_discriminator = 256
    info_constraint = 0.5
    lr_beta = 1e-4

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

    # TorchApproximator parameters (used for behaviour cloning)
    torch_approx_params = dict(batch_size=256,
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


def prepare_expert_data(data_path, n_samples, data_normalizer=None):
    if data_normalizer is None:
        data_normalizer = lambda x: x

    # load expert training data
    expert_files = np.load(data_path)
    inputs = data_normalizer(expert_files["obs"])[:int(n_samples)]
    outputs = expert_files["actions"][:int(n_samples)]
    return dict(states=inputs, actions=outputs)

def warmup_running_norm(normalizer, expert_data):
    #for d in expert_data["states"].copy():
    #    normalizer(d)
    # produces same results as above, but less readable(but can set num samples -> increase initial catcup rate)
    n = 25  # low n value so it has high catchup to actual statistics
    normalizer.set_state(dict(mean=np.mean(expert_data["states"], axis=0),
                              std=n * (np.std(expert_data["states"], axis=0)**2),
                              count=n))


def init_policy_with_bc(agent, expert_data):
    # initialize policy mu network through behaviour cloning
    agent.policy._mu.fit(expert_data["states"], expert_data["actions"],
                         n_epochs=50, patience=10)


def experiment(algorithm, env_kwargs, n_expert_trajectories,
               init_bc=False, discr_only_state=False):

    mdp = Gym(**env_kwargs)
    horizon = mdp.info.horizon
    expert_data_path = "expert_data/expert_dataset_{}.npz".format(env_id)

    # normalization callback(try to use limited range if mdp
    # observation limits are available)
    try:
        normalizer = MinMaxPreprocessor(mdp_info=mdp.info)
    except:
        normalizer = StandardizationPreprocessor(mdp_info=mdp.info)

    # plotting callback
    plotter = PlotDataset(mdp.info, obs_normalized=True)

    expert_data = prepare_expert_data(
            data_path=expert_data_path,
            n_samples=horizon*n_expert_trajectories,
            data_normalizer=None,
    )
    #warmup_running_norm(normalizer=normalizer, expert_data=expert_data)

    if algorithm == "GAIL":
        agent = _create_gail_agent(mdp, expert_data, discr_only_state)
    elif algorithm == "VAIL":
        # train new agent with gail, following expert trajectories
        agent = _create_vail_agent(mdp, expert_data, discr_only_state)
    else:
        raise NotImplementedError

    # evaluate untrained policy
    # Algorithm(with normalizing and plotting)
    core = Core(agent, mdp, callback_step=plotter, preprocessors=[normalizer])
    dataset = core.evaluate(n_episodes=20)
    J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
    R_mean = np.mean(compute_J(dataset))
    print('Before {} -> R: {},  J: {},  Entropy: {}'.format(algorithm, R_mean, J_mean, agent.policy.entropy()))

    if init_bc:
        init_policy_with_bc(agent, expert_data)
        # evaluate and show untrained policy
        core.evaluate(n_episodes=2, render=True)
        dataset = core.evaluate(n_episodes=20)
        J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
        R_mean = np.mean(compute_J(dataset))
        print('After BC -> R: {},  J: {},  Entropy: {}'.format(R_mean, J_mean, agent.policy.entropy()))


    epoch_js = []
    # gail train loop
    for it in range(50):
        core.learn(n_steps=10000, n_steps_per_fit=1024)
        dataset = core.evaluate(n_episodes=10)
        J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
        R_mean = np.mean(compute_J(dataset))
        print('Epoch: {}  ->  R: {},  J: {},  Entropy: {}'.format(it, R_mean, J_mean, agent.policy.entropy()))
        epoch_js.append(J_mean)

    print("--- The train has finished ---")
    import matplotlib.pyplot as plt
    plt.plot(epoch_js)
    plt.show()
    plt.title("Train J -> env_id: {},  algorithm: {}".format(env_id, algorithm))
    input("Press any key to visualize trained policy")

    # evaluate and show trained policy
    core.evaluate(n_episodes=2, render=True)
    dataset = core.evaluate(n_episodes=20)
    J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
    R_mean = np.mean(compute_J(dataset))
    print('After {} ->  R: {},  J: {},  Entropy: {}'.format(algorithm, R_mean, J_mean, agent.policy.entropy()))


if __name__ == "__main__":
    # algorithm to use(only GAIL and VAIL available)
    algorithm = ["GAIL", "VAIL"][0]

    # gym environment to use(env_id can be any gym env, as long
    # as demonstrations are available for it)
    # check out extract_expert_trajectories.py to see how to extract
    # expert trajectories.
    env_id = 'Hopper-v2'
    env_kwargs = dict(name=env_id, horizon=1000, gamma=0.99)
    n_expert_trajectories = 25

    experiment(algorithm=algorithm, env_kwargs=env_kwargs,
               n_expert_trajectories=n_expert_trajectories,
               init_bc=True, discr_only_state=False)