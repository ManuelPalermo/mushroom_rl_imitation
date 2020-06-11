

from copy import deepcopy
from joblib import Parallel, delayed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mushroom_rl.utils.callbacks import PlotDataset
from tqdm import tqdm, trange

from mushroom_rl.utils.preprocessors import MinMaxPreprocessor

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.environments import Gym
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl_imitation.imitation.vail import VAIL
from mushroom_rl_imitation.imitation.gail import GAIL


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


def _create_vail_agent(mdp, expert_data, disc_only_state, **kwargs):
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
    clip_eps_ppo = .2
    gae_lambda = .95
    policy_std_0 = 0.3

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


def _create_gail_agent(mdp, expert_data, disc_only_state, **kwargs):
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
    batch_size_policy = 128                # 64
    clip_eps_ppo = .2
    gae_lambda = .95
    policy_std_0 = 0.3

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
                         n_epochs=50, patience=10)


def experiment(exp_id, env_id, agent_id, disc_only_state,
               horizon=500, n_expert_traj=10, n_trials=3, n_epochs=250,
               behaviour_clone_start=True, use_plot=False, expert_J=np.nan,
               **kwargs):

    experience_js = []
    # add expert J(not available)
    experience_js.append([expert_J for _ in range(n_epochs)])
    print("\n------------------------ EXPERIENCE {} ---------------------------".format(exp_id))

    mdp = Gym(name=env_id, horizon=horizon, gamma=0.99)
    preprocessor = MinMaxPreprocessor(mdp_info=mdp.info)

    # prepare expert samples
    expert_data = prepare_expert_data(
            data_path="expert_data/expert_dataset_{}.npz".format(env_id),
            n_samples=horizon*n_expert_traj, normalizer=None,
    )

    # create dummy policy to train with behaviour cloning as baseline
    agent_bc = _create_gail_agent(mdp, expert_data=expert_data,
                                  disc_only_state=disc_only_state)

    # behaviour cloning baseline
    init_policy_with_bc(agent_bc, expert_data=expert_data)
    bc_core = Core(agent_bc, mdp, preprocessors=[preprocessor])

    # evaluate bc trained policy
    dataset_bc = bc_core.evaluate(n_episodes=50)
    R_mean_bc = np.mean(compute_J(dataset_bc))
    J_mean_bc = np.mean(compute_J(dataset_bc, mdp.info.gamma))
    experience_js.append([J_mean_bc for _ in range(n_epochs)])
    tqdm.write('BClon results  ->  J: {},  R: {},  Entropy: {}'
               .format(J_mean_bc, R_mean_bc, bc_core.agent.policy.entropy()))


    for trial_i in range(n_trials):
        if agent_id == "gail":
            agent = _create_gail_agent(mdp, expert_data=expert_data, disc_only_state=disc_only_state)
        elif agent_id == "vail":
            agent = _create_vail_agent(mdp, expert_data=expert_data, disc_only_state=disc_only_state)
        else:
            raise NotImplementedError

        if behaviour_clone_start:
            # train policy mu network through behaviour cloning
            init_policy_with_bc(agent_bc, expert_data=expert_data)
            dataset = bc_core.evaluate(n_episodes=10)
            J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
            print('After BC -> J: {}, Entropy: {}'.format(J_mean, agent.policy.entropy()))

        # gail train loop
        if use_plot:
            plotter = PlotDataset(mdp.info, obs_normalized=True)
            core = Core(agent_bc, mdp, callback_step=plotter, preprocessors=[preprocessor])
        else:
            core = Core(agent_bc, mdp, preprocessors=[preprocessor])

        epoch_js = []
        for it in range(n_epochs):
            core.learn(n_steps=10241, n_steps_per_fit=1024, render=False, quiet=True)
            dataset = core.evaluate(n_episodes=20, render=False)
            J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
            epoch_js.append(J_mean)
        experience_js.append(epoch_js)

        # evaluate trained policy
        dataset = core.evaluate(n_episodes=50, quiet=True)
        J_mean = np.mean(compute_J(dataset, mdp.info.gamma))
        R_mean = np.mean(compute_J(dataset))
        tqdm.write('Agent {} results  ->  J: {},  R: {},  Entropy: {}'
                   .format(trial_i, J_mean, R_mean, core.agent.policy.entropy()))
        del agent, core

    # plot runs
    experience_js = np.array(experience_js).T
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(experience_js[:, 0], c="r")                  # expert performance
    plt.plot(experience_js[:, 1], c="k")                  # behaviour cloning performance
    plt.plot(experience_js[:, 2:].mean(axis=1), c="b")    # runs average performance
    plt.plot(experience_js[:, 2:], alpha=0.33)            # runs performance

    plt.title("Algorithm: {}, env_id: {}".format(agent_id, env_id))
    plt.legend(["Expert", "Behav_clone", "Run_mean", *["Run_{}".format(i) for i in range(n_trials)]])

    save_path = "./output/algorithm_{}_env_id_{}".format(agent_id, env_id) \
                + ("_state_only_" if disc_only_state else "") \
                + ("_bc_start" if behaviour_clone_start else "") \
                + ".png"
    plt.savefig(save_path)


if __name__ == "__main__":
    run_experiments_parallel = True
    n_trials = 5
    n_epochs_trial = 200

    tests = [dict(traj_id=1, agent_id="vail", disc_only_state=False, behaviour_clone_start=True),
             dict(traj_id=1, agent_id="gail", disc_only_state=False, behaviour_clone_start=True),

             dict(traj_id=1, agent_id="vail", disc_only_state=True, behaviour_clone_start=True),
             dict(traj_id=1, agent_id="gail", disc_only_state=True, behaviour_clone_start=True),

             dict(traj_id=1, agent_id="vail", disc_only_state=False, behaviour_clone_start=False),
             dict(traj_id=1, agent_id="gail", disc_only_state=False, behaviour_clone_start=False),

             dict(traj_id=0, agent_id="vail", disc_only_state=False, behaviour_clone_start=True),
             dict(traj_id=0, agent_id="gail", disc_only_state=False, behaviour_clone_start=True),
            ]

    if run_experiments_parallel:
        Js = Parallel(n_jobs=-1)(delayed(experiment)(exp_id=exp_i, **params, ntrials=n_trials,
                                                     n_epochs=n_epochs_trial, use_plot=False)
                                 for exp_i, params in enumerate(tests))

    else:
        for exp_i, params in enumerate(tests):
            experiment(exp_id=exp_i, **params, n_trials=n_trials, n_epochs=n_epochs_trial, use_plot=False)
