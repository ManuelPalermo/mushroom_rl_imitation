import os
import pickle
import time
from collections import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl_imitation.utils.kernels import RBF
from mushroom_rl_imitation.utils.numpy_extended import find_nearest
from mushroom_rl_imitation.irl.feat_expect import MonteCarlo
from mushroom_rl_imitation.irl.scirl import SCIRL

STATE_RESOLUTION = 15
ACTION_RESOLUTION = 10

DIGITIZE_STATE_VALS = np.linspace(-np.pi, np.pi, STATE_RESOLUTION)
DIGITIZE_STATE_VALS = np.vstack([DIGITIZE_STATE_VALS, np.linspace(-8., 8., STATE_RESOLUTION)])

DIGITIZE_ACTION_VALS = np.array([np.linspace(-2., 2., ACTION_RESOLUTION)])

ACTION_GRID = np.meshgrid(*list(DIGITIZE_ACTION_VALS))
STATE_GRID = np.meshgrid(*list(DIGITIZE_STATE_VALS))

STATE_ACTION_GRID = np.meshgrid(STATE_GRID, ACTION_GRID)

for v in range(len(ACTION_GRID)):
    ACTION_GRID[v] = ACTION_GRID[v].ravel()
ACTION_GRID = np.vstack(ACTION_GRID)

for v in range(len(STATE_GRID)):
    STATE_GRID[v] = STATE_GRID[v].ravel()
STATE_GRID = np.vstack(STATE_GRID)

FEAT_VECTOR_SIZE = (STATE_GRID.shape[1] + 1) * ACTION_GRID.shape[1]


class CriticNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

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

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment():
    """Test of SCIRL on the Mountain Car environment."""
    # Settings
    # MDP Settings
    horizon = 200
    gamma = 0.99

    # Training Agent General Settings
    n_steps = 2000
    n_steps_test = 1000
    n_epochs_training_expert = 10
    n_epochs_training_new_agent = 10
    initial_replay_size = 64
    max_replay_size = 50000
    batch_size = 64
    n_features = 64
    warmup_transitions = 100
    tau = 0.005
    lr_alpha = 3e-4
    use_cuda = False
    alg = SAC

    # Expert Dataset Settings
    train_expert = True
    dataset_n = 200 * 50
    validation_fraction = 0.2

    # SCIRL Settings
    execute_irl = True
    initial_reward_weights = None
    max_epochs_scirl = 5
    it_per_epoch_scirl = 10
    batch_size_scirl = 2000
    stop_threshold = 0.01
    decay_constant_lr = 0.9

    RESOURCES_DIR = os.path.dirname(os.path.realpath(__file__)) + '/pendulum_data'
    os.makedirs(RESOURCES_DIR, exist_ok=True)

    # MDP
    mdp = Gym('Pendulum-v0', horizon, gamma)

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

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    # Agent
    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None)

    old_render = mdp.render

    if execute_irl:
        if train_expert:
            print('Trainin Expert')

            # Algorithm
            core = Core(agent, mdp)

            print('Acquiring dataset non-expert')
            mdp.render = render_slow(mdp.render, 0.03)
            core.evaluate(n_steps=600, render=True)
            mdp.render = old_render
            dataset_non_expert = core.evaluate(n_steps=dataset_n, render=False)

            print('Training')
            for n in range(n_epochs_training_expert):
                print('Epoch: ', n)
                core.learn(n_steps=n_steps, n_steps_per_fit=1)
                dataset = core.evaluate(n_steps=n_steps_test, render=False)
                J = compute_J(dataset, gamma)
                print('J: ', np.mean(J))

            print('Acquiring dataset expert')
            mdp.render = render_slow(mdp.render, 0.03)
            core.evaluate(n_steps=600, render=True)
            mdp.render = old_render
            dataset_expert = core.evaluate(n_steps=dataset_n, render=False)

            states, actions, reward, next_state, absorbing, last = parse_dataset(dataset_expert)

            absorbing = np.logical_or(absorbing, last)

            with open(RESOURCES_DIR + '/state_action_absorbing.pickle', 'wb') as traj_file:
                pickle.dump((states, actions, absorbing, next_state), traj_file)

        else:
            with open(RESOURCES_DIR + '/state_action_absorbing.pickle', 'rb') as traj_file:
                states, actions, absorbing, next_state = pickle.load(traj_file)

        states, actions = project_state_action(states, actions)
        states, actions = discretize(states, actions)

        ticks = np.argwhere(absorbing).squeeze()
        split_tick = (absorbing.size - 1) * (1 - validation_fraction)
        tick_index_chosen = find_nearest(split_tick, ticks)
        split_index = np.array([ticks[tick_index_chosen]]) + 1

        states_train, states_test = tuple(np.split(states, split_index))
        actions_train, actions_test = tuple(np.split(actions, split_index))

        feat_expect = MonteCarlo(
            state_to_psi=psi,
            action_space=ACTION_GRID.T,
            gamma=gamma)

        print('Monte Carlo rollouts')
        feat_expect.fit(states, actions, absorbing)

        seaborn.heatmap(feat_expect.feat_expect_val)
        plt.pause(1)

        print('SCIRL Running')
        irl_agent = SCIRL(DIGITIZE_ACTION_VALS.T,
                          feat_expect,
                          feat_expect.feat_expect_val[0].size,
                          init_theta=initial_reward_weights,
                          # init_theta=np.ones_like(psi(states[0,:], actions[0,:]))*2,
                          )

        theta = irl_agent.theta

        for i in range(max_epochs_scirl):
            theta, found = irl_agent.train(it_per_epoch_scirl, states_train, actions_train,
                                           lr=exp_decay_lr(decay_constant_lr),
                                           mini_batch_size=batch_size_scirl,
                                           stop_threshold=stop_threshold,
                                           )

            irl_agent.evaluate(states_test, actions_test)
            if found:
                break

        with open(RESOURCES_DIR + '/theta_vector.pickle', 'wb') as save_theta:
            pickle.dump(theta, save_theta)

    else:

        if initial_reward_weights is None:
            with open(RESOURCES_DIR + '/theta_vector.pickle', 'rb') as save_theta:
                theta = pickle.load(save_theta)
                plt.plot(theta)
                plt.pause(1)
        else:
            theta = initial_reward_weights

    if 'dataset_non_expert' in locals() and 'dataset_expert' in locals():
        dataset_parsed_non_expert = list(parse_dataset(dataset_non_expert))
        dataset_parsed_expert = list(parse_dataset(dataset_expert))

        dataset_parsed_non_expert[0], dataset_parsed_non_expert[1] = \
            project_state_action(dataset_parsed_non_expert[0], dataset_parsed_non_expert[1])
        dataset_parsed_expert[0], dataset_parsed_expert[1] = \
            project_state_action(dataset_parsed_expert[0], dataset_parsed_expert[1])

        dataset_parsed_non_expert[0], dataset_parsed_non_expert[1] = \
            discretize(dataset_parsed_non_expert[0], dataset_parsed_non_expert[1])
        dataset_parsed_expert[0], dataset_parsed_expert[1] = \
            discretize(dataset_parsed_expert[0], dataset_parsed_expert[1])

        psi_non_expert = psi(dataset_parsed_non_expert[0], dataset_parsed_non_expert[1])
        psi_expert = psi(dataset_parsed_expert[0], dataset_parsed_expert[1])

        reward_non_expert = np.dot(psi_non_expert, theta)
        reward_expert = np.dot(psi_expert, theta)

        non_expert_array_J = compute_v_array(reward_non_expert,
                                             np.logical_or(dataset_parsed_non_expert[4], dataset_parsed_non_expert[5]),
                                             gamma)
        expert_array_j = compute_v_array(reward_expert,
                                         np.logical_or(dataset_parsed_expert[4], dataset_parsed_expert[5]),
                                         gamma)

        with open(RESOURCES_DIR + '/plotting_data.pickle', 'wb') as save_plot_file:
            save_dict = dict(
                non_expert_data=dict(
                    state = dataset_parsed_non_expert[0],
                    action = dataset_parsed_non_expert[1],
                    original_reward = dataset_parsed_non_expert[2],
                    new_reward = reward_non_expert,
                    absorbing = dataset_parsed_non_expert[4],
                    last = dataset_parsed_non_expert[5],
                ),
                expert_data=dict(
                    state=dataset_parsed_expert[0],
                    action=dataset_parsed_expert[1],
                    original_reward=dataset_parsed_expert[2],
                    new_reward=reward_expert,
                    absorbing=dataset_parsed_expert[4],
                    last=dataset_parsed_expert[5],
                )
            )
            pickle.dump(save_dict, save_plot_file)

        plt.figure()
        plt.plot(non_expert_array_J[0:500],
                 color='b')
        plt.plot(expert_array_j[0:500],
                 color='g')
        plt.plot(np.ma.masked_where(np.logical_or(dataset_parsed_expert[4], dataset_parsed_expert[5]) is not True,
                                    expert_array_j)[0:500],
                 marker='o', linestyle='None', color='r')
        plt.legend(['Non_expert', 'Expert', 'Absorbing'])
        plt.pause(1)

    mdp.step = step_wrapper(mdp.step, theta)

    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None)

    core = Core(agent, mdp)

    for n in range(n_epochs_training_new_agent):
        print('Epoch: ', n)
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        J = compute_J(dataset, gamma)
        print('J: ', np.mean(J))

    input('Done')
    mdp.render = render_slow(mdp.render, 0.03)
    core.evaluate(n_steps=600, render=True)
    mdp.render = old_render


def trig_to_angle(cosine_val, sine_val):
    results = []
    if not isinstance(cosine_val, Iterable):
        cosine_val = [cosine_val]
    if not isinstance(sine_val, Iterable):
        sine_val = [sine_val]

    for c, s in zip(cosine_val, sine_val):
        results.append(np.angle([complex(c, s)]))
    return np.array(results)


def project_state_action(state, action):
    state = np.concatenate([trig_to_angle(state[:, 0], state[:, 1]), np.array([state[:, 2]]).T], axis=1)
    return state, action


DISC = np.linspace(-np.pi, np.pi, 500)
DISC = np.vstack([DISC, np.linspace(-8., 8., 500)])


def discretize(state, action):
    for i in range(DISC.shape[0]):
        state[:, i] = DISC[i, find_nearest(state[:, i], DISC[i, :]) - 1]
    return state, action


rbf_kernel = RBF(STATE_GRID.T, 10)


def psi(state, action):
    if len(state.shape) == 1:
        state = np.array([state])
        action = np.array([action])

    output_array = rbf_kernel(state)
    return output_array.squeeze()


def render_slow(old_render, time_ammount):
    def new_render(*args, **kwargs):
        time.sleep(time_ammount)
        return old_render(*args, **kwargs)

    return new_render


def step_wrapper(old_step, weigths_array):
    def new_step(action):
        old_step_return = list(old_step(action))
        state = old_step_return[0]
        state, action = project_state_action(np.array([state]), np.array([action]))
        state, action = discretize(state, action)
        psi_array = psi(state, action)
        old_step_return[1] = np.dot(psi_array, weigths_array)
        return tuple(old_step_return)

    return new_step


def compute_v_array(reward_array, absorbing_array, gamma):
    v_array = []
    temp_v = 0
    for reward, absorbing in zip(reversed(reward_array), reversed(absorbing_array)):
        if absorbing:
            temp_v = 0
        v_array.append(temp_v * gamma + reward)
    v_array.reverse()
    return np.array(v_array)


def exp_decay_lr(constant):
    def lr(iteration):
        return constant ** iteration

    return lr


if __name__ == '__main__':
    experiment()
