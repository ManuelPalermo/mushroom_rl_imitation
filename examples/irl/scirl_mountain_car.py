import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
from mushroom_rl.core import Core
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl_imitation.Utils.kernels import RBF
from mushroom_rl_imitation.Utils.numpy_extended import find_nearest
from mushroom_rl_imitation.irl.feat_expect import MonteCarlo
from mushroom_rl_imitation.irl.scirl import SCIRL

STATE_RESOLUTION = 15
ACTION_RESOLUTION = 3

DIGITIZE_STATE_VALS = np.linspace(-1.2, 0.6, STATE_RESOLUTION)
DIGITIZE_STATE_VALS = np.vstack([DIGITIZE_STATE_VALS, np.linspace(-0.07, 0.07, STATE_RESOLUTION)])

DIGITIZE_ACTION_VALS = np.array([np.linspace(0., 2., ACTION_RESOLUTION)])

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


def experiment():
    """Test of SCIRL on the Mountain Car environment."""
    # Settings
    # MDP Settings
    gamma = 0.9999
    horizon = 200

    # Training Agent General Settings
    epsilon_greedy_val = 0.01
    n_steps = 2000
    n_steps_test = 1000
    n_epochs_training_expert = 10
    n_epochs_training_new_agent = 20

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

    RESOURCES_DIR = os.path.dirname(os.path.realpath(__file__)) + '/traj_mountain_car'
    os.makedirs(RESOURCES_DIR, exist_ok=True)

    # Environment
    mdp = Gym(name='MountainCar-v0', horizon=horizon, gamma=gamma)

    # Policy
    epsilon = Parameter(value=epsilon_greedy_val)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    n_tilings = 7
    tilings = Tiles.generate(n_tilings, [7, 7],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high)
    features = Features(tilings=tilings)

    learning_rate = Parameter(.1 / n_tilings)

    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n)
    algorithm_params = {'learning_rate': learning_rate,
                        'lambda_coeff': .9}

    agent = TrueOnlineSARSALambda(mdp.info, pi,
                                  approximator_params=approximator_params,
                                  features=features, **algorithm_params)

    # Slowing rendering for better visualization
    mdp.render = render_slow(mdp.render, 0.005)

    if execute_irl:
        if train_expert:
            print('Trainin Expert')

            # Algorithm
            core = Core(agent, mdp)

            print('Acquiring dataset non-expert')
            core.evaluate(n_steps=600, render=True)
            dataset_non_expert = core.evaluate(n_steps=dataset_n, render=False)

            print('Training Expert')
            for n in range(n_epochs_training_expert):
                print('Epoch: ', n)
                core.learn(n_steps=n_steps, n_steps_per_fit=1)
                dataset = core.evaluate(n_steps=n_steps_test, render=False)
                J = compute_J(dataset, gamma)
                print('J: ', np.mean(J))

            print('Acquiring dataset expert')
            core.evaluate(n_steps=600, render=True)
            dataset_expert = core.evaluate(n_steps=dataset_n, render=False)

            states, actions, reward, next_state, absorbing, last = parse_dataset(dataset_expert)

            absorbing = np.logical_or(absorbing, last)  # to separate episodes

            with open(RESOURCES_DIR + '/state_action_absorbing.pickle', 'wb') as traj_file:
                pickle.dump((states, actions, absorbing, next_state), traj_file)

        else:
            with open(RESOURCES_DIR + '/state_action_absorbing.pickle', 'rb') as traj_file:
                states, actions, absorbing, next_state = pickle.load(traj_file)

        # Discretization of MDP for simplification of the problem
        states, actions = discretize(states, actions)

        # Separation of data in training and testing datasets
        ticks = np.argwhere(absorbing).squeeze()
        split_tick = (absorbing.size - 1) * (1 - validation_fraction)
        tick_index_chosen = find_nearest(split_tick, ticks)
        split_index = np.array([ticks[tick_index_chosen]]) + 1
        states_train, states_test = tuple(np.split(states, split_index))
        actions_train, actions_test = tuple(np.split(actions, split_index))

        # Building feature expectation with Monte Carlo methods

        print('Monte Carlo rollouts')
        feat_expect = MonteCarlo(
            state_to_psi=psi,
            action_space=ACTION_GRID.T,
            gamma=gamma)
        feat_expect.fit(states, actions, absorbing)
        seaborn.heatmap(feat_expect.feat_expect_val)  # Feature expectation results
        plt.pause(1)

        print('SCIRL Running')
        irl_agent = SCIRL(DIGITIZE_ACTION_VALS.T,
                          feat_expect,
                          feat_expect.feat_expect_val[0].size,
                          init_theta=initial_reward_weights
                          # init_theta=np.ones_like(psi(states[0, :], actions[0, :])) * 2,
                          )

        theta = irl_agent.theta
        for i in range(max_epochs_scirl):
            theta, found = irl_agent.train(it_per_epoch_scirl, states_train, actions_train,
                                           lr=exp_decay_lr(decay_constant_lr),
                                           mini_batch_size=batch_size_scirl,
                                           stop_threshold=stop_threshold,
                                           )
            #  If the values for training and testing are not the same overfitting is occuring
            irl_agent.evaluate(states_test, actions_test)

            if found:
                # if irl_agent found a theta with a loss smaller than the stop threshold
                break

        with open(RESOURCES_DIR + '/theta_vector.pickle', 'wb') as save_theta:
            pickle.dump(theta, save_theta)

    else:
        if initial_reward_weights is None:
            # Load previous weights saved
            with open(RESOURCES_DIR + '/theta_vector.pickle', 'rb') as save_theta:
                theta = pickle.load(save_theta)
                plt.plot(theta)
                plt.pause(1)
        else:
            # Use initial_reward_weights to train new agent
            theta = initial_reward_weights

    if 'dataset_non_expert' in locals() and 'dataset_expert' in locals():
        """
        Showing difference of reward produced by the new reward function
            between the expert trajectories and non-expert trajectories.
        The graphs presented, represent the state value function along the
            episode. The values of the expert should be superior that the 
            non-expert.
        """
        dataset_parsed_non_expert = parse_dataset(dataset_non_expert)
        dataset_parsed_expert = parse_dataset(dataset_expert)

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

        plt.figure()
        plt.plot(non_expert_array_J[0:200],
                 color='b')
        plt.plot(expert_array_j[0:200],
                 color='g')
        plt.plot(np.ma.masked_where(np.logical_or(dataset_parsed_expert[4], dataset_parsed_expert[5]) is not True,
                                    expert_array_j)[0:200],
                 marker='o', linestyle='None', color='r')
        plt.legend(['Non_expert', 'Expert', 'Absorbing'])
        plt.pause(1)

    # New reward function implemented
    mdp.step = new_reward_step_wrapper(mdp.step, theta)

    # Creating new agent for training
    agent = TrueOnlineSARSALambda(mdp.info, pi,
                                  approximator_params=approximator_params,
                                  features=features, **algorithm_params)

    core = Core(agent, mdp)
    for n in range(n_epochs_training_new_agent):
        print('Epoch: ', n)
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        J = compute_J(dataset, gamma)
        print('J: ', np.mean(J))

    input('Done')
    core.evaluate(n_steps=n_steps_test, render=True)


def discretize(state, action):
    if len(state.shape) == 1:
        state = np.array([state])
        action = np.array([action])

    position = DIGITIZE_STATE_VALS[0, find_nearest(state[:, 0], DIGITIZE_STATE_VALS[0, :])]
    velocity = DIGITIZE_STATE_VALS[1, find_nearest(state[:, 1], DIGITIZE_STATE_VALS[1, :])]
    state = np.array([position, velocity]).T

    action = DIGITIZE_ACTION_VALS[0, find_nearest(action, DIGITIZE_ACTION_VALS.squeeze())]
    action = np.array([action]).T

    return state, action


rbf_kernel = RBF(STATE_GRID.T, 10)


def psi(state, action):
    if len(state.shape) == 1:
        state = np.array([state])

    output_array = rbf_kernel(state)
    return output_array.squeeze()


def render_slow(old_render, time_ammount):
    def new_render(*args, **kwargs):
        time.sleep(time_ammount)
        return old_render(*args, **kwargs)

    return new_render


def new_reward_step_wrapper(old_step, weigths_array):
    def new_step(action):
        old_step_return = list(old_step(action))
        state = old_step_return[0]
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
    experiment()  # Main Script
