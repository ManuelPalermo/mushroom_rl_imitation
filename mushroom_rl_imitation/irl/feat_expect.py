import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip


class FeatExpect:
    def __call__(self, state, action):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError


class MonteCarlo(FeatExpect):
    def __init__(self, state_to_psi, action_space, gamma, quiet=True):

        self.state_to_psi = state_to_psi
        self.action_space = action_space
        self.gamma = gamma
        self.quiet = quiet

        self.feat_expect_key = None
        self.feat_expect_val = None
        self.n_estimates = None

    def __call__(self, state, action):
        results = list()
        for single_state, single_action in zip(state, action):
            match_index = self.get_key(single_state, single_action)
            results.append(self.feat_expect_val[match_index.squeeze()])
        return np.vstack(results)

    def fit(self, expert_states, expert_actions, absorbing_states):
        """
        Fits a batch of data for calculating the feature expectation

        Args:
            expert_states (np.ndarray):  matrix of states with shape (N, M) -
                N: number os steps, M: dimensionality of the state space;
            expert_actions (np.ndarray): matrix of actions with shape (N, M) -
                N: number os steps, M: dimensionality of the actions space;
            absorbing_states (np.ndarray):  matrix of absorbing with shape (N,) -
                N: number os steps
        """
        assert len(expert_states.shape) == len(expert_actions.shape) == 2
        assert len(absorbing_states.shape) == 1

        if self.feat_expect_key is None:
            self.feat_expect_key = np.empty(
                (0, np.concatenate([expert_states[0, :], expert_actions[0, :]]).size))
            self.feat_expect_val = np.empty(
                (0, self.state_to_psi(expert_states[0, :], expert_actions[0, :]).size))
            self.n_estimates = np.empty((0,))

        absorbing_states_indexes = np.argwhere(absorbing_states).squeeze() + 1

        for episodic_states, episodic_actions, episodic_absorbing in tzip(
                np.split(expert_states, absorbing_states_indexes),
                np.split(expert_actions, absorbing_states_indexes),
                np.split(absorbing_states, absorbing_states_indexes),
                position=0):
            if episodic_states.size != 0:
                self._calculate_feat_expect_episode(episodic_states, episodic_actions)

    def get_key(self, state, action):
        """
        Gets index of the inputs state action feature vector

        Args:
            psi_vector (np.ndarray): Array of shape (M,) with M being the dimensionality of the psi space

        Returns:
            Integer of the position of the vector in the feat_expect_key
        """
        match_index = np.argwhere((np.array([np.append(state, action)]) == self.feat_expect_key).all(axis=1)).squeeze()
        return match_index

    def _add_feat_expect(self, state, action, feat_expect_val):
        match_index = self.get_key(state, action)
        if match_index.size == 1:
            self.feat_expect_val[match_index, :] = np.add(self.feat_expect_val[match_index, :] *
                                                          self.n_estimates[match_index], feat_expect_val) \
                                                   / (self.n_estimates[match_index] + 1)
            self.n_estimates[match_index] += 1
        else:
            self.feat_expect_key = np.append(self.feat_expect_key, np.array([np.append(state, action)]), axis=0)
            self.feat_expect_val = np.append(self.feat_expect_val, np.zeros_like(np.array([feat_expect_val])), axis=0)
            self.n_estimates = np.append(self.n_estimates, 0)
            self._add_feat_expect(state, action, feat_expect_val)

    def _calculate_feat_expect_episode(self, states_trajectories, actions_trajectories):
        """
        Args:
            Calculates the rollout of one episode.

            states_trajectories (np.ndarray):  matrix of states with shape (N, M) -
                N: number os steps, M: dimensionality of the state space;
            actions_trajectories (np.ndarray): matrix of actions with shape (N, M) -
                N: number os steps, M: dimensionality of the actions space;

        """

        feat_expect = np.zeros_like(self.state_to_psi(states_trajectories[0], actions_trajectories[0]))
        for i in tqdm(range(states_trajectories.shape[0] - 1, -1, -1), position=1):
            step_psi = self.state_to_psi(states_trajectories[i], actions_trajectories[i])
            feat_expect = np.add(self.gamma * feat_expect, step_psi)
            self._add_feat_expect(states_trajectories[i], actions_trajectories[i], feat_expect)

            for action in self.action_space:
                self._add_feat_expect(states_trajectories[i], action,
                                      feat_expect * self.gamma)
