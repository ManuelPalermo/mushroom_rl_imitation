import itertools

import numpy as np


class SCIRL:
    def __init__(self, action_space, feat_expect, theta_size, debug_mode=False, init_theta=None):

        self.action_space = action_space
        self.feat_expect = feat_expect

        if init_theta is None:
            self.theta = np.zeros((theta_size,))
        else:
            self.theta = init_theta

        self.best_loss = np.inf
        self.best_theta = self.theta

        self.debug_mode = debug_mode

        self.epoch = 0
        self.found = False

    def _run(self, iterations, expert_states, expert_actions, lr, mini_batch_size=2, stop_threshold=0.0001,
             evaluate=False):

        if mini_batch_size > expert_states.shape[0]:
            print('MiniBatchSize capped at {}.'.format(expert_states.shape[0]))
            mini_batch_size = expert_states.shape[0]

        for i in range(iterations):
            indexes_mini_batch = np.random.choice(expert_states.shape[0], mini_batch_size)
            gradient, loss_metrics = self._gradient(expert_states[indexes_mini_batch, :],
                                                    expert_actions[indexes_mini_batch, :])
            gradient = gradient.squeeze()

            if evaluate:
                print('*Evaluation*: Loss mean: {}, Loss std: {}'.format(*loss_metrics))

            if not evaluate:
                print('Training: Epoch {} - Loss mean: {}, Loss std: {}'.format(self.epoch, *loss_metrics))
                gradient_norm_2 = np.linalg.norm(gradient, ord=2) + 1e-12
                self.theta -= gradient / gradient_norm_2 * lr(i)

                if loss_metrics[0] < self.best_loss:
                    self.best_loss = loss_metrics[0]
                    self.best_theta = self.theta.copy()

                if loss_metrics[0] < stop_threshold:
                    self.found = True
                    break

                self.epoch += 1

        return self.best_theta, self.found

    def _gradient(self, expert_states, expert_actions):

        state_action_expert = np.append(expert_states, expert_actions, axis=1)
        state_action_expert = np.append(np.array([np.arange(expert_states.shape[0])]).T, state_action_expert,
                                        axis=1)

        state_action_combs = np.array(
            list(map(list, list(itertools.product(state_action_expert.tolist(), self.action_space.tolist())))))
        state_action_expert, action_analysed = np.array(list(map(np.array, state_action_combs[:, 0]))), \
                                               np.array(list(map(np.array, state_action_combs[:, 1])))

        index1 = state_action_expert[:, 0]
        index2 = np.copy(index1)
        index2[1:] = index2[:-1]
        change_index = np.argwhere(index1 - index2).squeeze()

        state_expert = state_action_expert[:, 1:-expert_actions.shape[1]]
        action_expert = state_action_expert[:, -expert_actions.shape[1]:]

        analyzed_feat_expect = self.feat_expect(state_expert, action_analysed)
        expert_feat_expect = self.feat_expect(state_expert, action_expert)
        max_vals = np.dot(analyzed_feat_expect, self.theta) + (action_analysed != action_expert).any(axis=1)

        max_vals_splitted = np.split(max_vals, change_index)

        max_indexes = np.empty((0,))
        for max_val_per_state in max_vals_splitted:
            max_index = np.argmax(max_val_per_state)
            if max_index.size != 1:
                max_index = max_index[np.random.choice(max_index.size, 1)]
            max_indexes = np.append(max_indexes, max_index)

        restructured_indexes = np.copy(max_indexes)
        restructured_indexes[1:] = max_indexes[1:] + change_index
        restructured_indexes = restructured_indexes.astype('int')

        gradient = analyzed_feat_expect[restructured_indexes, :] - expert_feat_expect[restructured_indexes, :]
        gradient = np.mean(gradient, axis=0)

        loss = max_vals[restructured_indexes] - np.dot(expert_feat_expect[restructured_indexes, :], self.theta)

        print('Same action (%): ', np.count_nonzero(
            np.equal(action_analysed[restructured_indexes], action_expert[restructured_indexes]).all(axis=1))
              / action_analysed[restructured_indexes].shape[0] * 100)

        return gradient, (np.mean(loss), np.std(loss))

    def feature_expectation(self, state, action):
        return self.feat_expect(state, action)

    def train(self, iterations, expert_states, expert_actions, lr, mini_batch_size=2, stop_threshold=0.0001):
        return self._run(iterations, expert_states, expert_actions, lr, mini_batch_size, stop_threshold,
                         evaluate=False)

    def evaluate(self, expert_states, expert_actions):
        self._run(1, expert_states, expert_actions, 0, expert_states.shape[0], 0,
                  evaluate=True)
