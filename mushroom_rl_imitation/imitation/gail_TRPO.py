from copy import deepcopy

import torch
import numpy as np

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import get_gradient, zero_grad, to_float_tensor
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.value_functions import compute_gae
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.trpo import TRPO
from mushroom_rl.utils.minibatches import minibatch_generator


class GAIL(TRPO):
    """
    Generative Adversarial Imitation Learning(GAIL) implementation.

    "Generative Adversarial Imitation Learning"
    Ho, J., & Ermon, S. (2016).

    """

    def __init__(self, mdp_info, policy_class, policy_params,
                 discriminator_params, critic_params,
                 n_epochs_discriminator=1,
                 ent_coeff=0., max_kl=.001, lam=1.,
                 n_epochs_line_search=10, n_epochs_cg=10,
                 cg_damping=1e-2, cg_residual_tol=1e-10,
                 demonstrations=None, env_reward_frac=0.0,
                 state_mask=None, act_mask=None, quiet=True,
                 critic_fit_params=None, discriminator_fit_params=None):

        # initialize PPO agent
        policy = policy_class(**policy_params)
        super(GAIL, self).__init__(mdp_info, policy, critic_params,
                                   ent_coeff, max_kl, lam, n_epochs_line_search,
                                   n_epochs_cg, cg_damping, cg_residual_tol,
                                   quiet=quiet, critic_fit_params=critic_fit_params)

        # discriminator params
        self._discriminator_fit_params = (dict() if discriminator_fit_params is None
                                          else discriminator_fit_params)

        discriminator_params.setdefault("loss", torch.nn.BCELoss())
        discriminator_params.setdefault("batch_size", 128)
        self._D = Regressor(TorchApproximator, **discriminator_params)
        self._n_epochs_discriminator = n_epochs_discriminator

        self._env_reward_frac = env_reward_frac
        self._demonstrations = demonstrations   # should be: dict(states=np.array, actions=(np.array/None))
        assert 0.0 <= env_reward_frac <= 1.0, "Environment reward must be between [0,1]"
        assert demonstrations is not None or env_reward_frac == 1.0, "No demonstrations have been loaded"

        # select which observations / actions to discriminate
        if not "actions" in demonstrations:
            act_mask = []

        self._state_mask = np.arange(demonstrations["states"].shape[1]) \
            if state_mask is None else np.array(state_mask, dtype=np.int64)

        self._act_mask = np.arange(demonstrations["actions"].shape[1]) \
            if act_mask is None else np.array(act_mask, dtype=np.int64)

        self._add_save_attr(
            discriminator_fit_params='pickle',
            _n_epochs_discriminator='pickle',
            _D='pickle',
            _env_reward_frac='pickle',
            _demonstrations='pickle',
            _act_mask='pickle',
            _state_mask='pickle',
        )

    def load_demonstrations(self, demonstrations):
        self._demonstrations = demonstrations

    def fit(self, dataset):
        state, action, reward, next_state, absorbing, last = parse_dataset(dataset)
        x = state.astype(np.float32)
        u = action.astype(np.float32)
        r = reward.astype(np.float32)
        xn = next_state.astype(np.float32)

        obs = to_float_tensor(x, self.policy.use_cuda)
        act = to_float_tensor(u, self.policy.use_cuda)

        if self._env_reward_frac < 1.0:
            # fit discriminator
            self._fit_discriminator(x, u)

            # create reward from the discriminator(can use fraction of environment reward)
            r_disc = self._make_discrim_reward(x, u)
            r = r * self._env_reward_frac + r_disc * (1 - self._env_reward_frac)

        v_target, np_adv = compute_gae(self._V, x, xn, r, absorbing, last,
                                       self.mdp_info.gamma, self._lambda)
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = to_float_tensor(np_adv, self.policy.use_cuda)

        # Policy update
        self._old_policy = deepcopy(self.policy)
        old_pol_dist = self._old_policy.distribution_t(obs)
        old_log_prob = self._old_policy.log_prob_t(obs, act).detach()

        zero_grad(self.policy.parameters())
        loss = self._compute_loss(obs, act, adv, old_log_prob)

        prev_loss = loss.item()

        # Compute Gradient
        loss.backward()
        g = get_gradient(self.policy.parameters())

        # Compute direction through conjugate gradient
        stepdir = self._conjugate_gradient(g, obs, old_pol_dist)

        # Line search
        self._line_search(obs, act, adv, old_log_prob, old_pol_dist, prev_loss, stepdir)

        # VF update
        self._V.fit(x, v_target, **self._critic_fit_params)

        # Print fit information
        self._print_fit_info(dataset, x, v_target, old_pol_dist)
        self._iter += 1

    def _fit_discriminator(self, plcy_obs, plcy_act):
        plcy_obs = plcy_obs[:, self._state_mask]
        plcy_act = plcy_act[:, self._act_mask]

        for epoch in range(self._n_epochs_discriminator):
            # get batch of data to discriminate
            if not self._act_mask.size > 0:
                demo_obs = next(minibatch_generator(plcy_obs.shape[0],
                                                    self._demonstrations["states"]))[0]
                inputs = np.concatenate([plcy_obs, demo_obs.astype(np.float32)])
            else:
                demo_obs, demo_act = next(minibatch_generator(plcy_obs.shape[0],
                                                              self._demonstrations["states"],
                                                              self._demonstrations["actions"]))
                plcy_data = np.concatenate([plcy_obs, plcy_act], axis=1)
                demo_data = np.concatenate([demo_obs, demo_act], axis=1)
                inputs = np.concatenate([plcy_data, demo_data.astype(np.float32)])

            # create label targets with noisy flipped labels: (demos(~0) or policy(~1))
            plcy_target = np.random.uniform(low=0.80, high=0.99, size=(plcy_obs.shape[0], 1)).astype(np.float32)
            demo_target = np.random.uniform(low=0.01, high=0.10, size=(plcy_obs.shape[0], 1)).astype(np.float32)
            targets = np.concatenate([plcy_target, demo_target])

            self._D.fit(inputs, targets, **self._discriminator_fit_params)

    @torch.no_grad()
    def _make_discrim_reward(self, state, action):
        plcy_data = np.concatenate([state[:, self._state_mask],
                                    action[:, self._act_mask]], axis=1)
        plcy_prob = self._D(plcy_data)
        return np.squeeze(-np.log(plcy_prob + 1e-8)).astype(np.float32)