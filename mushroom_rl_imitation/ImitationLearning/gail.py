
import torch
import numpy as np

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.value_functions import compute_gae
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.ppo import PPO
from mushroom_rl.utils.minibatches import minibatch_generator


class GAIL(PPO):
    """
    Generative Adversarial Imitation Learning(GAIL) implementation. Uses
        PPO policy updates instead of TRPO.

    "Generative Adversarial Imitation Learning"
    Ho, J., & Ermon, S. (2016).

    """

    def __init__(self, mdp_info, policy_class, policy_params,
                 discriminator_params, critic_params, actor_optimizer,
                 n_epochs_policy, batch_size_policy, eps_ppo, lam,
                 demonstrations=None, env_reward_frac=0.0,
                 state_mask=None, act_mask=None, quiet=True,
                 critic_fit_params=None, discriminator_fit_params=None):

        # initialize PPO agent
        policy = policy_class(**policy_params)
        super(GAIL, self).__init__(mdp_info, policy, actor_optimizer, critic_params,
                                   n_epochs_policy, batch_size_policy, eps_ppo, lam,
                                   quiet=quiet, critic_fit_params=critic_fit_params)

        # discriminator params
        self._discriminator_fit_params = (dict(n_epochs=1) if discriminator_fit_params is None
                                          else discriminator_fit_params)

        discriminator_params.setdefault("loss", torch.nn.BCELoss())
        discriminator_params.setdefault("batch_size", 128)
        self._D = Regressor(TorchApproximator, **discriminator_params)

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

    def load_demonstrations(self, demonstrations):
        self._demonstrations = demonstrations

    def fit(self, dataset):
        # overrides PPO fit to add discriminator update step
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)

        obs = to_float_tensor(x, self.policy.use_cuda)
        act = to_float_tensor(u, self.policy.use_cuda)

        if self._env_reward_frac < 1.0:
            # fit discriminator
            self._fit_discriminator(x, u)

            # create reward from the discriminator(can use fraction of environment reward)
            r_disc = self._make_discrim_reward(x, u)
            r = r * self._env_reward_frac + r_disc * (1 - self._env_reward_frac)

        # fit actor_critic
        v_target, np_adv = compute_gae(self._V, x, xn, r, absorbing, last,
                                       self.mdp_info.gamma, self._lambda)
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = to_float_tensor(np_adv, self.policy.use_cuda)

        old_pol_dist = self.policy.distribution_t(obs)
        old_log_p = old_pol_dist.log_prob(act)[:, None].detach()

        self._V.fit(x, v_target, **self._critic_fit_params)
        self._update_policy(obs, act, adv, old_log_p)

        # Print fit information
        self._print_fit_info(dataset, x, v_target, old_pol_dist)
        self._iter += 1

    def _fit_discriminator(self, plcy_obs, plcy_act):
        plcy_obs = plcy_obs[:, self._state_mask]
        plcy_act = plcy_act[:, self._act_mask]

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

        # create label targets: (demos(~1) or policy(~0))
        plcy_target = np.zeros((plcy_obs.shape[0], 1), dtype=np.float32)
        demo_target = np.ones((demo_obs.shape[0], 1), dtype=np.float32)
        targets = np.concatenate([plcy_target, demo_target])

        self._D.fit(inputs, targets, **self._discriminator_fit_params)

    def _make_discrim_reward(self, state, action):
        plcy_data = np.concatenate([state[:, self._state_mask],
                                    action[:, self._act_mask]], axis=1)
        with torch.no_grad():
            plcy_prob = self._D(plcy_data)
            return np.squeeze(-np.log(1 - plcy_prob + 1e-8)).astype(np.float32)