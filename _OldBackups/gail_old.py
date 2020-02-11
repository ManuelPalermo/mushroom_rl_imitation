import numpy as np

import torch

from mushroom_rl.algorithms.actor_critic.deep_actor_critic.ppo import PPO

from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.value_functions import compute_gae


class GAIL(PPO):
    """
    Generative Adversarial Imitation Learning(GAIL) algorithm.
    "Generative Adversarial Imitation Learning"
    Ho, J., & Ermon, S. (2016).

    """

    def __init__(self, mdp_info, policy_class, policy_params, actor_optimizer,
                 discriminator_optimizer, discriminator_params, critic_params,
                 n_epochs_policy, n_epochs_discriminator,
                 batch_size, eps_ppo, lam, demonstrations=None,
                 quiet=True, critic_fit_params=None):

        # initialize PPO agent
        policy = policy_class(**policy_params)
        super(GAIL, self).__init__(mdp_info, policy, actor_optimizer, critic_params,
                                   n_epochs_policy, batch_size, eps_ppo, lam,
                                   quiet=quiet, critic_fit_params=critic_fit_params)

        self._n_epochs_discriminator = n_epochs_discriminator

        # initialize discriminator
        self._device = "cuda:0" if self.policy.use_cuda else "cpu"
        self._discr_approx = discriminator_params["network"](**discriminator_params)
        self._discr_approx.to(self._device)
        self._discr_optimizer = discriminator_optimizer['class'](self._discr_approx.parameters(),
                                                                 **discriminator_optimizer['params'])
        self._discr_loss = torch.nn.BCELoss()

        # if discriminator net should be used(or just PPO)
        # discriminator can be turned off to train policy directly
        # on env(instead of imitating expert trajectories)
        self._demonstrations = demonstrations
        self._use_discriminator = demonstrations is not None

    def load_demonstrations(self, demonstrations):
        self._demonstrations = demonstrations

    def fit(self, dataset):
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)

        obs = to_float_tensor(x, self.policy.use_cuda)
        act = to_float_tensor(u, self.policy.use_cuda)

        if self._use_discriminator:
            # fit discriminator
            r = self._make_discrim_reward(obs, act).astype(np.float32)
            self._fit_discriminator(obs, act)

        # fit actor_critic
        v_target, np_adv = compute_gae(self._V, x, xn, r, absorbing, last,
                                       self.mdp_info.gamma, self._lambda)
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = to_float_tensor(np_adv, self.policy.use_cuda)

        old_pol_dist = self.policy.distribution_t(obs)
        old_log_p = old_pol_dist.log_prob(act)[:, None].detach()

        self._V.fit(x, v_target, **self._critic_fit_params)
        self._update_policy(obs, act, adv, old_log_p)

    def _fit_discriminator(self, obs, act):
        self._discr_approx.train()
        np_demo_obs, np_demo_acts = self._demonstrations.get_next_batch(split="train")

        # guarantee same number of samples for policy and expert datasets
        min_samples = min(np_demo_obs.shape[0], obs.shape[0])
        obs, act = obs[:min_samples, :], act[:min_samples, :]
        np_demo_obs, np_demo_acts = np_demo_obs[:min_samples, :], np_demo_acts[:min_samples, :]

        demo_obs = to_float_tensor(np_demo_obs, self.policy.use_cuda)
        demo_act = to_float_tensor(np_demo_acts, self.policy.use_cuda)

        learner = self._discr_approx(torch.cat([obs, act], dim=1))
        expert = self._discr_approx(torch.cat([demo_obs, demo_act], dim=1))

        discrim_loss = self._discr_loss(learner, torch.ones((obs.shape[0], 1), device=self._device)) + \
                       self._discr_loss(expert, torch.zeros((demo_obs.shape[0], 1), device=self._device))

        self._discr_optimizer.zero_grad()
        discrim_loss.backward()
        self._discr_optimizer.step()

    def _make_discrim_reward(self, state, action):
        self._discr_approx.eval()
        state_action = torch.cat([state, action], dim=1)
        with torch.no_grad():
            return np.squeeze(-np.log(self._discr_approx(state_action).detach().cpu().numpy()))
