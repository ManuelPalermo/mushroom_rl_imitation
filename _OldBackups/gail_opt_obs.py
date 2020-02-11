

import torch
import numpy as np
from tqdm import tqdm

from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.value_functions import compute_gae
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.ppo import PPO


class GAIL(PPO):
    """
    Generative Adversarial Imitation Learning(GAIL) implementation. Uses
        PPO policy updates instead of TRPO.

    "Generative Adversarial Imitation Learning"
    Ho, J., & Ermon, S. (2016).

    """

    def __init__(self, mdp_info, policy_class, policy_params, actor_optimizer,
                 discriminator_optimizer, discriminator_params, critic_params,
                 n_epochs_policy, n_epochs_discriminator,
                 batch_size_policy, batch_size_discriminator,
                 eps_ppo, lam, demonstrations=None,
                 discriminate_only_state=False,
                 quiet=True, critic_fit_params=None):

        # initialize PPO agent
        policy = policy_class(**policy_params)
        super(GAIL, self).__init__(mdp_info, policy, actor_optimizer, critic_params,
                                   n_epochs_policy, batch_size_policy, eps_ppo, lam,
                                   quiet=quiet, critic_fit_params=critic_fit_params)

        self._n_epochs_discriminator = n_epochs_discriminator

        # initialize discriminator
        self._device = "cuda:0" if self.policy.use_cuda else "cpu"
        self._D_approx = discriminator_params["network"](**discriminator_params)
        self._D_approx.to(self._device)
        self._D_optimizer = discriminator_optimizer['class'](self._D_approx.parameters(),
                                                                 **discriminator_optimizer['params'])
        self._D_loss = torch.nn.BCELoss()
        self._D_batch_size = batch_size_discriminator
        self._D_only_state = discriminate_only_state
        self._D_obs_ids = None
        self._D_act_ids = None

        self._demonstrations = demonstrations
        assert demonstrations is not None

        self._use_discriminator = demonstrations is not None

    def load_demonstrations(self, demonstrations):
        self._demonstrations = demonstrations

    def set_discriminator_state(self, state):
        # can only use descriminator if there are expert trajectories available
        self._use_discriminator = (state and self._demonstrations is not None)

    def fit(self, dataset):
        # overrides PPO fit to add discriminator update step
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        xn = xn.astype(np.float32)

        obs = to_float_tensor(x, self.policy.use_cuda)
        act = to_float_tensor(u, self.policy.use_cuda)

        if self._use_discriminator:
            # fit discriminator
            self._D_approx.train()
            self._fit_discriminator(obs, act)
            self._D_approx.eval()

            # create reward from the discriminator
            r = self._make_discrim_reward(obs, act).astype(np.float32)
            if not self._quiet:
                tqdm.write("Discr reward: {}".format(float(r.mean())))
        else:
            # use reward from the environment
            r = r.astype(np.float32)

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
        for _ in range(self._n_epochs_discriminator):
            # get some expert trajectories(using ExpertDataset from baselines)
            # should probably create mushroom dataset or use np.arrays directly
            demo_obs, demo_act = self._demonstrations.get_next_batch(split="train")

            # guarantee same number of samples for policy and expert datasets
            min_samples = min(demo_obs.shape[0], plcy_obs.shape[0])
            plcy_obs, plcy_act = plcy_obs[:min_samples, :], plcy_act[:min_samples, :]
            demo_obs, demo_act = demo_obs[:min_samples, :], demo_act[:min_samples, :]

            # load demos to torch tensor
            demo_obs = to_float_tensor(demo_obs, self.policy.use_cuda)
            demo_act = to_float_tensor(demo_act, self.policy.use_cuda)

            # if should discriminate state-actions or just states
            # (sometimes expert actions might not be available)
            if self._D_only_state:
                plcy_data = torch.squeeze(plcy_obs[:,self._D_obs_ids], dim=1)
                demo_data = demo_obs
            else:
                plcy_data = torch.squeeze(
                        torch.cat([plcy_obs[:,self._D_obs_ids], plcy_act[:,self._D_act_ids]], dim=1),
                        dim=1)
                demo_data = torch.cat([demo_obs, demo_act], dim=1)

            # create label targets: (demos(~1) or policy(~0))
            plcy_target = torch.zeros((plcy_data.shape[0], 1), device=self._device)
            demo_target = torch.ones((demo_data.shape[0], 1), device=self._device)

            # fit data in batches
            for plcy_data_i, demo_data_i, plcy_target_i, demo_target_i in \
                    minibatch_generator(self._D_batch_size,
                                        plcy_data, demo_data,
                                        plcy_target, demo_target):

                # discriminated samples(classified into expert demos(~1) or policy(~0))
                plcy_prob = self._D_approx(plcy_data_i)
                demo_prob = self._D_approx(demo_data_i)

                # calculate and propagate discriminator loss
                discr_loss = (self._D_loss(plcy_prob, plcy_target_i) +
                              self._D_loss(demo_prob, demo_target_i))

                self._D_optimizer.zero_grad()
                discr_loss.backward()
                self._D_optimizer.step()

        if not self._quiet:
            # can be printed for debugging(discriminator acc on expert/policy data)
            with torch.no_grad():
                tqdm.write("Disc mean:  expert({}), agent({})".format(
                        float(demo_prob.detach().cpu().numpy().mean()),
                        float(plcy_prob.detach().cpu().numpy().mean()))
                )
                tqdm.write("Disc acc:  expert({}), agent({})".format(
                        float((demo_prob.detach().cpu().numpy() > 0.5).mean()),
                        float((plcy_prob.detach().cpu().numpy() < 0.5).mean()))
                )

    def _make_discrim_reward(self, state, action):
        plcy_data = (state if self._D_only_state
                     else torch.cat([state, action], dim=1))
        with torch.no_grad():
            plcy_prob = self._D_approx(plcy_data).detach().cpu().numpy()
            return np.squeeze(-np.log(1 - plcy_prob + 1e-8))