

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
    Generative Adversarial Imitation Learning(GAIL) algorithm. Uses
        PPO policy updates instead of TRPO. Uses some small GAN tricks
        to help stabilizing the training training

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

    def set_discriminator_state(self, state):
        # can only use descriminator if there are expert trajectories available
        self._use_discriminator = (state and self._demonstrations is not None)

    def fit(self, dataset):
        # overrides fit to add discriminator update step
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        xn = xn.astype(np.float32)

        obs = to_float_tensor(x, self.policy.use_cuda)
        act = to_float_tensor(u, self.policy.use_cuda)

        if self._use_discriminator:
            # fit discriminator
            self._discr_approx.train()
            self._fit_discriminator(obs, act)
            self._discr_approx.eval()

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

            for plcy_obs_i, plcy_act_i, demo_obs_i, demo_act_i in \
                    minibatch_generator(self._batch_size,
                                        plcy_obs, plcy_act,
                                        demo_obs, demo_act):

                # create targets with smooth labels: (demos(~0) or policy(~1))
                d_demo_target = torch.FloatTensor(demo_obs_i.shape[0], 1).uniform_(0.01, 0.10).to(self._device)
                d_plcy_target = torch.FloatTensor(plcy_obs_i.shape[0], 1).uniform_(0.90, 0.99).to(self._device)
                #d_demo_target = torch.ones((demo_obs_i.shape[0], 1), device=self._device)   # no label smoothing
                #d_plcy_target = torch.zeros((plcy_obs_i.shape[0], 1), device=self._device)  # no label smoothing

                # discriminated samples(classified into expert demos(~0) or policy(~1))
                d_plcy = self._discr_approx(torch.cat([plcy_obs_i, plcy_act_i], dim=1))
                d_demo = self._discr_approx(torch.cat([demo_obs_i, demo_act_i], dim=1))

                # calculate and propagate discriminator loss
                discr_loss = (self._discr_loss(d_plcy, d_plcy_target) +
                             (self._discr_loss(d_demo, d_demo_target)))

                self._discr_optimizer.zero_grad()
                discr_loss.backward()
                self._discr_optimizer.step()

        if not self._quiet:
            # can be printed for debugging(discriminator acc on expert/policy data)
            with torch.no_grad():
                tqdm.write("Disc mean:  expert({}), agent({})".format(
                        float(d_demo.detach().cpu().numpy().mean()),
                        float(d_plcy.detach().cpu().numpy().mean()))
                )
                tqdm.write("Disc acc:  expert({}), agent({})".format(
                        float((d_demo.detach().cpu().numpy() > 0.5).mean()),
                        float((d_plcy.detach().cpu().numpy() < 0.5).mean()))
                )

    def _make_discrim_reward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        with torch.no_grad():
            return np.squeeze(-np.log(self._discr_approx(state_action).detach().cpu().numpy()))