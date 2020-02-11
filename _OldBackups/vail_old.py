

import torch
from torch import nn

import numpy as np
from tqdm import tqdm

from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.value_functions import compute_gae
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.ppo import PPO


def kl_divergence(mu, logvar):
    kl_div = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1)
    return kl_div


class VDBnetwork(nn.Module):
    """
    Variational Discriminator Bottleneck network.

    """
    def __init__(self, input_shape, output_shape, n_features, z_size, **kwargs):
        super(VDBnetwork, self).__init__()
        # encoder
        self._in = nn.Linear(input_shape, n_features)
        self._z_mu = nn.Linear(n_features, z_size)
        self._z_v = nn.Linear(n_features, z_size)

        # decoder
        self._h1 = nn.Linear(z_size, n_features)
        self._out = nn.Linear(n_features, output_shape)

    def encoder(self, x):
        h = torch.tanh(self._in(x))
        return self._z_mu(h), self._z_v(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def discriminator(self, z):
        h = torch.tanh(self._h1(z))
        return torch.sigmoid(self._out(h))

    def forward(self, x, **kwargs):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        prob = self.discriminator(z)
        return prob, mu, logvar



class VAIL(PPO):
    """
    Variational Adversarial Imitation Learning(VAIL) implementation

    "Variational Discriminator Bottleneck: Improving Imitation Learning,
        Inverse RL, and GANs by Constraining Information Flow"
    Peng X. et al.. 2019.

    """

    def __init__(self, mdp_info, policy_class, policy_params, actor_optimizer,
                 discriminator_optimizer, discriminator_params, critic_params,
                 n_epochs_policy, n_epochs_discriminator,
                 batch_size_policy, batch_size_discriminator,
                 eps_ppo, lam, demonstrations=None,
                 discriminate_only_state=False,
                 info_constraint=0.5, lr_beta=1e-4,
                 quiet=True, critic_fit_params=None):

        # initialize PPO agent
        policy = policy_class(**policy_params)
        super(VAIL, self).__init__(mdp_info, policy, actor_optimizer, critic_params,
                                   n_epochs_policy, batch_size_policy, eps_ppo, lam,
                                   quiet=quiet, critic_fit_params=critic_fit_params)

        self._n_epochs_discriminator = n_epochs_discriminator

        # initialize discriminator
        self._device = "cuda:0" if self.policy.use_cuda else "cpu"
        self._discr_approx = VDBnetwork(**discriminator_params)
        self._discr_approx.to(self._device)
        self._discr_optimizer = discriminator_optimizer['class'](self._discr_approx.parameters(),
                                                                 **discriminator_optimizer['params'])
        self._discr_loss = torch.nn.BCELoss()
        self._discr_batch_size = batch_size_discriminator
        self._discr_only_state = discriminate_only_state

        # initialize information constraints
        self._info_constr = info_constraint
        self._lr_beta = lr_beta

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
        # overrides PPO fit to add discriminator update step
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
        beta = 0
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
            if self._discr_only_state:
                plcy_data = plcy_obs
                demo_data = demo_obs
            else:
                plcy_data = torch.cat([plcy_obs, plcy_act], dim=1)
                demo_data = torch.cat([demo_obs, demo_act], dim=1)

            # create label targets: (demos(~1) or policy(~0))
            plcy_target = torch.zeros((plcy_data.shape[0], 1), device=self._device)
            demo_target = torch.ones((demo_data.shape[0], 1), device=self._device)

            # fit data in batches
            for plcy_data_i, demo_data_i, plcy_target_i, demo_target_i in \
                    minibatch_generator(self._discr_batch_size,
                                        plcy_data, demo_data,
                                        plcy_target, demo_target):

                # discriminated samples(classified into expert demos(~1) or policy(~0))
                plcy_prob, plcy_mu, plcy_logvar = self._discr_approx(plcy_data_i)
                demo_prob, demo_mu, demo_logvar = self._discr_approx(demo_data_i)

                # calculate the KL divergence?
                plcy_kld = kl_divergence(plcy_mu, plcy_logvar).mean()
                demo_kld = kl_divergence(demo_mu, demo_logvar).mean()
                kld = 0.5 * (plcy_kld + demo_kld)

                # calculate the bottleneck loss
                bottleneck_loss = kld - self._info_constr
                beta = max(0, beta + self._lr_beta * bottleneck_loss)

                # calculate discriminator loss(demo data + plcy data + bottleneck)
                vdb_loss = self._discr_loss(plcy_prob, plcy_target_i) + \
                           self._discr_loss(demo_prob, demo_target_i) + \
                           beta * bottleneck_loss

                self._discr_optimizer.zero_grad()
                vdb_loss.backward(retain_graph=True)
                self._discr_optimizer.step()

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
        plcy_data = (state if self._discr_only_state
                     else torch.cat([state, action], dim=1))
        with torch.no_grad():
            plcy_prob = self._discr_approx(plcy_data)[0].detach().cpu().numpy()
            return np.squeeze(-np.log(1 - plcy_prob + 1e-8))
