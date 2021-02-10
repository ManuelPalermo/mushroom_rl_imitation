
import numpy as np

import torch
from torch import nn
from torch.nn.modules.loss import BCELoss
import torch.nn.functional as F

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.value_functions import compute_gae
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.ppo import PPO

from mushroom_rl_imitation.utils.minibatch_generator_extra import \
    dataset_as_sequential, minibatch_sample_sequential


class VAENet(nn.Module):
    """
    Variational-Autoencoder network.

    """
    def __init__(self, input_shape, output_shape, n_features, z_size, seq_size, **kwargs):
        super(VAENet, self).__init__()

        n_input = input_shape[-1] * seq_size
        n_output = output_shape[-1]

        if isinstance(n_features, int):
            n_features = [n_features, n_features // 2]

        # encoder
        self._enc0 = nn.Linear(n_input, n_features[0])
        self._enc1 = nn.Linear(n_features[0], n_features[1])

        self._z_mu = nn.Linear(n_features[1], z_size)
        self._z_v = nn.Linear(n_features[1], z_size)

        # decoder
        self._h1 = nn.Linear(z_size, n_features[1])
        self._dec0 = nn.Linear(n_features[1], n_features[0])
        self._dec1 = nn.Linear(n_features[0], n_output)

    def encoder(self, x):
        h = torch.relu(self._enc0(x))
        h = torch.relu(self._enc1(h))
        return self._z_mu(h), self._z_v(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def discriminator(self, z):
        h = torch.relu(self._h1(z))
        h = torch.relu(self._dec0(h))
        return torch.sigmoid(self._dec1(h))

    def forward(self, x, **kwargs):
        x = x.reshape(x.shape[0], -1)  # flatten time dimension
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        prob = self.discriminator(z)
        return prob, mu, logvar


class VDBloss(BCELoss):
    """
    Variational Discriminator Bottleneck loss.

    "Variational Discriminator Bottleneck: Improving Imitation Learning,
        Inverse RL, and GANs by Constraining Information Flow"
    Peng X. et al.. 2019.

    """
    __constants__ = ['reduction', 'weight']

    def __init__(self, info_constraint, lr_beta,
                 weight=None, size_average=None, reduce=None, reduction='mean'):
        super(VDBloss, self).__init__(weight, size_average, reduce, reduction)

        self._info_constr = info_constraint
        self._lr_beta = lr_beta
        self._beta = 0.1

    def forward(self, inputs, target):
        prob, mu, logvar = inputs

        # bottleneck loss
        kld = kl_divergence(mu, logvar).mean()
        bottleneck_loss = kld - self._info_constr

        # calculate discriminator loss(BinaryCrossEntropy + bottleneck_regularization)
        vdb_loss = (F.binary_cross_entropy(prob, target, weight=self.weight, reduction=self.reduction)
                    + self._beta * bottleneck_loss)

        self._update_beta(bottleneck_loss)
        return vdb_loss

    @torch.no_grad()
    def _update_beta(self, bottleneck_loss):
        self._beta = max(0, self._beta + self._lr_beta * bottleneck_loss)


def kl_divergence(mu, logvar):
    kl_div = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1)
    return kl_div


class VAIL(PPO):
    """
    Variational Adversarial Imitation Learning(VAIL) implementation

    "Variational Discriminator Bottleneck: Improving Imitation Learning,
        Inverse RL, and GANs by Constraining Information Flow"
    Peng X. et al.. 2019.

    """

    def __init__(self, mdp_info, policy_class, policy_params,
                 discriminator_params, critic_params, actor_optimizer,
                 n_epochs_policy, n_epochs_discriminator, batch_size_policy,
                 eps_ppo, lam, ent_coeff=0.01, demonstrations=None,
                 info_constraint=0.5, lr_beta=1e-4, env_reward_frac=0.0,
                 state_mask=None, act_mask=None, disc_seq_size=2,
                 critic_fit_params=None, discriminator_fit_params=None):

        # initialize PPO agent
        policy = policy_class(**policy_params)
        super(VAIL, self).__init__(mdp_info, policy, actor_optimizer, critic_params,
                                   n_epochs_policy, batch_size_policy, eps_ppo, lam,
                                   ent_coeff=ent_coeff,
                                   critic_fit_params=critic_fit_params)

        # discriminator params
        self._discriminator_fit_params = (dict() if discriminator_fit_params is None
                                          else discriminator_fit_params)

        discriminator_params["network"] = VAENet
        discriminator_params["loss"] = VDBloss(info_constraint, lr_beta)
        discriminator_params.setdefault("z_size", 8)
        discriminator_params.setdefault("batch_size", 128)
        discriminator_params.setdefault("output_shape", (1,))
        self._D = Regressor(TorchApproximator, **discriminator_params)
        self._n_epochs_discriminator = n_epochs_discriminator
        self._disc_seq_size = disc_seq_size
        assert disc_seq_size > 1, "Sequence size should be bigger greater than 1." \
                                  " Otherwise use standard GAIL"

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
            _discriminator_fit_params='pickle',
            _D='pickle',
            _env_reward_frac='pickle',
            _demonstrations='pickle',
            _act_mask='pickle',
            _state_mask='pickle',
        )

    def load_demonstrations(self, demonstrations):
        self._demonstrations = demonstrations

    def fit(self, dataset):
        # overrides PPO fit to add discriminator update step
        x, u, r, xn, absorbing, last = parse_dataset(dataset)

        x_disc, u_disc = dataset_as_sequential(x, u, seq_size=self._disc_seq_size)
        x_disc = x_disc.astype(np.float32)
        u_disc = u_disc.astype(np.float32)

        x = x.astype(np.float32)[:-(self._disc_seq_size - 1)]
        u = u.astype(np.float32)[:-(self._disc_seq_size - 1)]
        r = r.astype(np.float32)[:-(self._disc_seq_size - 1)]
        xn = xn.astype(np.float32)[:-(self._disc_seq_size - 1)]
        absorbing = absorbing[:-(self._disc_seq_size - 1)]
        last = last[:-(self._disc_seq_size - 1)]

        obs = to_float_tensor(x, self.policy.use_cuda)
        act = to_float_tensor(u, self.policy.use_cuda)

        if self._env_reward_frac < 1.0:
            # fit discriminator
            self._fit_discriminator(x_disc, u_disc)

            # create reward from the discriminator(can use fraction of environment reward)
            r_disc = self._make_discrim_reward(x_disc, u_disc)
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
        self._log_info(dataset, x, v_target, old_pol_dist)
        self._iter += 1

    def _fit_discriminator(self, plcy_obs, plcy_act):
        plcy_obs = plcy_obs[:, :, self._state_mask]
        plcy_act = plcy_act[:, :, self._act_mask]

        for epoch in range(self._n_epochs_discriminator):
            # get batch of data to discriminate
            if not self._act_mask.size > 0:
                demo_obs = minibatch_sample_sequential(plcy_obs.shape[0], self._disc_seq_size,
                                                       self._demonstrations["states"])[0]
                inputs = np.concatenate([plcy_obs, demo_obs.astype(np.float32)])
            else:
                demo_obs, demo_act = minibatch_sample_sequential(plcy_obs.shape[0], self._disc_seq_size,
                                                                 self._demonstrations["states"],
                                                                 self._demonstrations["actions"])
                plcy_data = np.concatenate([plcy_obs, plcy_act], axis=2)
                demo_data = np.concatenate([demo_obs, demo_act], axis=2)
                inputs = np.concatenate([plcy_data, demo_data.astype(np.float32)])

            # create label targets with flipped labels: (demos(~0) or policy(~1))
            plcy_target = np.ones((plcy_obs.shape[0], 1), dtype=np.float32)
            demo_target = np.zeros((demo_obs.shape[0], 1), dtype=np.float32)
            targets = np.concatenate([plcy_target, demo_target])

            self._D.fit(inputs, targets, **self._discriminator_fit_params)

    @torch.no_grad()
    def _make_discrim_reward(self, state, action):
        plcy_data = np.concatenate([state[:, :, self._state_mask],
                                    action[:, :, self._act_mask]], axis=2)
        plcy_prob = self._D(plcy_data)[0]
        return np.squeeze(-np.log(plcy_prob + 1e-8)).astype(np.float32)
