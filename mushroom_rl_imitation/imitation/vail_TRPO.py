from copy import deepcopy

import numpy as np

import torch
from mushroom_rl.algorithms.actor_critic import TRPO
from torch import nn
from torch.nn.modules.loss import BCELoss
import torch.nn.functional as F

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.torch import get_gradient, zero_grad, to_float_tensor
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.value_functions import compute_gae


class VDBnetwork(nn.Module):
    """
    Variational Discriminator Bottleneck network.

    "Variational Discriminator Bottleneck: Improving Imitation Learning,
        Inverse RL, and GANs by Constraining Information Flow"
    Peng X. et al.. 2019.

    """
    def __init__(self, input_shape, output_shape, n_features, z_size, **kwargs):
        super(VDBnetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[-1] # should always be 1(just like this for consistency)

        # encoder
        self._in = nn.Linear(n_input, n_features)
        self._z_mu = nn.Linear(n_features, z_size)
        self._z_v = nn.Linear(n_features, z_size)

        # decoder
        self._h1 = nn.Linear(z_size, n_features)
        self._out = nn.Linear(n_features, n_output)

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
        self._beta = 0

    def forward(self, inputs, target):
        prob, mu, logvar = inputs

        # bottleneck loss
        kld = kl_divergence(mu, logvar).mean()
        bottleneck_loss = kld - self._info_constr
        self._beta = max(0, self._beta + self._lr_beta * bottleneck_loss)

        # calculate discriminator loss(BinaryCrossEntropy + bottleneck)
        vdb_loss = (F.binary_cross_entropy(prob, target, weight=self.weight, reduction=self.reduction)
                    + self._beta * bottleneck_loss)
        return vdb_loss


def kl_divergence(mu, logvar):
    kl_div = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1)
    return kl_div


class TorchApproximator_(TorchApproximator):
    def _fit_batch(self, batch, use_weights, kwargs):
        loss = self._compute_batch_loss(batch, use_weights, kwargs)

        # needs to retain graph for the dual gradient descent(beta)
        self._optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self._optimizer.step()

        return loss.item()


class VAIL(TRPO):
    """
    Variational Adversarial Imitation Learning(VAIL) implementation

    "Variational Discriminator Bottleneck: Improving Imitation Learning,
        Inverse RL, and GANs by Constraining Information Flow"
    Peng X. et al.. 2019.

    """

    def __init__(self, mdp_info, policy_class, policy_params,
                 discriminator_params, critic_params,
                 ent_coeff=0., max_kl=.001, lam=1.,
                 n_epochs_line_search=10, n_epochs_cg=10,
                 cg_damping=1e-2, cg_residual_tol=1e-10,
                 demonstrations=None, info_constraint=0.5, lr_beta=1e-4,
                 env_reward_frac=0.0, state_mask=None, act_mask=None, quiet=True,
                 critic_fit_params=None, discriminator_fit_params=None):

        # initialize PPO agent
        policy = policy_class(**policy_params)
        super(VAIL, self).__init__(mdp_info, policy, critic_params,
                                   ent_coeff, max_kl, lam, n_epochs_line_search,
                                   n_epochs_cg, cg_damping, cg_residual_tol,
                                   quiet=quiet, critic_fit_params=critic_fit_params)

        # discriminator params
        self._discriminator_fit_params = (dict(n_epochs=1) if discriminator_fit_params is None
                                          else discriminator_fit_params)

        discriminator_params["network"] = VDBnetwork
        discriminator_params["loss"] = VDBloss(info_constraint, lr_beta)
        discriminator_params.setdefault("z_size", 8)
        discriminator_params.setdefault("batch_size", 128)
        self._D = Regressor(TorchApproximator_, **discriminator_params)

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

        self._D._impl.model._loss._beta = 0  # set dual beta=0
        self._D.fit(inputs, targets, **self._discriminator_fit_params)

    def _make_discrim_reward(self, state, action):
        plcy_data = np.concatenate([state[:, self._state_mask],
                                    action[:, self._act_mask]], axis=1)
        with torch.no_grad():
            plcy_prob = self._D(plcy_data)[0]
            return np.squeeze(-np.log(1 - plcy_prob + 1e-8)).astype(np.float32)
