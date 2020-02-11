import numpy as np
from copy import deepcopy

from mushroom_rl.utils.running_stats import RunningStandardization
from _wrapping_envs.WrappingEnv import WrappingEnvironment
from mushroom_rl.utils.spaces import Box


class NormalizedEnv(WrappingEnvironment):
    """
    Implements a wrapping environment which can be used to normalize
        the base environment using RunningStandardization. Can be
        used to normalize observations and rewards and clip extreme
        values from actions which could make the environment's
        simulation unstable. Especially useful when using neural
        networks to interact with the environment.

    """
    def __init__(self, env_class, env_kwargs=None,
                 run_norm_obs=True, run_norm_rew=False,
                 clip_act=1e4, clip_obs=10., clip_rew=10.,
                 alpha=1e-32, warmup_runstats_n_samples=100):
        """
        Constructor.

        Args:
            env_class (Environment): Environment class to be used.
            env_kwargs (dict): Parameters to instantiate environment
                class with.
            run_norm_obs (bool): If running norm should be applied to
                the observations.
            run_norm_rew (bool): If running norm should be applied to
                the rewards.
            clip_act (float): Values to clip the actions.
            clip_obs (float): Values to clip the normalized
                observations.
            clip_rew (float): Values to clip the normalized
                rewards.
            alpha (float): Moving average catchup parameter for
                the normalization.
            warmup_runstats_n_samples (int): Number of steps to apply
                random actions in order to initialize the running
                average statistics.

        """

        self.run_norm_obs = run_norm_obs
        self.run_norm_rew = run_norm_rew

        self.clip_act = clip_act
        self.clip_obs = clip_obs
        self.clip_rew = clip_rew

        super(NormalizedEnv, self).__init__(env_class=env_class,
                                            env_kwargs=env_kwargs)

        # only applies normalization to boxed actions spaces
        self.is_boxed_act = isinstance(self.environment.info.action_space, Box)
        self.is_boxed_obs = isinstance(self.environment.info.observation_space, Box)

        if not self.is_boxed_act:
            self.clip_act = False

        if not self.is_boxed_obs:
            self.clip_obs = False
            self.run_norm_obs = False

        # creates running standardization buffers
        self.rew_runstand = RunningStandardization(shape=(), alpha=alpha)
        self.obs_runstand = RunningStandardization(shape=self.info.observation_space.shape,
                                                   alpha=alpha)

        self._warmup_running_stats(warmup_runstats_n_samples)

    def _create_new_mdp_info(self):
        """
        Change mdp_info to reflect new environment as seen from
            outside, with observations and actions surely limited
            between the clip values.
        """
        new_mdp_info = deepcopy(self.environment.info)

        if isinstance(self.environment.info.action_space, Box):
            new_mdp_info.action_space._low = \
                -np.ones(new_mdp_info.action_space.shape[0]) * self.clip_act
            new_mdp_info.action_space._high = \
                np.ones(new_mdp_info.action_space.shape[0]) * self.clip_act

        if isinstance(self.environment.info.observation_space, Box):
            new_mdp_info.observation_space._low = \
                -np.ones(new_mdp_info.observation_space.shape[0]) * self.clip_obs
            new_mdp_info.observation_space._high = \
                np.ones(new_mdp_info.observation_space.shape[0]) * self.clip_obs
        return new_mdp_info

    def reset(self, state=None):
        """
        Resets the environment and normalizes the returned observation.

        Returns:
            A normalized observation.
        """
        obs = self.environment.reset()
        if self.run_norm_obs:
            return np.clip((obs - self.obs_runstand.mean) / (self.obs_runstand.std), -self.clip_obs, self.clip_obs)
        return obs

    def step(self, action):
        """
        Steps the environment by applying a normalized action and
            returns a normalized observation/reward.
        """
        action = np.clip(action, -self.clip_act, self.clip_act)
        obs, reward, done, info = self.environment.step(action)

        if self.run_norm_obs:
            self.obs_runstand.update_stats(obs)
            obs = np.clip((obs - self.obs_runstand.mean) / (self.obs_runstand.std), -self.clip_obs, self.clip_obs)

        if self.run_norm_rew:
            self.rew_runstand.update_stats(reward)
            reward = np.clip((reward - self.rew_runstand.mean) / (self.rew_runstand.std), -self.clip_rew, self.clip_rew)

        return obs, reward, done, info

    def _get_wrapping_env_state(self):
        data = dict(normalize_data=dict(obs_stand=(self.obs_runstand.get_state()
                                                   if self.is_boxed_obs else dict()),

                                        rew_stand=self.rew_runstand.get_state()))
        return data

    def _set_wrapping_env_state(self, data):
        normalize_data = data["normalize_data"]
        self.obs_runstand.set_state(normalize_data["obs_stand"])
        self.rew_runstand.set_state(normalize_data["rew_stand"])

    def _warmup_running_stats(self, n_warmup_samples=100):
        s = 0
        self.reset()
        while s < n_warmup_samples:
            done = False
            while not done:
                if isinstance(self.environment.info.action_space, Box):
                    random_action = np.random.uniform(low=self.info.action_space.low / 2.0,
                                                      high=self.info.action_space.high / 2.0,
                                                      size=self.info.action_space.shape[0])
                else:
                    random_action = np.random.randint(low=0,
                                                      high=self.info.action_space.n)

                obs, reward, done, _ = self.step(random_action)
                s += 1
            self.reset()


class NormalizedBoxedEnv(NormalizedEnv):
    def __init__(self, env_class, env_kwargs=None,
                 run_norm_obs=True, run_norm_rew=False,
                 clip_act=1e4, clip_obs=10., clip_rew=10.,
                 alpha=1e-32, warmup_runstats_n_samples=0):
        """
        Constructor.

        Args:
            env_class (Environment): Environment class to be used.
            env_kwargs (dict): Parameters to instantiate environment
                class with.
            run_norm_obs (bool): If running norm should be applied to
                the observations.
            run_norm_rew (bool): If running norm should be applied to
                the rewards.
            clip_act (float): Values to clip the actions.
            clip_obs (float): Values to clip the normalized
                observations.
            clip_rew (float): Values to clip the normalized
                rewards.
            alpha (float): Moving average catchup parameter for
                the normalization.
            warmup_runstats_n_samples (int): Number of steps to apply
                random actions in order to initialize the running
                average statistics.

        """

        super(NormalizedBoxedEnv, self).__init__(env_class=env_class, run_norm_obs=run_norm_obs,
                                                 run_norm_rew=run_norm_rew, clip_act=clip_act,
                                                 clip_obs=clip_obs, clip_rew=clip_rew,
                                                 alpha=alpha, env_kwargs=env_kwargs,
                                                 warmup_runstats_n_samples=0)

        if self.is_boxed_act:
            # create mask where actions will be normalized between
            # boxed values(not inf)
            act_low, act_high = (self.environment.info.action_space.low.copy(),
                                 self.environment.info.action_space.high.copy())
            self.stand_act_mask = np.where(~(np.isinf(act_low) | np.isinf(act_high)))

            # mean/delta values to use for normalization for actions
            self.act_mean = np.zeros_like(act_low)
            self.act_delta = np.ones_like(act_low)
            self.act_mean[self.stand_act_mask] = (act_high[self.stand_act_mask]
                                                  + act_low[self.stand_act_mask]) / 2.0
            self.act_delta[self.stand_act_mask] = (act_high[self.stand_act_mask]
                                                   - act_low[self.stand_act_mask]) / 2.0

        if self.is_boxed_obs:
            # create mask where observations will be normalized between
            # boxed values(not inf)
            obs_low, obs_high = (self.environment.info.observation_space.low.copy(),
                                 self.environment.info.observation_space.high.copy())

            self.stand_obs_mask = np.where(~(np.isinf(obs_low) | np.isinf(obs_high)))

            # turn off running stats if all observations will be boxed
            self.run_norm_obs = len(np.squeeze(self.stand_obs_mask)) != obs_low.shape[0]

            # mean/delta values to use for normalization for observations
            self.obs_mean = np.zeros_like(obs_low)
            self.obs_delta = np.ones_like(obs_low)
            self.obs_mean[self.stand_obs_mask] = (obs_high[self.stand_obs_mask]
                                                  + obs_low[self.stand_obs_mask]) / 2.0
            self.obs_delta[self.stand_obs_mask] = (obs_high[self.stand_obs_mask]
                                                   - obs_low[self.stand_obs_mask]) / 2.0


        # Change mdp_info to reflect new environment as seen from
        # outside, with observations and actions limited
        # between the clip values where original environment was
        # not limited, and boxed observations limited between [-1,1]
        # as they wil be standardized.

        # set new mdp_info action space
        if isinstance(self.environment.info.action_space, Box):
            self.info.action_space._low[self.stand_act_mask] = \
                -np.ones(self.info.action_space.shape[0])[self.stand_act_mask]
            self.info.action_space._high[self.stand_act_mask] = \
                np.ones(self.info.action_space.shape[0])[self.stand_act_mask]

        # set new mdp_info observation space
        if isinstance(self.environment.info.observation_space, Box):
            self.info.observation_space._low[self.stand_obs_mask] = \
                -np.ones(self.info.observation_space.shape[0])[self.stand_obs_mask]
            self.info.observation_space._high[self.stand_obs_mask] = \
                np.ones(self.info.observation_space.shape[0])[self.stand_obs_mask]

        self._warmup_running_stats(warmup_runstats_n_samples)

    def step(self, action):
        denormalized_action = action.copy()

        if self.is_boxed_act:
            denormalized_action[self.stand_act_mask] = \
                ((action * self.act_delta) + self.act_mean)[self.stand_act_mask]
            denormalized_action = np.clip(denormalized_action, -self.clip_act, self.clip_act)

        obs, reward, done, info = self.environment.step(denormalized_action)
        orig_obs = obs.copy()

        if self.is_boxed_obs:
            if self.run_norm_obs:
                self.obs_runstand.update_stats(obs)
                obs = np.clip((obs - self.obs_runstand.mean) / (self.obs_runstand.std), -self.clip_obs, self.clip_obs)

            obs[self.stand_obs_mask] = \
                ((orig_obs - self.obs_mean) / self.obs_delta)[self.stand_obs_mask]

        if self.run_norm_rew:
            self.rew_runstand.update_stats(reward)
            reward = np.clip((reward - self.rew_runstand.mean) / (self.rew_runstand.std), -self.clip_rew, self.clip_rew)

        return obs, reward, done, info
