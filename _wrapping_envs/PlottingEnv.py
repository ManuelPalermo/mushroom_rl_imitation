
import numpy as np
from mushroom_rl.utils import plots
from _wrapping_envs import WrappingEnvironment
from mushroom_rl.utils.spaces import Box


class PlottingEnv(WrappingEnvironment):
    def __init__(self, env_class, env_kwargs=None,
                 window_size=1000, update_freq=10,
                 obs_normalized=False):

        super(PlottingEnv, self).__init__(env_class=env_class,
                                          env_kwargs=env_kwargs)

        mdp_info = self.info
        # create buffers
        self.action_buffers_list = []
        for i in range(mdp_info.action_space.shape[0]):
            self.action_buffers_list.append(
                    plots.DataBuffer('Action_' + str(i), window_size))

        self.observation_buffers_list = []
        for i in range(mdp_info.observation_space.shape[0]):
            self.observation_buffers_list.append(
                    plots.DataBuffer('Observation_' + str(i), window_size))

        self.instant_reward_buffer = \
            plots.DataBuffer("Instant_reward", window_size)

        self.training_reward_buffer = plots.DataBuffer("Episode_reward")

        self.episodic_len_buffer_training = plots.DataBuffer("Episode_len")

        if isinstance(mdp_info.action_space, Box):
            high_actions = mdp_info.action_space.high.tolist()
            low_actions = mdp_info.action_space.low.tolist()
        else:
            high_actions = None
            low_actions = None

        # create plots
        actions_plot = plots.common_plots.Actions(self.action_buffers_list,
                                                  maxs=high_actions,
                                                  mins=low_actions)

        dotted_limits = None
        if isinstance(mdp_info.observation_space, Box):
            high_mdp = mdp_info.observation_space.high.tolist()
            low_mdp = mdp_info.observation_space.low.tolist()
            if obs_normalized:
                dotted_limits = []
                for i in range(len(high_mdp)):
                    if abs(high_mdp[i]) == np.inf:
                        dotted_limits.append(True)
                    else:
                        dotted_limits.append(False)

                    high_mdp[i] = 1
                    low_mdp[i] = -1


        else:
            high_mdp = None
            low_mdp = None

        observation_plot = plots.common_plots.Observations(self.observation_buffers_list,
                                                           maxs=high_mdp,
                                                           mins=low_mdp,
                                                           dotted_limits=dotted_limits)

        step_reward_plot = plots.common_plots.RewardPerStep(self.instant_reward_buffer)

        training_reward_plot = plots.common_plots.RewardPerEpisode(self.training_reward_buffer)

        episodic_len_training_plot = \
            plots.common_plots.LenOfEpisodeTraining(self.episodic_len_buffer_training)

        # create window
        self.plot_window = plots.Window(
                plot_list=[training_reward_plot, episodic_len_training_plot,
                           step_reward_plot, actions_plot, observation_plot],
                title="EnvironmentPlot",
                track_if_deactivated=[True, True, False, False, False],
                update_freq=update_freq)

        self.plot_window.show()

    def _create_new_mdp_info(self):
        return self.environment.info

    def step(self, action):
        obs, reward, absorbing, info = self.environment.step(action)

        for i in range(action.size):
            self.action_buffers_list[i].update([action[i]])

        for i in range(obs.size):
            self.observation_buffers_list[i].update([obs[i]])

        self.instant_reward_buffer.update([reward])
        #self.training_reward_buffer.update([[reward, absorbing]])
        #self.episodic_len_buffer_training.update([[1, absorbing]])

        self.plot_window.refresh()

        return obs, reward, absorbing, info

    def _get_wrapping_env_state(self):
        data = dict(plot_data={plot.name: {buffer.name: buffer.get()}
                               for plot in self.plot_window.plot_list
                               for buffer in plot.data_buffers})
        return data

    def _set_wrapping_env_state(self, data):
        normalize_data = data["plot_data"]
        for plot_name, buffer_dict in normalize_data.items():
            # could use keys to find if plots where in dicts instead of list
            for plot in self.plot_window.plot_list:
                if plot.name == plot_name:

                    # could use keys to find if plots where in dicts instead of list
                    for buffer_name, buffer_data in buffer_dict.items():
                        for buffer in plot.data_buffers:
                            if buffer.name == buffer_name:
                                buffer.set(buffer_data)

    def get_params_info(self):
        params_info = dict(PlottingEnv=self.__class__.__name__,
                           **self.environment.get_params_info())
        return params_info