import pickle
from mushroom_rl.environments import Environment


class WrappingEnvironment(Environment):
    """
    Implements a common interface for wrapping environments, these
        can be used to change/add functionality to general mushroom
        environments while maintaining the same general interface to
        the outside.

    """
    def __init__(self, env_class, env_kwargs=None):
        """
        Constructor.

        Args:
            env_class (object): Environment class to be used.
            env_kwargs (dict): Parameters to instantiate environment
                class with.
        """
        self.environment = env_class(**(dict() if env_kwargs is None else env_kwargs))
        super(WrappingEnvironment, self).__init__(self._create_new_mdp_info())

    def _create_new_mdp_info(self):
        """
        Change mdp info to reflect new environment with
            actions/observation values(as seen from outside).
            By default returns same info as self.environment.

        Returns:
            A new mdp info object.

        """
        return self.environment.info

    @property
    def env(self):
        """
        Property which allows wrapper envs to access base env and their
            variables when needed, even when using nested
            _wrapping_envs(calls recursively).

        Returns:
            The child environment if it is a wrapping environment or itself
            if its the base environment.
        """
        if hasattr(self.environment, "env"):
            return self.environment.env
        return self

    def render(self):
        """
        Default environment render, can be overwritten to
            change it's functionality.
        """
        self.environment.render()

    def stop(self):
        """
        Default environment stop, can be overwritten to
            change it's functionality.
        """
        self.environment.stop()

    def reset(self, state=None):
        """
        Default environment reset, can be overwritten to
            change it's functionality.
        """
        obs = self.environment.reset(state)
        return obs

    def step(self, action):
        """
        Default environment step, can be overwritten to
            change it's functionality.
        """
        return self.environment.step(action)

    def _get_wrapping_env_state(self):
        """
        To be overwritten by the extending env, allows passing current
            env parameters to be saved. By default passes nothing.
        """
        return dict()

    def _set_wrapping_env_state(self, data):
        """
        To be overwritten by the extending env, allows setting additional
            parameters to the current env. By default sets nothing.
        """
        pass

    def get_all_wrapping_envs_state(self):
        """
        Gets relevant state from all the wrapping environments inside
            and itself and returns it in a structured way. Can be used to
            save a state which can later be loaded

        Returns
            A Structured Dictionary, which contains its relevant data
                and all the child wrapping environments data.

        """
        data = self._get_wrapping_env_state()
        # if still has wrapping env inside which might need to save state
        if hasattr(self.environment, "wrapping_env_get_state"):
            data = dict(**data, **self.environment.wrapping_env_get_state())
        return data

    def set_all_wrapping_envs_state(self, data):
        """
        Gets relevant state from all the wrapping environments inside
            and itself and returns it in a structured way. Can be used to
            save a state which can later be loaded

        Args
            data (dict): A Structured Dictionary, which contains relevant
                data for the wrapping environment and all its child's data
                to be set.

        """
        self._set_wrapping_env_state(data)
        # if still has wrapping env inside which might need to load state
        if hasattr(self.environment, "wrapping_env_set_state"):
            self.environment.wrapping_env_set_state(data)

    def wrapping_env_save_state(self, path):
        """
        Saves the state info which can be used to recover the
            wrapping env and all its child's state as a pickled data dump.

        Args
            path (string): Path to save a data dump.

        """
        data = self.get_all_wrapping_envs_state()
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=3)

    def wrapping_env_load_state(self, path):
        """
        Loads the state info which can be used to recover the
            wrapping env and all its child's state from a pickled
            data dump path.

        Args
            path (string): Path from where to load a data dump from.

        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.set_all_wrapping_envs_state(data)

    def get_params_info(self):
        return dict()