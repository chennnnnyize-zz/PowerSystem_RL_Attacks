"""The base class for a 6-bus and 7-device :code:`gym-anm` environment."""

import datetime as dt
import numpy as np

from gym_anm import ANMEnv
from network_39_new import network
from utils import random_date


class ANM39(ANMEnv):
    """
    The base class for a 39-bus and 7-device :code:`gym-anm` environment.

    The structure of the electricity distribution network used for this
    environment is shown below:

    Slack ----------------------------
            |            |           |
          -----       -------      -----
         |     |     |       |    |     |
        House  PV  Factory  Wind  EV   DES

    This environment supports rendering (web-based) through the functions
    :py:func:`render()` and :py:func:`close()`.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, observation, K, delta_t, gamma, lamb,
                 aux_bounds=None, costs_clipping=(None, None), seed=None):

        super().__init__(network, observation, K, delta_t, gamma, lamb,
                         aux_bounds, costs_clipping, seed)

        # Rendering variables.
        self.network_specs = self.simulator.get_rendering_specs()
        self.timestep_length = dt.timedelta(minutes=int(60 * delta_t))
        self.date = None
        self.date_init = None
        self.year_count = 0
        self.skipped_frames = None
        self.render_mode = None
        self.is_rendering = False



    def step(self, action):
        obs, r, done, info = super().step(action)

        # Increment the date (for rendering).
        self.date += self.timestep_length

        # Increment the year count.
        self.year_count = (self.date - self.date_init).days // 365

        return obs, r, done, info

    def reset(self, date_init=None):
        # Save rendering setup to restore after the reset().
        render_mode = self.render_mode

        obs = super().reset()

        # Restore the rendering setup.
        self.render_mode = render_mode

        # Reset the date (for rendering).
        self.year_count = 0
        if date_init is None:
            self.date_init = random_date(self.np_random, 2020)
        else:
            self.date_init = date_init
        self.date = self.date_init

        return obs

    def reset_date(self, date_init):
        """Reset the date displayed in the visualization (and the year count)."""
        self.date_init = date_init
        self.date = date_init


