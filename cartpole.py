"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union
# Cope's personal Python package, it's where SimpleGym is located
from Cope.gym import SimpleGym
# A lovely library that adds local variables to error traces
from traceback_with_variables import activate_by_import
import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

try:
    import pygame
    from pygame import gfxdraw
except ImportError:
    from warnings import warn
    warn('pygame not installed, this should be running on the robot')


class CartPoleEnv(SimpleGym):
    """ ### Description
        This is the enviorment for our self-balancing robot, Lenny. It will have 2 modes: pygame and
        hardware, which correspond to the render_modes

        #### Action Space

        The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
        of the fixed force the cart is pushed with.

        | Num | Action                 |
        |-----|------------------------|
        | 0   | Push cart to the left  |
        | 1   | Push cart to the right |

        **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
        the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

        ### Observation Space

        The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

        | Num | Observation           | Min                 | Max               |
        |-----|-----------------------|---------------------|-------------------|
        | 0   | Cart Position         | -4.8                | 4.8               |
        | 1   | Cart Velocity         | -Inf                | Inf               |
        | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
        | 3   | Pole Angular Velocity | -Inf                | Inf               |

        **Note:** While the ranges above denote the possible values for observation space of each element,
            it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
        -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
        if the cart leaves the `(-2.4, 2.4)` range.
        -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
        if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

        ### Rewards

        Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
        including the termination step, is allotted. The threshold for rewards is 475 for v1.

        ### Starting State

        All observations are assigned a uniformly random value in `(-0.05, 0.05)`

        ### Episode End

        The episode ends if any one of the following occurs:

        1. Termination: Pole Angle is greater than ±12°
        2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
        3. Truncation: Episode length is greater than 500 (200 for v0)

        ### Arguments

        ```
        gym.make('CartPole-v1')
        ```

        No additional arguments are currently supported.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, max_steps=-1, screen_size=500, **kwargs)
        # The default is dark grey
        self.background_color = (255, 255, 255)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None
        self.steps_beyond_terminated = None

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {}

    def _get_terminated(self):
        x, x_dot, theta, theta_dot = self.state

        return bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

    def _get_reward(self):
        if not self._get_terminated():
            return 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            return 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            return 0.0

    def step(self, action) -> ('obs', 'reward', 'terminated', 'truncated', 'info'):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        return super().step(action)

    def reset(self, *args, **kwargs) -> ('obs', 'info'):
        rtn = super().reset(*args, **kwargs)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            None, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        return rtn

    def render_pygame(self):
        world_width = self.x_threshold * 2
        scale = self.size / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return

        x = self.state

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.size / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.size, carty, (0, 0, 0))

        # It's upside down for some reason
        self.surf = pygame.transform.flip(self.surf, False, True)


# TODO: Simple gym should have a default function for _get_info()
# TODO: Simple gym should handle initialization better
# TODO: Simple gym should handle FPS and pause and step itself
# TODO: Simple gym should have a custom-render method (call render_{name} with getattr())
