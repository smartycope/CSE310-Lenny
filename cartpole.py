"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union
# Cope's personal Python package, it's where SimpleGym is located
from Cope.gym import SimpleGym
import Cope
assert Cope.__version__ >= '2.4.0', "SimpleGym requires Cope>=2.4.0. You can update Cope with `pip install --upgrade Cope`"
# A lovely library that adds local variables to error traces
# from traceback_with_variables import activate_by_import
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

    def __init__(self, *args, force=10., start_angle_range=.05, bounds_size=2.4, **kwargs):
        super().__init__(
            *args,
            max_steps=-1,
            screen_size=500,
            background_color=(255, 255, 255),
            name='Lenny Simulation',
            show_vars={'FPS': 'fps'},
            assert_valid_action=False,
            verbose=False,
            **kwargs
        )
        self._show_help = True

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = bounds_size

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds.
        high = np.array(
            [
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Pole angle and pole angular velocity
        self.state = None
        # cart x pos and cart velocity
        self.pos_info = (0, 0)
        self.start_angle_range = start_angle_range
        # Whether the angle was previously positive or nto
        self.prev_pos = None

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return self.pos_info

    def _get_terminated(self):
        x, x_vel = self.pos_info
        theta, theta_vel = self.state

        return (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

    def _get_reward(self):
        # Falling hurts -- this helped a lot
        if self._get_terminated():
            return -100

        x, x_vel = self.pos_info
        theta, theta_vel = self.state

        # Base starting score
        score = -50

        # The longer the episode has gone on, the better
        score += self.steps / 3

        # *really* incentivize really long episodes
        # if self.steps > 100:
            # score += self.steps

        # We don't like the stick being at an angle
        score -= abs(theta) * 300

        # Stay near the center, dang it
        score -= abs(x) * 100

        # We like it if the stick is moving slowly
        # score -= abs(theta_vel) * 10

        # If we've just gone from +angle to -angle, or vice versa, that's good, that means that we've
        # either just corrected ourselves, or we're balancing straight up
        is_pos = theta > 0
        if is_pos != self.prev_pos:
            score += 200

        return score
        # Try bounding the reward maybe??
        # return min(max(-100, round(score)), 300)

    def _step(self, action):
        x, x_vel = self.pos_info
        theta, theta_vel = self.state
        force = self.force_mag * action[0]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        self.prev_pos = theta > 0

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_vel**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_vel
            x_vel = x_vel + self.tau * xacc
            theta = theta + self.tau * theta_vel
            theta_vel = theta_vel + self.tau * thetaacc
        else:  # semi-implicit euler
            x_vel = x_vel + self.tau * xacc
            x = x + self.tau * x_vel
            theta_vel = theta_vel + self.tau * thetaacc
            theta = theta + self.tau * theta_vel

        self.pos_info = (x, x_vel)
        if x == np.nan or x_vel == np.nan:
            self.pos_info = (0, 0)
        self.state = (theta, theta_vel)

    def _reset(self, seed=None, options=None):
        self.state = self.np_random.uniform(low=-self.start_angle_range, high=self.start_angle_range, size=(2,))
        # self.state[1] = 0
        self.pos_info = (0, 0)
        self.prev_pos = self.state[0] > 0

    def render_pygame(self):
        if self.state is None or self.pos_info is None: return

        world_width = self.x_threshold * 2
        scale = self.size / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        # polelen = 10 * polewidth
        cartwidth = 50.0
        cartheight = 30.0

        x, x_vel = self.pos_info
        theta, theta_vel = self.state

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x * scale + self.size / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        # Specify the bounds of the cart
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        # Apply the x offset to the bounds
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
            coord = pygame.math.Vector2(coord).rotate_rad(-theta)
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
