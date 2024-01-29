from cartpole import CartPoleEnv
import time
import numpy as np

env = CartPoleEnv(render_mode='pygame')
obs, info = env.reset()
FPS = 20
i = 0
# default_action = np.array([1.])
default_action = .5

while i < 1000:
    env.render()
    if env.pause:
        i -= 1
        if env.increment:
            env.increment = False
            obs, reward, terminated, _, info = env.step(default_action)
            # env.step(np.array([ant.wander() for ant in ants]))
        continue
    obs, reward, terminated, _, info = env.step(default_action)
    if terminated:
        obs, info = env.reset()
    time.sleep(1/FPS)
env.close()



# obs, info = env.reset()
# for _ in range(1000):
#     env.render()
#     obs, reward, terminated, _, _ = env.step(1)
#     if terminated:
#         obs, info = env.reset()
# env.close()
