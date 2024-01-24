import gymnasium as gym
from cartpole import CartPoleEnv
# env = gym.make("LunarLander-v2", render_mode="human")
env = CartPoleEnv(render_mode='pygame')
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    env.render()
env.close()