from tspmdp.modules.rnd import RandomNetworkDistillation
import tensorflow as tf
import gym
import numpy as np


class NetworkBuilder:
    def __call__(self):

        network = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128),
        ])
        return network


def episode(env: gym.Env, rnd: tf.keras.models.Model):
    observation = env.reset()
    done = False
    intrinsic = 0
    step = 0
    while not done:
        action = env.action_space.sample()
        observation, _, done, _ = env.step(action)
        reward = rnd(np.expand_dims(observation, 0))
        intrinsic += reward
        step += 1
    return intrinsic / step


def average_reward(env: gym.Env, rnd, n_episodes):
    intrinsic = 0
    for _ in range(n_episodes):
        intrinsic += episode(env, rnd)
    return intrinsic / n_episodes


def one_test():
    env = gym.make("CartPole-v0")
    rnd = RandomNetworkDistillation(NetworkBuilder())
    n_episodes = 10
    ave = average_reward(env, rnd, n_episodes)
    ave2 = average_reward(env, rnd, n_episodes)


def test_rnd():
    for _ in range(10):
        one_test()


test_rnd()
