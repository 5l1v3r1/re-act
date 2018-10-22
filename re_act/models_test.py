from random import random

from anyrl.envs import batched_gym_env
from anyrl.spaces import gym_spaces
from anyrl.rollouts import TruncatedRoller
import gym
import numpy as np
import tensorflow as tf

from .models import ReActFF
from .network import MatMul


def test_output_consistency():
    """
    Test that outputs from stepping are consistent with
    the batched model outputs.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            env = batched_gym_env([DummyEnv] * 16, sync=True)
            model = ReActFF(sess, *gym_spaces(env),
                            input_scale=1.0,
                            input_dtype=tf.float32,
                            base=lambda x: tf.layers.dense(x, 12),
                            actor=MatMul,
                            critic=lambda x: MatMul(x, 1))
            sess.run(tf.global_variables_initializer())
            roller = TruncatedRoller(env, model, 8)
            for _ in range(10):
                rollouts = roller.rollouts()
                actor_out, critic_out = model.batch_outputs()
                info = next(model.batches(rollouts))
                actor_out, critic_out = sess.run(model.batch_outputs(), feed_dict=info['feed_dict'])
                idxs = enumerate(zip(info['rollout_idxs'], info['timestep_idxs']))
                for i, (rollout_idx, timestep_idx) in idxs:
                    outs = rollouts[rollout_idx].model_outs[timestep_idx]
                    assert np.allclose(actor_out[i], outs['action_params'][0])
                    assert np.allclose(critic_out[i], outs['values'][0])


class DummyEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=1.0, dtype='float32', shape=(15,))

    def reset(self):
        return np.random.normal(size=(15,)).astype('float32')

    def step(self, action):
        return self.reset(), 0.5, random() < 0.1, {}
