import argparse

from anyrl.algos import PPO
from anyrl.envs import batched_gym_env
from anyrl.spaces import gym_spaces
from anyrl.utils.ppo import ppo_cli_args, ppo_kwargs, ppo_loop_kwargs, mpi_ppo_loop
from gym.wrappers import TimeLimit
from mazenv import HorizonEnv, parse_2d_maze
import tensorflow as tf

from re_act import ReActFF, Stack, MatMul, Bias, ReLU


def main():
    args = arg_parser().parse_args()
    env = make_env(args)
    with tf.Session() as sess:
        model = make_model(args, sess, env)
        ppo = PPO(model, **ppo_kwargs(args))
        print('Initializing model variables...')
        sess.run(tf.global_variables_initializer())
        mpi_ppo_loop(ppo, env, **ppo_loop_kwargs(args),
                     rollout_fn=lambda _: sess.run(model.reptile.apply_updates))


def make_env(args):
    maze_data = ("A.......\n" +
                 "wwwwwww.\n" +
                 "wwx.www.\n" +
                 "www.www.\n" +
                 "www.www.\n" +
                 "www.www.\n" +
                 "www.www.\n" +
                 "........")
    maze = parse_2d_maze(maze_data)

    def _make_env():
        return TimeLimit(HorizonEnv(maze, sparse_rew=True, horizon=2),
                         max_episode_steps=args.max_timesteps)

    return batched_gym_env([_make_env] * args.num_envs, sync=True)


def make_model(args, sess, env):
    return ReActFF(sess, *gym_spaces(env),
                   input_scale=1.0,
                   inner_lr=args.inner_lr,
                   outer_lr=args.outer_lr,
                   base=base_network,
                   actor=actor_network)


def base_network(inputs):
    out = tf.layers.flatten(inputs)
    out = tf.layers.dense(out, 32, activation=tf.nn.relu)
    out = tf.layers.dense(out, 32, activation=tf.nn.relu)
    return out


def actor_network(num_in, num_out):
    return Stack([
        MatMul(num_in, 32),
        Bias(32),
        ReLU(),
        MatMul(32, num_out, initializer=tf.zeros_initializer()),
        Bias(num_out),
    ])


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-envs', help='parallel environments', type=int, default=8)
    parser.add_argument('--max-timesteps', help='maximum timesteps per episode',
                        default=100, type=int)
    parser.add_argument('--inner-lr', help='online LR', default=0.01, type=float)
    parser.add_argument('--outer-lr', help='reptile LR', default=0.01, type=float)
    ppo_cli_args(parser)
    return parser


if __name__ == '__main__':
    main()
