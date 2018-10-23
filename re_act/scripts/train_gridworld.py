import argparse

from anyrl.algos import PPO
from anyrl.envs import batched_gym_env
from anyrl.spaces import gym_spaces
from anyrl.utils.ppo import ppo_cli_args, ppo_kwargs, ppo_loop_kwargs, mpi_ppo_loop
from gym.wrappers import TimeLimit
from mazenv import HorizonEnv, parse_2d_maze
import tensorflow as tf

from re_act import ReActFF


def main():
    args = arg_parser().parse_args()

    maze_data = ("A......\n" +
                 "wwwwww.\n" +
                 "wwwwww.\n" +
                 "wwwwww.\n" +
                 "wwwwww.\n" +
                 "x......")
    maze = parse_2d_maze(maze_data)

    def make_env():
        return TimeLimit(HorizonEnv(maze, horizon=2), max_episode_steps=args.max_timesteps)
    env = batched_gym_env([make_env] * args.num_envs, sync=True)

    with tf.Session() as sess:
        model = ReActFF(sess, *gym_spaces(env), input_scale=1.0, step_size=args.lr,
                        base=base_network)
        ppo = PPO(model, **ppo_kwargs(args))
        print('Initializing model variables...')
        sess.run(tf.global_variables_initializer())
        mpi_ppo_loop(ppo, env, **ppo_loop_kwargs(args))


def base_network(inputs):
    out = tf.layers.flatten(inputs)
    out = tf.layers.dense(out, 32, activation=tf.nn.relu)
    out = tf.layers.dense(out, 32, activation=tf.nn.relu)
    return out


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-envs', help='parallel environments', type=int, default=8)
    parser.add_argument('--max-timesteps', help='maximum timesteps per episode',
                        default=100, type=int)
    parser.add_argument('--lr', help='online LR', default=0.01, type=float)
    ppo_cli_args(parser)
    return parser


if __name__ == '__main__':
    main()
