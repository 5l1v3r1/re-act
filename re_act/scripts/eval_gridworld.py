import argparse

from anyrl.algos import PPO
from anyrl.envs import batched_gym_env
from anyrl.rollouts import TruncatedRoller
from anyrl.spaces import gym_spaces
from anyrl.utils.ppo import ppo_cli_args, ppo_kwargs, ppo_loop_kwargs, mpi_ppo_loop
from gym.wrappers import TimeLimit
from mazenv import HorizonEnv, parse_2d_maze
import tensorflow as tf

from re_act import ReActFF, Stack, MatMul, Bias, ReLU
from re_act.scripts.train_gridworld import arg_parser, base_network, actor_network


def main():
    args = arg_parser().parse_args()

    maze_data = ("A.......\n" +
                 "wwwwwww.\n" +
                 "wwx.www.\n" +
                 "www.www.\n" +
                 "www.www.\n" +
                 "www.www.\n" +
                 "www.www.\n" +
                 "........")
    maze = parse_2d_maze(maze_data)

    def make_env():
        return TimeLimit(HorizonEnv(maze, sparse_rew=True, horizon=2),
                         max_episode_steps=args.max_timesteps)
    env = batched_gym_env([make_env] * args.num_envs, sync=True)

    with tf.Session() as sess:
        model = ReActFF(sess, *gym_spaces(env),
                        input_scale=1.0,
                        inner_lr=args.inner_lr,
                        outer_lr=args.outer_lr,
                        base=base_network,
                        actor=actor_network)
        print('Initializing model variables...')
        sess.run(tf.global_variables_initializer())
        roller = TruncatedRoller(env, model, 128)
        total, good = 0, 0
        while True:
            r = [r for r in roller.rollouts() if not r.trunc_end]
            sess.run(model.reptile.apply_updates)
            total += len(r)
            good += len([x for x in r if x.total_reward > 0])
            print('got %f (%d out of %d)' % (good / total, good, total))

if __name__ == '__main__':
    main()
