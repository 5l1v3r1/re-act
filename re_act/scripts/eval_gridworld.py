from anyrl.rollouts import TruncatedRoller
import tensorflow as tf

from re_act.scripts.train_gridworld import arg_parser, make_env, make_model


def main():
    args = arg_parser().parse_args()
    env = make_env(args)
    with tf.Session() as sess:
        model = make_model(args, sess, env)
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
