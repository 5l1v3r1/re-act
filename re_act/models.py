"""
RL models for implementing re-act.
"""


from anyrl.models import TFActorCritic
from anyrl.models.util import impala_cnn, mini_batches, mix_init_states
from anyrl.rollouts.util import inject_state
import numpy as np
import tensorflow as tf

from .network import head_fc
from .reptile import Reptile


class ReActFF(TFActorCritic):
    """
    A base-class for feed-forward ReAct models.
    """

    def __init__(self,
                 sess,
                 act_dist,
                 obs_vect,
                 input_scale=1 / 0xff,
                 input_dtype=tf.uint8,
                 inner_lr=0.01,
                 outer_lr=0.01,
                 base=impala_cnn,
                 actor=head_fc,
                 critic=lambda ins: head_fc(ins, 1)):
        super().__init__(sess, act_dist, obs_vect)
        self.obs_ph = tf.placeholder(input_dtype, shape=(None,) + obs_vect.out_shape, name='obs')
        obs = tf.cast(self.obs_ph, tf.float32) * input_scale
        with tf.variable_scope(None, default_name='base'):
            self.base_out = base(obs)
        with tf.variable_scope(None, default_name='actor'):
            num_outs = int(np.prod(act_dist.param_shape))
            self.actor = actor(self.base_out.get_shape()[-1].value, num_outs)
        with tf.variable_scope(None, default_name='critic'):
            self.critic = critic(self.base_out.get_shape()[-1].value)
        self.reptile = Reptile(self.actor, inner_lr, outer_lr)
        self.state_phs = tuple(tf.placeholder(v.dtype.base_dtype,
                                              shape=(None,) + tuple(x.value for x in v.get_shape()),
                                              name='state_%d' % i)
                               for i, v in enumerate(self.actor.variables))
        self.actor_out = self.actor.apply(self.base_out, self.state_phs)
        self.critic_out = self.critic.apply_init(self.base_out)[:, 0]
        self.new_state_graph = self._make_new_state_graph()
        self.sequence_graph = self._make_sequence_graph()

    @property
    def stateful(self):
        return True

    def start_state(self, batch_size):
        values = self.session.run(self.actor.variables)
        return tuple(np.array([x] * batch_size) for x in values)

    def step(self, observations, states):
        feed_dict = {self.obs_ph: self.obs_vectorizer.to_vecs(observations)}
        feed_dict.update(dict(zip(self.state_phs, states)))
        act_params, vals, base_out = self.session.run((self.actor_out, self.critic_out,
                                                       self.base_out), feed_dict)
        actions = self.action_dist.sample(act_params)
        del feed_dict[self.obs_ph]
        feed_dict.update({
            self.base_out: base_out,
            self.new_state_graph['action_ph']: self.action_dist.to_vecs(actions),
        })
        states = self.session.run(self.new_state_graph['new_state'], feed_dict=feed_dict)
        return {
            'action_params': act_params,
            'actions': actions,
            'states': states,
            'values': vals
        }

    def batch_outputs(self):
        return self.sequence_graph['actor_out'], self.sequence_graph['critic_out']

    def batches(self, rollouts, batch_size=None):
        sizes = [r.num_steps for r in rollouts]
        for rollout_indices in mini_batches(sizes, batch_size):
            batch = [rollouts[i] for i in rollout_indices]
            max_len = max([r.num_steps for r in batch])
            gather_indices = np.zeros((max_len, len(batch)), dtype='int32')
            timestep_idxs = []
            rollout_idxs = []
            masks = []
            obses = []
            actions = []
            for timestep in range(max_len):
                for i, rollout in enumerate(batch):
                    if timestep < rollout.num_steps:
                        gather_indices[timestep, i] = len(obses)
                        obses.append(rollout.observations[timestep])
                        actions.append(rollout.model_outs[timestep]['actions'][0])
                        timestep_idxs.append(timestep)
                        rollout_idxs.append(rollout_indices[i])
                        masks.append(True)
                    else:
                        masks.append(False)
            feed_dict = {
                self.obs_ph: self.obs_vectorizer.to_vecs(obses),
                self.sequence_graph['action_ph']: self.action_dist.to_vecs(actions),
                self.sequence_graph['index_ph']: gather_indices,
                self.sequence_graph['mask_ph']: masks,
                self.sequence_graph['news_ph']: [not r.trunc_start for r in batch],
            }
            self._add_first_states(feed_dict, batch)
            yield {
                'rollout_idxs': rollout_idxs,
                'timestep_idxs': timestep_idxs,
                'feed_dict': feed_dict
            }

    def _add_first_states(self, feed_dict, batch):
        result = self.start_state(len(batch))
        for i, rollout in enumerate(batch):
            inject_state(result, rollout.start_state, i)
        feed_dict.update(dict(zip(self.state_phs, result)))

    def _make_new_state_graph(self):
        action_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_dist.out_shape)
        log_probs = self.action_dist.log_prob(self.actor_out, action_ph)
        new_state = self.reptile.updates(self.state_phs, log_probs)
        return {
            'action_ph': action_ph,
            'new_state': new_state,
        }

    def _make_sequence_graph(self):
        action_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_dist.out_shape,
                                   name='actions')
        index_ph = tf.placeholder(tf.int32, shape=(None, None), name='index')
        action_seq = tf.gather(action_ph, index_ph)
        base_seq = tf.gather(self.base_out, index_ph)

        shape = tf.shape(action_seq)
        seq_len = shape[0]
        batch_size = shape[1]
        news_ph = tf.placeholder(tf.bool, shape=(None,), name='news')
        init_state = mix_init_states(news_ph, self.actor.variables, self.state_phs)

        def step_fn(states, actor_arr, critic_arr, i):
            sub_base = base_seq[i]
            outputs = self.actor.apply(sub_base, states)
            values = self.critic.apply_init(sub_base)[:, 0]
            log_probs = self.action_dist.log_prob(outputs, action_seq[i])
            new_state = self.reptile.updates(states, log_probs, redundant=True)
            return new_state, actor_arr.write(i, outputs), critic_arr.write(i, values), i + 1

        _, actor_arr, critic_arr, _ = tf.while_loop(lambda a, b, c, d: d < seq_len,
                                                    step_fn,
                                                    (init_state,
                                                     tf.TensorArray(tf.float32, size=seq_len),
                                                     tf.TensorArray(tf.float32, size=seq_len),
                                                     tf.constant(0, dtype=tf.int32)))

        mask_ph = tf.placeholder(tf.bool, shape=(None,), name='mask')

        return {
            'action_ph': action_ph,
            'index_ph': index_ph,
            'mask_ph': mask_ph,
            'news_ph': news_ph,
            'batch_size': batch_size,
            'actor_out': tf.boolean_mask(actor_arr.concat(), mask_ph),
            'critic_out': tf.boolean_mask(critic_arr.concat(), mask_ph),
        }
