"""
Reptile inner-loop updates for heads.
"""

import tensorflow as tf


class Reptile:
    """
    A small Reptile-like graph for meta-learning better
    adaptive head parameters.
    """

    def __init__(self, head, inner_lr, outer_lr):
        self.head = head
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr

        with tf.variable_scope(None, default_name='reptile'):
            self._num_updates = tf.get_variable('num_updates', dtype=tf.float32, trainable=False,
                                                shape=())
            self._update_vars = []
            for i, v in enumerate(head.variables):
                update_var = tf.get_variable('update_%d' % i,
                                             dtype=v.dtype,
                                             shape=v.get_shape(),
                                             initializer=tf.zeros_initializer(),
                                             trainable=False)
                self._update_vars.append(update_var)
            assigns = []
            for update, v in zip(self._update_vars, head.variables):
                assigns.append(tf.assign_add(v, self.outer_lr * update / self._num_updates))
            with tf.control_dependencies(assigns.copy()):
                assigns.append(tf.assign(self._num_updates, tf.zeros_like(self._num_updates)))
                for update in self._update_vars:
                    assigns.append(tf.assign(update, tf.zeros_like(update)))
            self.apply_updates = tf.group(*assigns)

    def updates(self, states, log_probs, redundant=False):
        updates = []
        for state, grad in zip(states, tf.gradients(log_probs, states)):
            update = grad * self.inner_lr if grad is not None else tf.zeros_like(state)
            updates.append(update)
        deps = []
        if not redundant:
            deps.append(self._add_updates(updates))
        with tf.control_dependencies(deps):
            return tuple(state + update for state, update in zip(states, updates))

    def _add_updates(self, updates):
        with tf.variable_scope(None, default_name='reptile_add'):
            assigns = []
            for update_var, update in zip(self._update_vars, updates):
                assigns.append(tf.assign_add(update_var, tf.reduce_sum(update, axis=0)))
            assigns.append(tf.assign_add(self._num_updates,
                                         tf.cast(tf.shape(updates[0])[0], tf.float32)))
            return tf.group(*assigns)
