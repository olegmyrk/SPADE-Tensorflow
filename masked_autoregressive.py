import tensorflow as tf
import tensorflow_probability as tfp
#from tensorflow_probability.python.math.numeric import clip_by_value_preserve_gradient
tfb = tfp.bijectors

def unmasked_dense(inputs,
                 units,
                 kernel_initializer=None,
                 activation=None,
                 reuse=None,
                 name=None,
                 *args,  # pylint: disable=keyword-arg-before-vararg
                 **kwargs):
  input_depth = tf.dimension_value(inputs.shape.with_rank_at_least(1)[-1])

  if kernel_initializer is None:
    kernel_initializer = tf.glorot_normal_initializer()

  with tf.name_scope(name or 'unmasked_dense'):
    layer = tf.layers.Dense(
        units,
        kernel_initializer=kernel_initializer,
        activation=activation,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _scope=name,
        _reuse=reuse,
        *args,  # pylint: disable=keyword-arg-before-vararg
        **kwargs)
    return layer.apply(inputs)

def conditional_masked_dense(context,
             inputs,
             units,
             num_blocks=None,
             exclusive=False,
             kernel_initializer=None,
             activation=None,
             reuse=None,
             name=None,
             *args,  # pylint: disable=keyword-arg-before-vararg
             **kwargs): 
    masked_dense = tfb.masked_dense(
                inputs=inputs,
                units=units,
                num_blocks=num_blocks,
                exclusive=exclusive,
                kernel_initializer=kernel_initializer,
                activation=None,
                reuse=reuse,
                name=name + "/masked_dense",
                *args,  # pylint: disable=keyword-arg-before-vararg
                **kwargs)
    dense = unmasked_dense(
                inputs=context,
                units=units,
                kernel_initializer=kernel_initializer,
                activation=None,
                reuse=reuse,
                name=name + "/dense"
                )
    result = (masked_dense + dense)
    if activation:
        result = activation(result)
    return result

def conditional_masked_autoregressive_template(
                                           context,
                                           hidden_layers,
                                           shift_only=False,
                                           activation=tf.nn.relu,
                                           log_scale_min_clip=-5.,
                                           log_scale_max_clip=3.,
                                           log_scale_clip_gradient=False,
                                           name=None,
                                           *args,  # pylint: disable=keyword-arg-before-vararg
                                           **kwargs):

  with tf.name_scope(name):
    def _fn(x):
      input_depth = tf.dimension_value(x.shape.with_rank_at_least(1)[-1])
      input_shape = tf.shape(x)
      for i, units in enumerate(hidden_layers):
        x = conditional_masked_dense(
                context=context,
                inputs=x,
                units=units,
                num_blocks=input_depth,
                exclusive=True if i == 0 else False,
                activation=activation,
                name=name + "/hidden_" + str(i),
                *args,  # pylint: disable=keyword-arg-before-vararg
                **kwargs)
      x = conditional_masked_dense(
              context=context,
              inputs=x,
              units=(1 if shift_only else 2) * input_depth,
              num_blocks=input_depth,
              activation=None,
              name=name + "/linear",
              *args,  # pylint: disable=keyword-arg-before-vararg
              **kwargs)
      if shift_only:
        x = tf.reshape(x, shape=input_shape)
        return x, None
      x = tf.reshape(x, shape=tf.concat([input_shape, [2]], axis=0))
      shift, log_scale = tf.unstack(x, num=2, axis=-1)
      which_clip = (
          tf.clip_by_value
          if log_scale_clip_gradient else _clip_by_value_preserve_grad)
      log_scale = which_clip(log_scale, log_scale_min_clip, log_scale_max_clip)
      return shift, log_scale

    return tf.make_template(name, _fn)

def _clip_by_value_preserve_grad(x, clip_value_min, clip_value_max, name=None):
  """Clips input while leaving gradient unaltered."""
  with tf.name_scope(name, "clip_by_value_preserve_grad",
                     [x, clip_value_min, clip_value_max]):
    clip_x = tf.clip_by_value(x, clip_value_min, clip_value_max)
    return x + tf.stop_gradient(clip_x - x)

