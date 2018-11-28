"""Sequence-to-sequence model for human motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn import _transpose_batch_time

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import rnn_cell_extensions # my extensions of the tf repos
import data_utils

class Seq2SeqModel(object):
  """Sequence-to-sequence model for human motion prediction"""

  def __init__(self,
               architecture,
               max_seq_len,
               human_size,
               rnn_size, # hidden recurrent layer size
               num_layers,
               max_gradient_norm,
               stddev,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               summaries_dir,
               loss_to_use,
               number_of_actions,
               one_hot=True,
               residual_velocities=False,
               dtype=tf.float32):
    """Create the model.

    Args:
      architecture: [basic, tied] whether to tie the decoder and decoder.
      source_seq_len: lenght of the input sequence.
      #target_seq_len: lenght of the target sequence.
      rnn_size: number of units in the rnn.
      num_layers: number of rnns to stack.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      summaries_dir: where to log progress for tensorboard.
      loss_to_use: [supervised, sampling_based]. Whether to use ground truth in
        each timestep to compute the loss after decoding, or to feed back the
        prediction from the previous time-step.
      number_of_actions: number of classes we have.
      one_hot: whether to use one_hot encoding during train/test (sup models).
      residual_velocities: whether to use a residual connection that models velocities.
      dtype: the data type to use to store internal variables.
    """

    self.HUMAN_SIZE = human_size
    self.input_size = self.HUMAN_SIZE + number_of_actions if one_hot else self.HUMAN_SIZE

    print( "One hot is ", one_hot )
    print( "Input size is %d" % self.input_size )

    # Summary writers for train and test runs
    self.train_writer = tf.summary.FileWriter(os.path.normpath(os.path.join( summaries_dir, 'train')))
    self.test_writer  = tf.summary.FileWriter(os.path.normpath(os.path.join( summaries_dir, 'test')))

    self.max_seq_len = max_seq_len
    self.rnn_size = rnn_size
    self.batch_size = batch_size
    self.learning_rate = tf.Variable( float(learning_rate), trainable=False, dtype=dtype )
    self.learning_rate_decay_op = self.learning_rate.assign( self.learning_rate * learning_rate_decay_factor )
    self.global_step = tf.Variable(0, trainable=False)

    # === Create the RNN that will keep the state ===
    print('rnn_size = {0}'.format( rnn_size ))
    cell = tf.contrib.rnn.GRUCell( self.rnn_size )

    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell( [tf.contrib.rnn.GRUCell(self.rnn_size) for _ in range(num_layers)] )

    # === Transform the inputs ===
    with tf.name_scope("inputs_gts"):

      inputs  = tf.placeholder(dtype, shape=[None, self.max_seq_len+1, self.input_size], name="inputs")
      gts     = tf.placeholder(dtype, shape=[None, self.max_seq_len, self.input_size], name="gts")
      seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
      
      self.inputs  = inputs
      self.gts     = gts
      self.seq_len = seq_len
      '''
      inputs = tf.transpose(inputs, [1, 0, 2])
      gts    = tf.transpose(gts, [1, 0, 2])

      inputs = tf.reshape(inputs, [-1, self.input_size])
      gts    = tf.reshape(gts,    [-1, self.input_size])

      inputs = tf.split(inputs, self.max_seq_len, axis=0)
      gts    = tf.split(gts,    self.max_seq_len, axis=0)
      '''
      inputs = _transpose_batch_time(inputs)
      gts    = _transpose_batch_time(gts)

    # === Add space decoder ===
    cell = rnn_cell_extensions.LinearSpaceDecoderWrapper( cell, self.input_size )

    # Finally, wrap everything in a residual layer if we want to model velocities
    if residual_velocities:
      cell = rnn_cell_extensions.ResidualWrapper( cell )

    # Store the outputs here
    outputs  = []

    self.stddev = stddev
    def addGN(inputs):
      noise = tf.random_normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev, dtype=tf.float32)
      return inputs + noise

    self.is_training = tf.placeholder(dtype=tf.bool)
    
    # Build the RNN
    if architecture == "basic":
      cell_init_state = tf.Variable(np.zeros([1,cell.state_size]),trainable=True, dtype=tf.float32)
      init_input = tf.Variable(np.zeros([63]), trainable=True, dtype=tf.float32)
      output_ta = tf.TensorArray(size=self.max_seq_len, dtype=tf.float32)
      def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output
        if cell_output is None:
          #next_cell_state = cell.zero_state(self.batch_size, tf.float32)
          next_cell_state = tf.tile(cell_init_state, [tf.shape(inputs[0])[0],1])
          next_input = tf.cond(self.is_training,
                               lambda: tf.concat([tf.tile(tf.expand_dims(init_input,0),[tf.shape(inputs[0])[0],1]),
                                                  addGN(inputs[time])], axis=1),
                               lambda: tf.concat([tf.tile(tf.expand_dims(init_input,0),[tf.shape(inputs[0])[0],1]),
                                                  inputs[time]], axis=1))
          next_loop_state = output_ta
        else:
          next_cell_state = cell_state
          next_input = tf.cond(self.is_training,
                               lambda: tf.concat([cell_output, addGN(inputs[time])], axis=1),
                               lambda: tf.concat([cell_output, inputs[time]], axis=1))

          next_loop_state = loop_state.write(time-1, cell_output)

        finished = (time > self.max_seq_len-1)
        #finished = False
        return (finished, next_input, next_cell_state, emit_output, next_loop_state)
          
      
      # Basic RNN does not have a loop function in its API, so copying here.
      with vs.variable_scope("raw_rnn"):
        _, _, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn)
        #outputs = _transpose_batch_time(loop_state_ta.stack())
        outputs = loop_state_ta.stack()
        
    self.outputs = outputs
    mask1 = tf.tile(tf.expand_dims(tf.transpose(tf.sequence_mask(
      self.seq_len,
      dtype=tf.float32,
      maxlen=self.max_seq_len)),-1), [1,1,self.input_size])
    mask2 = tf.tile(tf.expand_dims(tf.transpose(tf.sequence_mask(
      self.seq_len-1,
      dtype=tf.float32,
      maxlen=self.max_seq_len-1)),-1), [1,1,self.input_size])
    with tf.name_scope("loss_pos"):
      loss_pos = tf.reduce_mean(tf.square(tf.subtract(tf.multiply(outputs,mask1),
                                                         tf.multiply(gts,mask1))))
    with tf.name_scope("loss_smooth"):
      loss_smooth = tf.reduce_mean(tf.square(
        tf.multiply(tf.subtract(outputs[1:],outputs[:-1]),mask2)))
    #self.loss         = tf.add(loss_pos, loss_smooth*1000)
    self.loss = loss_pos
    self.loss_summary = tf.summary.scalar('loss/loss', self.loss)

    self.loss_each_data = tf.reduce_mean(tf.square(tf.subtract(tf.multiply(gts,mask1),
                                                               tf.multiply(outputs,mask1))),
                                         axis=[0,2]) \
                          + tf.reduce_mean(tf.square(tf.multiply(tf.subtract(
                          outputs[1:], outputs[:-1]),mask2)),axis=[0,2])
    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Update all the trainable parameters
    gradients = tf.gradients( self.loss, params )

    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    self.gradient_norms = norm
    self.updates = opt.apply_gradients(
      zip(clipped_gradients, params), global_step=self.global_step)

    self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

    self.saver = tf.train.Saver( tf.global_variables(), max_to_keep=10 )

  def step(self, session, inputs, gts, seq_len,
           forward_only, sample=False ):
    """Run a step of the model feeding the given inputs.

    Args
      session: tensorflow session to use.
      encoder_inputs: list of numpy vectors to feed as encoder inputs.
      decoder_inputs: list of numpy vectors to feed as decoder inputs.
      decoder_outputs: list of numpy vectors that are the expected decoder outputs.
      forward_only: whether to do the backward step or only forward.
      sample: True if you want to evaluate using the sequences of SRNN
    Returns
      A triple consisting of gradient norm (or None if we did not do backward),
      mean squared error, and the outputs.
    Raises
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """

    # Output feed: depends on whether we do a backward step or not.
    if not sample:
      if not forward_only:

        input_feed = {self.inputs: inputs,
                      self.gts: gts,
                      self.seq_len: seq_len,
                      self.is_training: True}
        # Training step
        output_feed = [self.updates,         # Update Op that does SGD.
                       self.gradient_norms,  # Gradient norm.
                       self.loss,
                       self.loss_summary,
                       self.learning_rate_summary]

        outputs = session.run( output_feed, input_feed )
        return outputs[1], outputs[2], outputs[3], outputs[4]  # Gradient norm, loss, summaries

      else:
        input_feed = {self.inputs: inputs,
                      self.gts: gts,
                      self.seq_len: seq_len,
                      self.is_training: False}
        # Validation step, not on SRNN's seeds
        output_feed = [self.loss, # Loss for this batch.
                       self.loss_summary]

        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1]# No gradient norm
    else:
      # Validation on SRNN's seeds
      input_feed = {self.inputs: inputs,
                    self.gts: gts,
                    self.seq_len: seq_len,
                    self.is_training: True}
      output_feed = [self.loss_each_data, # Loss for this batch.
                     self.outputs,
                     self.loss_summary]

      outputs = session.run(output_feed, input_feed)

      return outputs[0], outputs[1], outputs[2]  # No gradient norm, loss, outputs.

