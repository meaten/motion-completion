
"""Simple code for training an RNN for motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model

# Learning
tf.app.flags.DEFINE_float("learning_rate", .005, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate is multiplied by this much. 1 means no decay.")
tf.app.flags.DEFINE_integer("learning_rate_step", 10000, "Every this many steps, do decay.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("iterations", int(1e5), "Iterations to train for.")
# Architecture
tf.app.flags.DEFINE_string("architecture", "tied", "Seq2seq architecture to use: [basic, tied].")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("seq_length_in", 10, "Number of frames to feed into the encoder. 30 fps")
#tf.app.flags.DEFINE_integer("seq_length_out", 10, "Number of frames that the decoder has to predict. 30 fps")
tf.app.flags.DEFINE_integer("seq_max_length", 200, "")
tf.app.flags.DEFINE_boolean("omit_one_hot", True, "Whether to remove one-hot encoding from the data")
tf.app.flags.DEFINE_boolean("residual_velocities", False, "Add a residual connection that effectively models velocities")
# Directories
tf.app.flags.DEFINE_string("data_dir", os.path.normpath("./data/"), "Data directory")
tf.app.flags.DEFINE_string("train_dir", os.path.normpath("./experiments/"), "Training directory.")

tf.app.flags.DEFINE_string("action","fistbump", "The action to train on.")
tf.app.flags.DEFINE_string("loss_to_use","supervised", "The type of loss to use, supervised or sampling_based")

tf.app.flags.DEFINE_integer("test_every", 100, "How often to compute error on the test set.")
tf.app.flags.DEFINE_integer("save_every", 100, "How often to compute error on the test set.")
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")

tf.app.flags.DEFINE_string("gpu_assignment", "1", "Set gpu assignment")

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_assignment



train_dir = os.path.normpath(os.path.join( FLAGS.train_dir, FLAGS.action,
  'iterations_{0}'.format(FLAGS.iterations),
  FLAGS.architecture,
  FLAGS.loss_to_use,
  'omit_one_hot' if FLAGS.omit_one_hot else 'one_hot',
  'depth_{0}'.format(FLAGS.num_layers),
  'size_{0}'.format(FLAGS.size),
  'lr_{0}'.format(FLAGS.learning_rate),
  'residual_vel' if FLAGS.residual_velocities else 'not_residual_vel'))

summaries_dir = os.path.normpath(os.path.join( train_dir, "log" )) # Directory for TB summaries

def create_model(session, actions):
  """Create translation model and initialize or load parameters in session."""
  
  model = seq2seq_model.Seq2SeqModel(
    FLAGS.architecture,
    FLAGS.seq_length_in,
    #FLAGS.seq_length_out if not sampling else 100,
    FLAGS.seq_max_length,
    FLAGS.size, # hidden layer size
    FLAGS.num_layers,
    FLAGS.max_gradient_norm,
    FLAGS.batch_size,
    FLAGS.learning_rate,
    FLAGS.learning_rate_decay_factor,
    summaries_dir,
    FLAGS.loss_to_use ,
    len( actions ),
    not FLAGS.omit_one_hot,
    FLAGS.residual_velocities,
    dtype=tf.float32)

  if FLAGS.load <= 0:
    print("Creating model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    return model

  ckpt = tf.train.get_checkpoint_state( train_dir, latest_filename="checkpoint")
  print( "train_dir", train_dir )

  if ckpt and ckpt.model_checkpoint_path:
    # Check if the specific checkpoint exists
    if FLAGS.load > 0:
      if os.path.isfile(os.path.join(train_dir,"checkpoint-{0}.index".format(FLAGS.load))):
        ckpt_name = os.path.normpath(os.path.join( os.path.join(train_dir,"checkpoint-{0}".format(FLAGS.load)) ))
      else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
    else:
      ckpt_name = os.path.basename( ckpt.model_checkpoint_path )

    print("Loading model {0}".format( ckpt_name ))
    model.saver.restore( session, ckpt.model_checkpoint_path )
    return model
  else:
    print("Could not find checkpoint. Aborting.")
    raise( ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )

  return model


def train():
  """Train a seq2seq model on human motion"""

  actions = define_actions( FLAGS.action )

  number_of_actions = len( actions )

  train_set, test_set, rp_stats, ch_stats = read_all_data(
    actions, FLAGS.seq_length_in, FLAGS.data_dir, not FLAGS.omit_one_hot )

  # Limit TF to take a fraction of the GPU memory
  gpu_options = tf.GPUOptions(allow_growth=True)
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}

  with tf.Session(config=tf.ConfigProto( gpu_options=gpu_options, device_count = device_count )) as sess:

    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

    model = create_model( sess, actions )
    model.train_writer.add_graph( sess.graph )
    print( "Model created" )

    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
    previous_losses = []

    step_time, loss = 0, 0

    for _ in xrange( FLAGS.iterations ):

      start_time = time.time()
      batch_shuffle = [ i for i in range(len(train_set))][:FLAGS.batch_size]
      
      # === Training step ===
      #encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch( train_set, not FLAGS.omit_one_hot )
      encoder_inputs = train_set[batch_shuffle, 0, :FLAGS.seq_length_in]
      decoder_inputs = train_set[batch_shuffle, 0, FLAGS.seq_length_in:]
      decoder_outputs= train_set[batch_shuffle, 1, FLAGS.seq_length_in:]

      _, step_loss, loss_summary, lr_summary = model.step( sess, encoder_inputs, decoder_inputs, decoder_outputs, False )
      
      model.train_writer.add_summary(summary=loss_summary,global_step=current_step)
      #model.train_writer.add_summary( summary=lr_summary, global_step=current_step )
      
      if current_step % 10 == 0:
        print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss ))

      step_time += (time.time() - start_time) / FLAGS.test_every
      loss += step_loss / FLAGS.test_every
      current_step += 1

      # === step decay ===
      if current_step % FLAGS.learning_rate_step == 0:
        sess.run(model.learning_rate_decay_op)

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.test_every == 0:

        # === Validation with randomly chosen seeds ===
        forward_only = True

        encoder_inputs = test_set[:, 0, :FLAGS.seq_length_in]
        decoder_inputs = test_set[:, 0, FLAGS.seq_length_in:]
        decoder_outputs= test_set[:, 1, FLAGS.seq_length_in:]
        
        step_loss, loss_summary = model.step(sess,
            encoder_inputs, decoder_inputs, decoder_outputs, forward_only)
        val_loss = step_loss # Loss book-keeping

        model.test_writer.add_summary(summary=loss_summary,global_step=current_step)

        
        print()
        print("============================\n"
              "Global step:         %d\n"
              "Learning rate:       %.4f\n"
              "Step-time (ms):     %.4f\n"
              "Train loss avg:      %.4f\n"
              "--------------------------\n"
              "Val loss:            %.4f\n"
              "============================" % (model.global_step.eval(),
              model.learning_rate.eval(), step_time*1000, loss,val_loss))
        print()

        previous_losses.append(loss)

        # Save the model
        if current_step % FLAGS.save_every == 0:
          print( "Saving the model..." ); start_time = time.time()
          model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'checkpoint')), global_step=current_step )
          print( "done in {0:.2f} ms".format( (time.time() - start_time)*1000) )

        # Reset global time and loss
        step_time, loss = 0, 0

        sys.stdout.flush()
      model.train_writer.flush()
      model.test_writer.flush()


def sample():
  """Sample predictions for srnn's seeds"""

  if FLAGS.load <= 0:
    raise( ValueError, "Must give an iteration to read parameters from")

  actions = define_actions( FLAGS.action )

  # Use the CPU if asked to
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto( device_count = device_count )) as sess:

    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, actions)
    print("Model created")

    # Load all the data
    train_set, test_set, rp_stats, ch_stats = read_all_data(
      actions, FLAGS.seq_length_in, FLAGS.data_dir, not FLAGS.omit_one_hot )
    
    # Clean and create a new h5 file of samples
    SAMPLES_DNAME = 'samples'
    try:
      import shutil; shutil.rmtree( SAMPLES_DNAME )
    except OSError:
      pass

    # Predict and save for each action
    for action in actions:


      encoder_inputs = test_set[:, 0, :FLAGS.seq_length_in]
      decoder_inputs = test_set[:, 0, FLAGS.seq_length_in:]
      decoder_outputs= test_set[:, 1, FLAGS.seq_length_in:]
      
      forward_only = True
      pred_loss, pred_poses, _ = model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs, forward_only, False)

      pred_poses = np.array(pred_poses)

      
      # denormalizes too
      test_set[:,0] = data_utils.unNormalizeData(test_set[:,0], rp_stats, actions, False)
      test_set[:,1] = data_utils.unNormalizeData(test_set[:,1], ch_stats, actions, False)
      pred_poses = data_utils.unNormalizeData(np.reshape(pred_poses, [pred_poses.shape[1],
                                                                      pred_poses.shape[0],
                                                                      -1]),
                                                         ch_stats, actions, False)

      # Save the conditioning seeds

      # Save the samples
      os.mkdir(SAMPLES_DNAME)
      for i in range(len(pred_poses)):
        np.savez(SAMPLES_DNAME+'/sample{}.npz'.format(i),
                 real_person=test_set[i,0],
                 character=pred_poses[i],
                 ground_truth=test_set[i,1],
                 loss=pred_loss)
      '''' 
      # Compute and save the errors here
      mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )

      for i in np.arange(8):

        eulerchannels_pred = srnn_pred_expmap[i]

        for j in np.arange( eulerchannels_pred.shape[0] ):
          for k in np.arange(0,eulerchannels_pred.shape[1],3):
            eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
              data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

        eulerchannels_pred[:,0:6] = 0

        # Pick only the dimensions with sufficient standard deviation. Others are ignored.
        idx_to_use = np.where( np.std( eulerchannels_pred, 0 ) > 1e-4 )[0]

        euc_error = np.power( srnn_gts_euler[action][i][:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt( euc_error )
        mean_errors[i,:] = euc_error

      mean_mean_errors = np.mean( mean_errors, 0 )
      print( action )
      print( ','.join(map(str, mean_mean_errors.tolist() )) )

      with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
        node_name = 'mean_{0}_error'.format( action )
        hf.create_dataset( node_name, data=mean_mean_errors )
      '''
  return


def define_actions( action ):
  """
  Define the list of actions we are using.

  Args
    action: String with the passed action. Could be "all"
  Returns
    actions: List of strings of actions
  Raises
    ValueError if the action is not included in H3.6M
  """

  actions = ["fistbump"]

  if action in actions:
    return [action]

  if action == "all":
    return actions

  if action == "all_srnn":
    return ["walking", "eating", "smoking", "discussion"]

  raise( ValueError, "Unrecognized action: %d" % action )


def read_all_data( actions, seq_length_in, data_dir, one_hot ):
  """
  Loads data for training/testing and normalizes it.

  Args

  Returns

  """

  # === Read training data ===
  print ("Reading training data (seq_len_in: {0}).".format(
           seq_length_in))

  #####
  #need to add one-hot vector to each frame later
  #####
  for i, action in enumerate(actions):
    action_seq = np.load(data_dir+'/'+action+'.npz')['seq']
    action_seq_len = np.load(data_dir+'/'+action+'.npz')['seq_len']
    all_seq = np.concatenate((all_seq, action_seq), axis=0) if i > 0 else action_seq
    all_seq_len = np.concatenate((all_seq_len, action_seq_len), axis=0) \
                  if i > 0 else action_seq_len

  """
  all_seq: [data_index, skeleton_num, frame_len, joints, xyz]
  """
  all_seq = np.concatenate((data_utils.rotate_data(all_seq),
                            data_utils.rotate_data(all_seq[:,[1,0]])), axis=0)

  #all_seq = np.concatenate((all_seq,all_seq[:,[1,0]]), axis=0)
  
  all_seq_len = np.asarray(np.concatenate((all_seq_len, all_seq_len), axis=0), dtype=int)
  shape = all_seq.shape
  all_seq = np.reshape(all_seq, [shape[0], shape[1], shape[2], -1])
  
  from sklearn.model_selection import train_test_split
  train_seq, test_seq, train_seq_len, test_seq_len = train_test_split(all_seq,
                                                                      all_seq_len,
                                                                      test_size=0.25,
                                                                      random_state=0)
  
  for i, (seq, seq_len) in enumerate(zip(train_seq, train_seq_len)):
    try:
      complete_seq = np.concatenate((complete_seq, seq[: ,:seq_len]), axis=1) if i > 0 \
                     else seq[:, :seq_len]
    except ValueError:
      import pdb; pdb.set_trace()
  complete_real_person, complete_character = complete_seq
  
  # Compute normalization stats
  rp_stats = data_utils.normalization_stats(complete_real_person)
  ch_stats = data_utils.normalization_stats(complete_character)

  # Normalize -- subtract mean, divide by stdev
  train_seq[:,0] = data_utils.normalize_data( train_seq[:,0], rp_stats, actions,one_hot )
  train_seq[:,1] = data_utils.normalize_data( train_seq[:,1], ch_stats, actions,one_hot )
  test_seq[:, 0] = data_utils.normalize_data( test_seq[:, 0], rp_stats, actions,one_hot )
  test_seq[:, 1] = data_utils.normalize_data( test_seq[:, 1], ch_stats, actions,one_hot )
  
  print("done reading data.")

  return train_seq, test_seq, rp_stats, ch_stats


def main(_):
  if FLAGS.sample:
    sample()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
