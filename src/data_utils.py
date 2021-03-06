
"""Functions that help with data processing for human3.6m"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import copy
from sklearn.decomposition import PCA

def rotmat2euler( R ):
  """
  Converts a rotation matrix to Euler angles
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

  Args
    R: a 3x3 rotation matrix
  Returns
    eul: a 3x1 Euler angle representation of R
  """
  if R[0,2] == 1 or R[0,2] == -1:
    # special case
    E3   = 0 # set arbitrarily
    dlta = np.arctan2( R[0,1], R[0,2] );

    if R[0,2] == -1:
      E2 = np.pi/2;
      E1 = E3 + dlta;
    else:
      E2 = -np.pi/2;
      E1 = -E3 + dlta;

  else:
    E2 = -np.arcsin( R[0,2] )
    E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
    E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

  eul = np.array([E1, E2, E3]);
  return eul


def quat2expmap(q):
  """
  Converts a quaternion to an exponential map
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

  Args
    q: 1x4 quaternion
  Returns
    r: 1x3 exponential map
  Raises
    ValueError if the l2 norm of the quaternion is not close to 1
  """
  if (np.abs(np.linalg.norm(q)-1)>1e-3):
    print(np.linalg.norm(q), ' >1e-3 so, normalize')
    q = q/np.linalg.norm(q)
    #raise ValueError("quat2expmap: input quaternion is not norm 1")

  sinhalftheta = np.linalg.norm(q[1:])
  coshalftheta = q[0]

  r0    = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
  theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
  theta = np.mod( theta + 2*np.pi, 2*np.pi )

  if theta > np.pi:
    theta =  2 * np.pi - theta
    r0    = -r0

  r = r0 * theta
  return r

def rotmat2quat(R):
  """
  Converts a rotation matrix to a quaternion
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

  Args
    R: 3x3 rotation matrix
  Returns
    q: 1x4 quaternion
  """
  rotdiff = R - R.T;

  r = np.zeros(3)
  r[0] = -rotdiff[1,2]
  r[1] =  rotdiff[0,2]
  r[2] = -rotdiff[0,1]
  sintheta = np.linalg.norm(r) / 2;
  r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps );

  costheta = (np.trace(R)-1) / 2;

  theta = np.arctan2( sintheta, costheta );

  q      = np.zeros(4)
  q[0]   = np.cos(theta/2)
  q[1:] = r0*np.sin(theta/2)
  return q

def rotmat2expmap(R):
  return quat2expmap( rotmat2quat(R) );

def expmap2rotmat(r):
  """
  Converts an exponential map angle to a rotation matrix
  Matlab port to python for evaluation purposes
  I believe this is also called Rodrigues' formula
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

  Args
    r: 1x3 exponential map
  Returns
    R: 3x3 rotation matrix
  """
  theta = np.linalg.norm( r )
  r0  = np.divide( r, theta + np.finfo(np.float32).eps )
  r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
  r0x = r0x - r0x.T
  R = np.eye(3,3) + np.sin(theta)*r0x + (1-np.cos(theta))*(r0x).dot(r0x);
  return R


def unNormalizeData(normalizedData, stats, actions, one_hot ):
  """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

  Args
    normalizedData: nxd matrix with normalized data
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    actions: list of strings with the encoded actions
    one_hot: whether the data comes with one-hot encoding
  Returns
    origData: data originally used to
  """
  data_mean = stats[0]
  data_std = stats[1]
  dimensions_to_use = stats[3]
  pca = stats[4]

  if pca is not None:
    shape = normalizedData.shape
    normalizedData = np.reshape( normalizedData, [-1, pca.get_params()['n_components']])
    normalizedData = pca.inverse_transform(normalizedData)
    normalizedData = np.reshape(normalizedData, [shape[0], shape[1], -1])
  else:
    pass
  
  D = data_mean.shape[0]
  origData = np.zeros((normalizedData.shape[0],
                       normalizedData.shape[1],
                       D), dtype=np.float32)

  if one_hot:
    origData[:, :, dimensions_to_use] = normalizedData[:, :-len(actions)]
  else:
    origData[:, :, dimensions_to_use] = normalizedData

  # potentially ineficient, but only done once per experiment
  stdMat = data_std.reshape((1, D))
  #stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  #meanMat = np.repeat(meanMat, T, axis=0)
  origData = np.multiply(origData, stdMat) + meanMat
  return origData


def revert_output_format(poses, data_mean, data_std, dim_to_ignore, actions, one_hot):
  """
  Converts the output of the neural network to a format that is more easy to
  manipulate for, e.g. conversion to other format or visualization

  Args
    poses: The output from the TF model. A list with (seq_length) entries,
    each with a (batch_size, dim) output
  Returns
    poses_out: A tensor of size (batch_size, seq_length, dim) output. Each
    batch is an n-by-d sequence of poses.
  """
  seq_len = len(poses)
  if seq_len == 0:
    return []

  batch_size, dim = poses[0].shape

  poses_out = np.concatenate(poses)
  poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
  poses_out = np.transpose(poses_out, [1, 0, 2])

  poses_out_list = []
  for i in xrange(poses_out.shape[0]):
    poses_out_list.append(
      unNormalizeData(poses_out[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot))

  return poses_out_list


def readCSVasFloat(filename):
  """
  Borrowed from SRNN code. Reads a csv and returns a float matrix.
  https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

  Args
    filename: string. Path to the csv file
  Returns
    returnArray: the read data in a float32 matrix
  """
  returnArray = []
  lines = open(filename).readlines()
  for line in lines:
    line = line.strip().split(',')
    if len(line) > 0:
      returnArray.append(np.array([np.float32(x) for x in line]))

  returnArray = np.array(returnArray)
  return returnArray


def rotmat(x,y):
  norm_x = np.linalg.norm(x)
  norm_y = np.linalg.norm(y)
  if norm_x == 0 or norm_y == 0 or np.allclose(x, y):
    return np.eye(3)
  x = x / norm_x
  y = y / norm_y
                          
  #return R matrix that rotate x to y
  dot_xy = np.dot(x, y)
  cos = dot_xy
  rad = np.arccos(cos)
  cross = np.cross(x,y)
  if np.linalg.norm(cross) == 0:
    return np.eye(3)
  n = cross / np.linalg.norm(cross)
  R = np.array([[np.cos(rad)+n[0]*n[0]*(1-np.cos(rad)),
                 n[0]*n[1]*(1-np.cos(rad))-n[2]*np.sin(rad),
                 n[0]*n[2]*(1-np.cos(rad))+n[1]*np.sin(rad)],
                [n[1]*n[0]*(1-np.cos(rad))+n[2]*np.sin(rad),
                 np.cos(rad)+n[1]*n[1]*(1-np.cos(rad)),
                 n[1]*n[2]*(1-np.cos(rad))-n[0]*np.sin(rad)],
                [n[2]*n[0]*(1-np.cos(rad))-n[1]*np.sin(rad),
                 n[2]*n[1]*(1-np.cos(rad))+n[0]*np.sin(rad),
                 np.cos(rad)+n[2]*n[2]*(1-np.cos(rad))]])
  return R


def rotate_data(all_seq):
  all_seq_out = np.zeros_like(all_seq)
  for i, seq in enumerate(all_seq):
    origin = seq[0,0,0]
    seq[:,:,0] -= origin
    R = rotmat([1,0,0],seq[1,0,0] - seq[0,0,0])
    seq = np.matmul(seq, R)
    R = rotmat([0,0,1], seq[0,0,1])
    seq = np.matmul(seq, R)
    all_seq_out[i] = seq
  
  return all_seq_out

def normalize_data( data, stats, actions, one_hot ):
  """
  Normalize input data by removing unused dimensions, subtracting the mean and
  dividing by the standard deviation

  Args
    data: nx96 matrix with data to normalize
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dim_to_use: vector with dimensions used by the model
    actions: list of strings with the encoded actions
    one_hot: whether the data comes with one-hot encoding
  Returns
    data_out: the passed data matrix, but normalized
  """
  data_out = np.zeros_like(data)
  data_mean = stats[0]
  data_std = stats[1]
  dim_to_use = stats[3]
  pca = stats[4]
  
  if not one_hot:
    # No one-hot encoding... no need to do anything special
    data_out = np.divide( (data - data_mean), data_std )
    data_out = data_out[ :, :,dim_to_use ]
    if pca is not None:
      shape = data_out.shape
      data_out = np.reshape(data_out, [-1,shape[2]])
      data_out = pca.transform(data_out)
      data_out = np.reshape(data_out, [shape[0], shape[1], pca.get_params()['n_components']])
    else:
      pass

  else:
    print('this has not been implemented')

  return data_out


def normalization_stats(completeData, dim_to_compressed):
  """"
  Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

  Args
    completeData: nx99 matrix with data to normalize
  Returns
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    dimensions_to_use: vector with dimensions used by the model
  """
  data_mean = np.mean(completeData, axis=0)
  data_std  =  np.std(completeData, axis=0)

  dimensions_to_ignore = []
  dimensions_to_use    = []

  dimensions_to_ignore.extend( list(np.where(data_std < 1e-4)[0]) )
  dimensions_to_use.extend( list(np.where(data_std >= 1e-4)[0]) )


  if dim_to_compressed is not None:
    data_std[dimensions_to_ignore] = 1.0

    pca = PCA(n_components=dim_to_compressed)
    pca.fit(completeData)
  else:
    pca = None

  '''
  #to determine n_components

  ev_ratio = pca.explained_variance_ratio_
  ev_ratio = np.hstack([0,ev_ratio.cumsum()])
  import matplotlib.pyplot as plt
  plt.plot(ev_ratio)
  plt.show()
  '''
  
  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use, pca
