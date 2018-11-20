from __future__ import division

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import viz
import time
import copy
import data_utils

def fkl_relative(xyz, parent):
  assert len(xyz) == 63
  xyz = np.reshape(xyz, (-1, 3))
  for i in range(len(xyz)):
    if parent[i] == -1:
      pass
    else:
      xyz[i] += xyz[ parent[i] ]

  return np.reshape(xyz, [-1])
def fkl( angles, parent, offset, rotInd, expmapInd ):
  """
  Convert joint angles and bone lenghts into the 3d points of a person.
  Based on expmap2xyz.m, available at
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

  Args
    angles: 99-long vector with 3d position and 3d joint angles in expmap format
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  Returns
    xyz: 32x3 3d points that represent a person in 3d space
  """

  assert len(angles) == 99

  # Structure that indicates parents for each joint
  njoints   = 32
  xyzStruct = [dict() for x in range(njoints)]

  for i in np.arange( njoints ):

    if not rotInd[i] : # If the list is empty
      xangle, yangle, zangle = 0, 0, 0
    else:
      xangle = angles[ rotInd[i][0]-1 ]
      yangle = angles[ rotInd[i][1]-1 ]
      zangle = angles[ rotInd[i][2]-1 ]

    r = angles[ expmapInd[i] ]

    thisRotation = data_utils.expmap2rotmat(r)
    thisPosition = np.array([xangle, yangle, zangle])

    if parent[i] == -1: # Root node
      xyzStruct[i]['rotation'] = thisRotation
      xyzStruct[i]['xyz']      = np.reshape(offset[i,:], (1,3)) + thisPosition
    else:
      xyzStruct[i]['xyz'] = (offset[i,:] + thisPosition).dot( xyzStruct[ parent[i] ]['rotation'] ) + xyzStruct[ parent[i] ]['xyz']
      xyzStruct[i]['rotation'] = thisRotation.dot( xyzStruct[ parent[i] ]['rotation'] )

  xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
  xyz = np.array( xyz ).squeeze()
  xyz = xyz[:,[0,2,1]]
  # xyz = xyz[:,[2,0,1]]


  return np.reshape( xyz, [-1] )

def _some_variables():
  """
  We define some variables that are useful to run the kinematic tree

  Args
    None
  Returns
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  """

  parent = np.array([-1,0,1,2,3,2,5,6,7,2,9,10,11,0,13,14,15,0,17,18,19])
  return parent

def main():

  parent = _some_variables()
  
  # numpy implementation
  real_person = np.load('sample.npz')['real_person']
  real_person = np.reshape(real_person, [real_person.shape[0], -1])
  character = np.load('sample.npz')['character']
  character = np.reshape(character, [character.shape[0], -1])
  
  nframes_condition = real_person.shape[0] - character.shape[0]
  nframes_pred = character.shape[0]
  
  # === Plot and animate ===
  fig = plt.figure()
  ax = plt.gca(projection='3d')
  ob = viz.Ax3DPose(ax)

  # Plot the conditioning ground truth
  if nframes_condition > 0:
    for i in range(nframes_condition):
      ob.update(fkl_relative(real_person[i], parent))
      plt.show(block=False)
      fig.canvas.draw()
      plt.pause(0.01)

  # Plot the prediction
  for i in range(nframes_pred):
    ob.update( fkl_relative(real_person[nframes_condition+i], parent),
               fkl_relative(character[i], parent))
    plt.show(block=False)
    fig.canvas.draw()
    plt.pause(0.01)


if __name__ == '__main__':
  main()
