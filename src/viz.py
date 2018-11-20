"""Functions to visualize human poses"""

import matplotlib.pyplot as plt
import data_utils
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D

class Ax3DPose(object):
  def __init__(self, ax, rlcolor="#3498db", rrcolor="#e74c3c",
                         clcolor="#9b59b6", crcolor="#2ecc71"):
    """
    Create a 3d pose visualizer that can be updated with new poses.

    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation
    self.I   = np.array([0,1,2,3,2,5,6,7,2, 9,10,11, 0,13,14,15, 0,17,18,19])
    self.J   = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    # Left / right indicator
    self.LR  = np.array([0,0,0,0,0,0,0,0,1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
    self.ax = ax

    self.rlcolor=rlcolor
    self.rrcolor=rrcolor
    self.clcolor=clcolor
    self.crcolor=crcolor

    vals_real_person = np.zeros((21, 3))
    vals_character = np.zeros((21,3))

    # Make connection matrix
    self.plots = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals_real_person[self.I[i], 0], vals_real_person[self.J[i], 0]] )
      y = np.array( [vals_real_person[self.I[i], 1], vals_real_person[self.J[i], 1]] )
      z = np.array( [vals_real_person[self.I[i], 2], vals_real_person[self.J[i], 2]] )
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=self.rlcolor if self.LR[i] \
                                                    else self.rrcolor))
    for i in np.arange( len(self.I) ):
      x = np.array( [vals_character[self.I[i], 0], vals_character[self.J[i], 0]] )
      y = np.array( [vals_character[self.I[i], 1], vals_character[self.J[i], 1]] )
      z = np.array( [vals_character[self.I[i], 2], vals_character[self.J[i], 2]] )
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=self.clcolor if self.LR[i] \
                                                    else self.crcolor))

    self.ax.set_xlabel("x")
    self.ax.set_ylabel("y")
    self.ax.set_zlabel("z")

  def update(self, real_person_channels, character_channels=None):
    """
    Update the plotted 3d pose.

    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert real_person_channels.size == 63, "channels should have 96 entries, it has %d instead" % real_person_channels.size
    real_person_vals = np.reshape( real_person_channels, (21, -1) )
    
    for i in np.arange( len(self.I) ):
      x = np.array( [real_person_vals[self.I[i], 0], real_person_vals[self.J[i], 0]] )
      y = np.array( [real_person_vals[self.I[i], 1], real_person_vals[self.J[i], 1]] )
      z = np.array( [real_person_vals[self.I[i], 2], real_person_vals[self.J[i], 2]] )
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)


    if character_channels is not None:
      assert character_channels.size == 63, "channels should have 96 entries, it has %d instead" % character_channels.size
      character_vals = np.reshape( character_channels, (21, -1) )
    
      for i in np.arange( len(self.I)):
        x = np.array( [character_vals[self.I[i], 0], character_vals[self.J[i], 0]] )
        y = np.array( [character_vals[self.I[i], 1], character_vals[self.J[i], 1]] )
        z = np.array( [character_vals[self.I[i], 2], character_vals[self.J[i], 2]] )
        self.plots[i+len(self.I)][0].set_xdata(x)
        self.plots[i+len(self.I)][0].set_ydata(y)
        self.plots[i+len(self.I)][0].set_3d_properties(z)


    r = 2;
    xroot, yroot, zroot = real_person_vals[0,0],\
                          real_person_vals[0,1],\
                          real_person_vals[0,2]
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])

    self.ax.set_aspect('equal')
