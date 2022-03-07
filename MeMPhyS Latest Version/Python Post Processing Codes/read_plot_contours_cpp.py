#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:44:05 2022

@author: Dr. Shantanu Shahane
"""
import data_functions as data_f
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri


main_folder='/home/shantanu/Desktop/MEGA/Research Thesis/meshless_reform/examples/navier_stokes_2D/'
ecc_ratio=0.5; #range: [0.0, 0.5]
r_i=0.6; #range: [0.1, 0.6]
n_xi=21; n_theta=24; #discretization sizes
r_o=1.0; #fixed parameter
T_o=0.0; T_i=1.0

x_interp, y_interp, xi_interp, theta_interp, d, ecc = data_f.get_interp_grid(ecc_ratio, r_i, r_o, n_xi, n_theta)

fields, triangles = data_f.read_simulations(main_folder)

triangulation = tri.Triangulation(fields['x'], fields['y'], triangles);

interp_T_x = tri.LinearTriInterpolator(triangulation, fields['T'])
T_interp = interp_T_x(x_interp, y_interp); T_interp[:,0]=T_i; T_interp[:,-1]=T_o

x_interp=np.append(x_interp, x_interp[0,:].reshape(1,-1), axis=0);
y_interp=np.append(y_interp, y_interp[0,:].reshape(1,-1), axis=0)
xi_interp=np.append(xi_interp, xi_interp[0,:].reshape(1,-1), axis=0);
theta_interp=np.append(theta_interp, 2*np.pi*np.ones((1,theta_interp.shape[1])), axis=0)
T_interp=np.append(T_interp, T_interp[0,:].reshape(1,-1), axis=0)
plt.figure()
plt.contourf(x_interp, y_interp, T_interp, cmap=plt.cm.jet, levels = 51)
plt.colorbar(); plt.gca().set_aspect(1);