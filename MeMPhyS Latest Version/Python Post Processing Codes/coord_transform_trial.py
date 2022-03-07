#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 10:01:19 2022

@author: Dr. Shantanu Shahane
"""
import numpy as np
import matplotlib.pyplot as plt
import data_functions as data_f

ecc_ratio=0.5; #range: [0.0, 0.5]
r_i=0.6; #range: [0.1, 0.6]
n_xi=11; n_theta=30; #discretization sizes
r_o=1.0; #fixed parameter

x, y, xi, theta, d, ecc = data_f.get_interp_grid(ecc_ratio, r_i, r_o, n_xi, n_theta)

plt.figure(figsize=(18,12))
plt.subplot(1,2,1)
plt.plot(r_o*np.cos(np.linspace(0.0, 2*np.pi, 361)), r_o*np.sin(np.linspace(0.0, 2*np.pi, 361)), '-b' )
plt.plot(ecc + r_i*np.cos(np.linspace(0.0, 2*np.pi, 361)), r_i*np.sin(np.linspace(0.0, 2*np.pi, 361)), '-b' )
plt.plot(x, y, 'ro'); plt.gca().set_aspect(1); plt.grid(); plt.xlabel('X'); plt.ylabel('Y')
plt.plot(ecc, 0, 'bo', markersize=11); plt.title('Cartesian Coordinates')
plt.subplot(1,2,2)
plt.plot(xi, theta/np.max(theta), 'ro'); plt.gca().set_aspect(1); plt.grid(); plt.title('Transformed Coordinates')
plt.xlabel('xi'); plt.ylabel('theta scaled')