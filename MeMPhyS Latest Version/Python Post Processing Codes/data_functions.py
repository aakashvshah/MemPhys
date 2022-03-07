#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:39:04 2022

@author: Dr. Shantanu Shahane
"""
import numpy as np
import glob
import pandas as pd

def get_interp_grid(ecc_ratio, r_i, r_o, n_xi, n_theta):
    d=r_o-r_i; ecc=ecc_ratio*d; #computed parameters

    xi=np.linspace(0.0, 1.0, n_xi); theta_scaled=np.linspace(0.0, 1.0, n_theta, endpoint=False)
    xi,theta_scaled = np.meshgrid(xi, theta_scaled)

    theta = 2*np.pi*theta_scaled
    l = np.sqrt(r_o**2 - (ecc*np.sin(theta))**2 ) -ecc*np.cos(theta)
    x = ecc + (r_i + ((l-r_i)*xi) )*np.cos(theta)
    y = (r_i + ((l-r_i)*xi) )*np.sin(theta)

    return x, y, xi, theta, d, ecc

def read_simulations(main_folder, fields_filename_ending = 'fields_gmsh.csv', triangles_filename_ending = 'triangles_gmsh.csv'):
    fname=glob.glob(main_folder+ '/*'+ fields_filename_ending)
    assert len(fname)==1, 'Found more than single file'
    fields=pd.read_csv(fname[0]);
    fname=glob.glob(main_folder+ '/*'+ triangles_filename_ending)
    assert len(fname)==1, 'Found more than single file'
    triangles=pd.read_csv(fname[0]);
    return fields, triangles