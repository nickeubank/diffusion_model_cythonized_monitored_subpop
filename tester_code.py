# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 19:05:06 2017

@author: Nick
"""

import pandas as pd
import numpy as np
import os
os.chdir('/users/nick/dropbox/GAPP/02_Main Evaluation/Activities/18_voting_and_networks/2_code/diffusion_model')
import igraph as ig

# Cython import
import pyximport; pyximport.install()
import diffusion_model_cython as dm





g = ig.Graph()
g.add_vertices(4)
g.add_edges([(0,1), (1,2)])

# Simple test:
test = dm.diffusion_run(g, 0.1, number_of_runs=3, num_starting_infections=1)

graph, p, number_of_runs, 
                  num_starting_infections=None,  
                  initially_infected_nodes=None,
                  max_iter=1000,
                  thresholds=[0.1, 0.25, 0.5, 0.75],
                  agg_function=np.mean,
                  tests=False):

assert (test == pd.Series([1/3,0.5], index=[1,2])).all()


g2 = ig.Graph.Erdos_Renyi(n=20, p=0.1)



a = ni.average_integration(g2, num_runs=4, num_steps = 5, debug=True)

print(a)


