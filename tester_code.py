# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 19:05:06 2017

@author: Nick
"""

import pandas as pd
import numpy as np
import os
path = '/users/nick/dropbox/GAPP/02_Main Evaluation/Activities/'\
       '18_voting_and_networks/2_code/diffusion_model_cythonized'
os.chdir(path)

import igraph as ig

# Cython import
import pyximport; pyximport.install()
import diffusion_model_cython as dm





g = ig.Graph()
g.add_vertices(4)
g.add_edges([(0,1), (1,2)])

# Simple test:
test = dm.diffusion_run(g, 1, number_of_runs=3, initially_infected_nodes={0},
                        tests=True)
                        
                        


assert (test == pd.Series([1/3,0.5], index=[1,2])).all()


g2 = ig.Graph.Erdos_Renyi(n=20, p=0.1)



a = ni.average_integration(g2, num_runs=4, num_steps = 5, debug=True)

print(a)


