#!python
# -*- coding: utf-8 -*-
from __future__ import division
"""
Created on Sat Feb 13 12:10:37 2016

@author: Nick
"""


##
import igraph as ig
import random
import pandas as pd
import numpy as np


#######
# Start with the Cython functions. 
#
# These are stand-alones that will be compiled with 
# cython to massively speed up run time. 
#
#######

from libc.stdlib cimport rand, RAND_MAX

cdef float randnum():
    return rand() / RAND_MAX


cdef list sample_from_neighbors(list neighbors, float p):
    cdef list to_return
    cdef int i

    to_return = []    
    
    for i in neighbors:
        if randnum() <= p:
            to_return.append(i)

    return to_return


cdef set stand_alone_step(object graph, set infected, float p):

        # Infected vertices spread the infeciton
        cdef set newly_infected
        cdef int vertex 

        newly_infected = set()
                        
        for vertex in infected:
            neighbors = graph.neighbors(vertex)
            choices = sample_from_neighbors(neighbors, p)
            newly_infected.update(choices)

        return newly_infected
        

######
# End cython. Into regular python components
######

class SIModel(object):
    """
    SI epidemic model for networks.

    Nodes start in uninfected and move to infected. 
    Starting infections is number of nodes to being with infected.

    Will pick at random unless `initial_infections` also set. 
    If those are both set (and are mutually coherent) 

    """
    
    def __init__(self, graph,  p=0.1, num_starting_infections=None,  
                 tests=False, initially_infected_nodes=None):
        """
        Constructs an SI model on the given `graph` with
        infection rate `p`
        """

        self.graph = graph
        self.network_size = graph.vcount()
        self.p = p
        self.infected = set()
        self.uninfected = set(range(graph.vcount()-1))
        self.tests = tests

        # Setup initial state

        # number specified, but not specific nodes
        if initially_infected_nodes is None and num_starting_infections is not None:
            self.set_initial_infections(num_starting_infections)

        elif initially_infected_nodes is not None and num_starting_infections is None:
            self.move_to_infected(initially_infected_nodes) 

        elif initially_infected_nodes is None and num_starting_infections is None:
            raise ValueError("Must specify initially_infected_nodes or num_starting_infections")

        else:
            raise ValueError("Cannot specify both num_starting_infections and initially_infected_nodes")
        
    def set_initial_infections(self, num_starting_infections):
        initials = set( random.sample(self.uninfected, num_starting_infections) )
        self.move_to_infected(initials)

    def step(self, iteration_counter=None):
        """
        Runs a single step of the SI model simulation.

        Calls not-in-class function that it optimized by Cython 
        (since it's slowest part)

        """
        newly_infected = stand_alone_step(self.graph, self.infected, self.p)
        self.move_to_infected(newly_infected)

    def move_to_infected(self, newly_infected):
        # Move into infected sets, remove if in uninfected
        self.infected.update(newly_infected)
        self.uninfected.difference_update(newly_infected)
        
        # Embedded tests
        if self.tests:
            assert len(self.infected.intersection(self.uninfected)) == 0        

    @property
    def share_infected(self):        
        return len(self.infected) / self.network_size


def diffusion_run(graph, p, number_of_runs, 
                  num_starting_infections=None,  
                  initially_infected_nodes=None,
                  max_iter=1000,
                  thresholds=[0.1, 0.25, 0.5, 0.75],
                  agg_function=np.mean,
                  tests=False):

    """ 
    Run diffusion model and report back number of steps required to 
    infect shares of nodes set by threshold. 

    graph:  iGraph object on which to run model
    p:      probability infection spreads across an edge in each step. 
    num_starting_infections: Number of nodes initially infected. 
            will be selected uniformly at random. Cannot be combined
            with initially_infected_nodes. 
    initially_infected_nodes: list of specific nodes to be set as 
            infected at start of simulation. Cannot be combined 
            with num_starting_infections. 
    number_of_runs: Number of times to run model. 
    max_iter: maximum number of steps allowed in simulation. 
            Default 1000. 
    thresholds: infection thresholds to be evaluated.
            Must be monotonic from smallest to largest. 
            Default [0.1, 0.25, 0.5, 0.75].
    agg_function: function used to aggregate all the runs. 
            Default is np.mean. 
    tests: run internal integrity tests. Adds to running time, 
            mostly for internal purposes. 
    
    """

    # Tests!
    if tests:
        test_suite()

    # Make sure monotonic
    diffs = np.diff(np.array(thresholds))
    if not (diffs > 0).all():
        raise ValueError("Thresholds must be strictly monotonic")

    if np.max(thresholds) > 1:
        raise ValueError("Thresholds must be between 0 and 1")        
    
    results = pd.DataFrame(np.nan, index=thresholds, columns=range(number_of_runs))


    # Run 
    for run in range(number_of_runs):

        # Setup model.                   
        si_model = SIModel(graph=graph, p=p, num_starting_infections=num_starting_infections, 
                           initial_infection=initially_infected_nodes, tests=tests)
    
    

        for i in range(0, max_iter):
        
            # Check coverage
            for threshold in thresholds:
                if pd.isnull(results.loc[threshold, run]) and si_model.share_infected >= threshold:                
                    results.loc[threshold, run] = i
            
            # If last threshold hit, break out
            if pd.notnull(results.loc[thresholds[-1], run]): 
                break
            
            # Iterate!
            si_model.step(i)


    result = results.apply(agg_function, axis=1)

    return result
    

#####
# Tests
#####

def test_suite():
    from pandas.util.testing import assert_series_equal
    line = ig.Graph()
    line.add_vertices(range(2))
    line.add_edges([(0,1)])    
    
    # Start with 1 of 2, should spread from half (starting seed) to full from 0 to 1. 
    test_result = diffusion_run(graph=line, p=1)
    assert_series_equal(test_result, pd.Series([0.0,0,0,1],index = [0.1, 0.25, 0.5, 0.75], name=0))

    # If one watched and one seeded, should all trigger at 0.
    test_result = diffusion_run(graph=line, nodes_to_watch=set(range(1)), p=1, tests=False,
                      district=0)
    assert_series_equal(test_result, pd.Series([0.0,0,0,0],index = [0.1, 0.25, 0.5, 0.75],name=0))                      

    # If watch 2 and start with 2, trigger right away
    test_result = diffusion_run(graph=line, nodes_to_watch=set(range(2)), p=1, tests=False,
                      district=0, starting_infections=2)
    assert_series_equal(test_result , pd.Series([0.0,0,0,0],index = [0.1, 0.25, 0.5, 0.75],name=0))                    


    # Immediate spread in full graph
    full = ig.Graph.Full(10)
    test_result = diffusion_run(graph=full, nodes_to_watch=set(range(10)), p=1, tests=False,
                      district=0)
    assert_series_equal(test_result , pd.Series([0.0,1,1,1],index = [0.1, 0.25, 0.5, 0.75],name=0))

    # No spread in disconnected
    sparse = ig.Graph()
    sparse.add_vertices(4)
    sparse.add_edges([(0,1), (2,3)])
    test_result = diffusion_run(graph=sparse, nodes_to_watch=set(range(4)), p=1, tests=False,
                      district=0)
    assert_series_equal(test_result , pd.Series([0.0,0,1,np.nan],index = [0.1, 0.25, 0.5, 0.75],name=0))                      
    
    
    # No spread if p=0
    sparse = ig.Graph.Full(2)
    test_result = diffusion_run(graph=sparse, nodes_to_watch=set(range(2)), p=0, tests=False,
                      district=0)
    assert_series_equal(test_result , pd.Series([0.0,0,0,np.nan],index = [0.1, 0.25, 0.5, 0.75],name=0))                      
    
    

    # If make line and start at tail, should diffuse one per step
    line = ig.Graph()
    line.add_vertices(range(10))
    line.add_edges([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9)])    
    
    test_result = diffusion_run(graph=line, nodes_to_watch=set(range(10)), p=1, initial_infection={0}, district=0)
    assert_series_equal(test_result, pd.Series([0.0,2,4,7],index = [0.1, 0.25, 0.5, 0.75], name=0))

    # Modify thresholds
    test_result = diffusion_run(graph=line, nodes_to_watch=set(range(10)), p=1, initial_infection={0}, district=0, thresholds=[0,0.1,1])
    assert_series_equal(test_result, pd.Series([0.0,0,9],index = [0,0.1,1], name=0))

    ######
    # Test probabilities
    ####

    # negative binomial -- num failures before finishing, so subtract 1 from step count
    # For p=0.2
    # Mean 4, sd 4.5 -- 3se is 0.42. 
    run = 1000
    output = pd.Series(np.nan*run)
    for i in range(run):
        r = diffusion_run(graph=ig.Graph.Full(2), nodes_to_watch=set(range(2)), p=0.2, tests=False,
                      district=0, thresholds=[1], initial_infection={0})
        output.loc[i] = r.loc[1]

    output.mean()-1
    try:       
        assert output.mean()-1 > 3.5 and output.mean()-1 <4.5
    except:
        raise ValueError("Test on 1-link diffusion should have taken avg 5 steps (se 0.14), took {}".format(output.mean()))


    # More complex. Failures before 9 successes, so steps minus 9.
    # mean 3.86, sd2.3, 3 standard errors thus ~.7
    # for p=0.7
    line = ig.Graph()
    line.add_vertices(range(10))
    line.add_edges([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9)])    
    run = 100
    output = pd.Series(np.nan*run)
    for i in range(run):
        r = diffusion_run(graph=line, nodes_to_watch=set(range(10)), p=0.7, tests=False,
                      district=0, thresholds=[1], initial_infection={0})
        output.loc[i] = r.loc[1]

    output.mean() - 9
    try:       
        assert output.mean() - 9 < 4.5 and output.mean() - 9 > 3.1
    except:
        raise ValueError("Test on 9-line diffusion should have taken avg 12.86 steps (se 0.22), took {}".format(output.mean()))

    print('test graphs ok')

