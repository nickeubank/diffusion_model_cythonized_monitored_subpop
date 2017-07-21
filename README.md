# Diffusion Model with Subpopulation monitoring

Model for simulating simple diffusion on igraph graphs with cython speedups. 

Executes diffusion model on full network, but if values are passed to `watch`, 
allows for monitoring of infection rate on specified sub-population. 

See doc-string of `diffusion_run_threshold` function `diffusion_run_step` function 
in `diffusion_model_w_subpop.pyx`. Former counts steps required to reach a level of 
infection, the later reports infection rate at specified step counts. 

On most systems with pyximport installed, can be imported via:

	import os
	os.chdir(PATH_TO_PYX_FILE)
    import pyximport; pyximport.install()
    import diffusion_model_w_subpop


