# diffusion_model_cythonized
Model for simulating simple diffusion on igraph graphs with cython speedups. 

See doc-string of `diffusion_run_threshold` function `diffusion_run_step` function 
in `diffusion_model_cython.pyx`. Former counts steps required to reach a level of 
infection, the later reports infection rate at specified step counts. 

On most systems with pyximport installed, can be imported via:

	import os
	os.chdir(PATH_TO_PYX_FILE)
    import pyximport; pyximport.install()
    import diffusion_model_cython


