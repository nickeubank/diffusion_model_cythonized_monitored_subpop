# diffusion_model_cythonized
Model for simulating simple diffusion on igraph graphs with cython speedups. 

See doc-string of `diffusion_run` function in `diffusion_model_cython.pyx`. 

On most systems with pyximport installed, can be imported via:

	import os
	os.chdir(PATH_TO_PYX_FILE)
    import pyximport; pyximport.install()
    import diffusion_model_cython


