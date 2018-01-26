# Complex Background Subtraction using Gaussian Models

To install dependences, initialize a virtual environment and execute:  
$ pip install -r requirements.txt

The zip file docs/test_data_dir.gz holds the recommended directory structure for the dataset used.  
To request the dataset used, please email Yannis Karakozis at [ick at princeton dot edu].  

The core executables used to run the models are:  
gmm_mov.py - Gaussian Mutlivariate Model  
mog_mov.py - Mixtures of Gausian Model  

testing.py allows you to compute core evaluation metrics once you have extracted foreground mattes using the above python scripts.  
overlay.py allows you to overlay foreground onto any background you desire by giving it as input an appropriate foreground matte.  
