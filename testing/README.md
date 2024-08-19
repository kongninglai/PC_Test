# PC Algorithm Testing

code to test PC algorithm (a constraint-based causal discovery algorithm).
PC algorithm Implementation: [pgmpy](https://github.com/pgmpy/pgmpy)
Data generation and result evaluation: [CausalPowerAnalysis](https://github.com/lelandwilliams/CausalPowerAnalysis)

## Environment Setup
First install all libraries
```
$ pip install -r requirements.txt
```
Then setup pgmpy and run test
```
$ cd pgmpy-dev
$ python setup.py install
$ pytest -v pgmpy
```

## Test
To test pc algorithm, first go to `testing`
```
$ cd ../testing
```

### Generate Data
To generate data, run
```
$ python generate_data.py num_var num_edge num_sample
```
`num_var`: number of variables (colums). 

`num_edge`: number of edges in the graph (recommend `1 * num_var` to `2 * num_var`). 

`num_sample`: number of samples (rows). 

The data will be saved in `testing/datasets` folder, including the sample data in `data.csv` and the edge information in `edge.txt`.
### Estimation and Evaluation
To estimate and evaluate pc algorithm, run
```
$ python pc.py num_var num_edge num_sample n_jobs
```
The result will be saved in `testing/results` folder, including estimated edges in `est_edges.txt`, the data(time and scores) in `'experiment_data.txt`, and true and estimated graph in `true.pdf`, `estimated.pdf`.

### Test 1 GB data and 10 GB data
```
$ cd testing

# generate and test 1 GB data
$ python generate_data.py 25 30 2000000
$ python pc.py 25 30 2000000 1

# generate and test 10 GB data
$ python generate_data.py 25 30 20000000
$ python pc.py 25 30 20000000 1
```

Based on my experiments, when num_var = 25, num_edges = 30, and num_sample = 2,000,000, the data size is 989.9 MB. The PC algorithm's run time is 4 minutes and 23 seconds when n_jobs = 1 (run on my Mac). I did not test with 10GB of data because my Mac has only 16GB of memory, but I believe it should complete in around 1 hour. If it takes significantly longer than 1 hour, please let me know, and I can continue to make adjustments.

### multiprocessing
The pgmpy implementation of parallel PC is using joblib for parallelism. The default is 'locky', to change to 'multiprocessing':
Open `pgmpy-dev/pgmpy/estimators/PC.py` and modify 'backend' in line 475
```
# locky
line 475: results = Parallel(n_jobs=n_jobs, backend='loky')(

# multiprocessing
line 475: results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
```
Then go to `pgmpy-dev` and setup again.
```
$ cd pgmpy-dev
$ python setup.py install
```


