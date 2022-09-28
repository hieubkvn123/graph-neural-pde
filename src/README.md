# DeepGRAND : Deeper Neural Diffusion
This is the code for DeepGRAND's submission for ICLR 2023. This code is based on the implementation of Graph Neural Diffusion (https://arxiv.org/pdf/2106.10934.pdf).

## 1. Single run
To run a single run with DeepGRAND, you will need to execute:
```
python3 run_GNN.py --dataset [Cora|Citeseer|Pubmed|CoauthorCS|Computers|Photo]
		--block attention
		--function ext_laplacian3
		--time [time]
		--alpha_ [alpha]
		--epsilon_ [epsilon]
		--epoch [epoch]
		--experiment
```

Major command line options:
	- dataset : The benchmark used to run DeepGRAND. Select among : Cora, Citeseer, Pubmed, Computers, Photo and CoauthorCS.
	- time : Integration limit T.
	- alpha_ : The exponential alpha specified in the dynamics of DeepGRAND.
	- epsilon_ : The epsilon value specified in the dynamics of DeepGRAND.
	- epoch : Number of training iterations.

## 2. Ablation study
To re-produce the ablation study results specified in table 1, 2, 3. Run the following commands:

### 2.1. Ablation study - depth
```
python3 ablation_study_deepgrand_depth.py --experiment_set [experiment_set]
```

## 2.2. Ablation study - label rate
```
python3 ablation_study_deepgrand_labelrate.py --experiment_set [experiment_set]
```

Where the experiment_set argument specify the sets of benchmarks to run ablation study on.
	- experiment_set 0 : For Cora, Citeseer and Pubmed.
	- experiment_set 1 : For Computers, Photo and CoauthorCS.
