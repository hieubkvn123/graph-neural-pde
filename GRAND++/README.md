# GRAND++

This is a code base for GRAND++, submitted to ICLR 2022. The code is based on GRAND (https://arxiv.org/pdf/2106.10934.pdf).

To run GRAND++, you'll need 

python src/run_GNP.py --trusted_mask --nox0 --icxb 1

the option --no_early is not supported.

Major command line arguments

1. Datasets: --dataset Cora
2. Linear / nonlinear: --block attention --function laplacian / --block constant --function transformer
3. Train sample size: --num_train_per_class 5
4. Time of integration (depth): --time 16.0
5. Scaling of source: --source_scale 1.0
6. Rewiring (nonlinear only): --rewiring gdc
7. Multiple runs: --run_num 100

The imputation of GRAND++ modification into GRAND code base is on numerical solver level. Please refrain from changing numerical solver which may cause unexpected errors.
