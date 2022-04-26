# For Linear GRAND
# Dataset CORA - table. 2
python3 test_multiple_splits.py --dataset Cora --function laplacian --block attention --num_splits 10 --time 128.0 --experiment --log_file 1_tbl_2_lr_grand_cora.csv

# Dataset CITESEER - table. 2
python3 test_multiple_splits.py --dataset Citeseer --function laplacian --block attention --num_splits 10 --time 128.0 --experiment --log_file 2_tbl_2_lr_grand_citeseer.csv

# Dataset PUBMet - table. 2
python3 test_multiple_splits.py --dataset Pubmed --function laplacian --block attention --num_splits 10 --time 128.0 --experiment --log_file 3_tbl_2_lr_grand_pubmed.csv

# For Non-Linear GRAND
# Dataset CORA - table. 2
python3 test_multiple_splits.py --dataset Cora --function transformer --block constant --num_splits 10 --time 128.0 --experiment --log_file 4_tbl_2_nlr_grand_cora.csv

# Dataset CITESEER - table. 2
python3 test_multiple_splits.py --dataset Citeseer --function transformer --block constant --num_splits 10 --time 128.0 --experiment --log_file 5_tbl_2_nlr_grand_citeseer.csv

# Dataset PUBMed - table. 2
python3 test_multiple_splits.py --dataset Pubmed --function transformer --block constant --num_splits 10 --time 128.0 --experiment --log_file 6_tbl_2_nlr_grand_pubmet.csv

# For Linear DeepGRAND
# Dataset CORA - table. 2
python3 test_multiple_splits.py --function ext_laplacian3\
			  --dataset Cora\
                          --block attention \
                          --experiment \
                          --max_iters 1000\
                          --time 128.0\
                          --max_nfe 100000000000000\
                          --alpha_ 3.0\
                          --clip_bound 0.4\
                          --num_splits 10\
                          --l1_weight_decay 0.0\
                          --decay 0.0001\
                          --dropout 0\
			  --log_file 7_tbl_2_lr_deepgrand_cora.csv

# Dataset CITESEER - table. 2
python3 test_multiple_splits.py --function ext_laplacian3\
			  --dataset Citeseer\
                          --block attention \
			  --attention_type scaled_dot\
                          --experiment \
                          --max_iters 1000\
                          --time 128.0\
                          --max_nfe 100000000000000\
                          --alpha_ 3.0\
                          --clip_bound 0.4\
                          --num_splits 10\
                          --l1_weight_decay 0.0\
                          --decay 0.0001\
                          --dropout 0\
			  --log_file 8_tbl_2_lr_deepgrand_citeseer.csv

# Dataset PUBMed - table. 2
python3 test_multiple_splits.py --function ext_laplacian3\
			  --dataset Pubmed\
                          --block attention \
                          --experiment \
                          --max_iters 1000\
                          --time 128.0\
                          --max_nfe 100000000000000\
                          --alpha_ 3.0\
                          --clip_bound 0.4\
                          --num_splits 10\
                          --l1_weight_decay 0.0\
                          --decay 0.0001\
                          --dropout 0\
			  --log_file 9_tbl_2_lr_deepgrand_pubmed.csv

