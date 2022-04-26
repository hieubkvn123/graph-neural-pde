# For Linear GRAND
# Dataset CORA - table. 1
# python3 test_multiple_planetoid_splits.py --dataset Cora --function laplacian --block attention --epoch 100 --num_splits 10 --time 128.0 --experiment --log_file 1_tbl_1_lr_grand_cora.json

# Dataset CITESEER - table. 1
python3 test_multiple_planetoid_splits.py --dataset Citeseer --function laplacian --block attention --epoch 100 --num_splits 10 --time 128.0 --experiment --log_file 2_tbl_1_lr_grand_citeseer.json

# Dataset PUBMet - table. 1
# python3 test_multiple_planetoid_splits.py --dataset Pubmed --function laplacian --block attention --epoch 100 --num_splits 10 --time 128.0 --experiment --log_file 3_tbl_1_lr_grand_pubmed.json

# For Non-Linear GRAND
# Dataset CORA - table. 1
python3 test_multiple_planetoid_splits.py --dataset Cora --function transformer --block constant --epoch 100 --num_splits 10 --time 128.0 --experiment --log_file 4_tbl_1_nlr_grand_cora.json

# Dataset CITESEER - table. 1
python3 test_multiple_planetoid_splits.py --dataset Citeseer --function transformer --block constant --epoch 100 --num_splits 10 --time 128.0 --experiment --log_file 5_tbl_1_nlr_grand_citeseer.json

# Dataset PUBMed - table. 1
# python3 test_multiple_planetoid_splits.py --dataset Pubmed --function transformer --block constant --epoch 100 --num_splits 10 --time 128.0 --experiment --log_file 6_tbl_1_nlr_grand_pubmet.json

# For Linear DeepGRAND
# Dataset CORA - table. 1
python3 test_multiple_planetoid_splits.py --function ext_laplacian3\
			  --dataset Cora\
			  --epoch 100\
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
			  --log_file 7_tbl_1_lr_deepgrand_cora.json

# Dataset CITESEER - table. 1
python3 test_multiple_planetoid_splits.py --function ext_laplacian3\
			  --dataset Citeseer\
			  --epoch 100\
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
			  --log_file 8_tbl_1_lr_deepgrand_citeseer.json

# Dataset PUBMed - table. 1
#python3 test_multiple_planetoid_splits.py --function ext_laplacian3\
#			  --dataset Pubmed\
#			  --epoch 100\
#                          --block attention \
#                          --experiment \
#                          --max_iters 1000\
#                          --time 128.0\
#                          --max_nfe 100000000000000\
#                          --alpha_ 3.0\
#                          --clip_bound 0.4\
#                          --num_splits 10\
#                          --l1_weight_decay 0.0\
#                          --decay 0.0001\
#                          --dropout 0\
#			  --log_file 9_tbl_1_lr_deepgrand_pubmed.json

