# For Linear GRAND
# Dataset COAUTHORCS - table. 2
# python3 test_multiple_splits.py --dataset CoauthorCS --function laplacian --block attention --epoch 250 --num_splits 10 --time 128.0 --experiment --log_file 10_tbl_2_lr_grand_coauthorcs.csv

# Dataset PHOTO - table. 2
# python3 test_multiple_splits.py --dataset Photo --function laplacian --block attention --num_splits 10 --time 128.0 --experiment --log_file 11_tbl_2_lr_grand_photo.csv

# For Non-Linear GRAND
# Dataset COAUTHORCS - table. 2
# python3 test_multiple_splits.py --dataset CoauthorCS --function transformer --block constant --epoch 250 --num_splits 10 --time 128.0 --experiment --log_file 12_tbl_2_nlr_grand_coauthorcs.csv

# Dataset PHOTO - table. 2
# python3 test_multiple_splits.py --dataset Photo --function transformer --block constant --num_splits 10 --time 128.0 --experiment --log_file 13_tbl_2_nlr_grand_photo.csv

# For Linear DeepGRAND
# Dataset COAUTHORCS - table. 2
python3 test_multiple_splits.py --function ext_laplacian3\
			  --dataset CoauthorCS\
                          --block attention \
                          --experiment \
                          --max_iters 1000\
                          --time 128.0\
                          --max_nfe 100000000000000\
                          --alpha_ 3.0\
                          --clip_bound 0.4\
			  --epoch 300\
                          --num_splits 10\
                          --l1_weight_decay 0.0\
                          --decay 0.0001\
                          --dropout 0\
			  --log_file 14_tbl_2_lr_deepgrand_coauthorcs.csv

# Dataset PHOTO - table. 2
python3 test_multiple_splits.py --function ext_laplacian3\
			  --dataset Photo\
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
			  --log_file 15_tbl_2_lr_deepgrand_photo.csv

