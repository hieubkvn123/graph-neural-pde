# Table .1 - DeepGRAND on multiple planetoid splits
python3 test_multiple_planetoid_splits.py --dataset Pubmed --function ext_laplacian3 --block attention --alpha_ 3.0 --clip_bound 0.4 --decay 0.0001 --epoch 100 --num_splits 10 --time 128.0 --experiment --log_file 9_tbl_1_lr_deepgrand_pubmed.json

# Table .2 - DeepGRAND on multople random splits
python3 test_multiple_splits.py --dataset Computers --function ext_laplacian3 --block attention --alpha_ 3.0 --clip_bound 0.4 --decay 0.0001 --epoch 100 --num_splits 10 --time 128.0 --experiment --log_file 16_tbl_2_lr_deepgrand_computers.json
