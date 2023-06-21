# ONO
Requirement:
python==3.7.16
`pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`

`python -u Burger_example.py --gpu 4 --model ONO2  --n-hidden 128 --n-heads 8 --n-layers 5 --lr 0.001 --use_tb 1 --attn_type nystrom --max_grad_norm 0.1 --orth 1 --psi_dim 16 --downsample 32 --batch-size 8`

rel_err:0.00876

`python -u Burger_example.py --gpu 3 --model ONO2  --n-hidden 128 --n-heads 8 --n-layers 5 --lr 0.001 --use_tb 1 --attn_type nystrom --max_grad_norm 0.1 --orth 1 --psi_dim 16 --downsample 16 --batch-size 8`

rel_err:0.01168

`python -u Burger_example.py --gpu 2 --model ONO2  --n-hidden 128 --n-heads 8 --n-layers 5 --lr 0.001 --use_tb 1 --attn_type nystrom --max_grad_norm 0.1 --orth 1 --psi_dim 16 --downsample 8 --batch-size 8`

rel_err:0.01259

`python -u Burger_example.py --gpu 1 --model ONO2  --n-hidden 128 --n-heads 8 --n-layers 5 --lr 0.001 --use_tb 1 --attn_type nystrom --max_grad_norm 0.1 --orth 1 --psi_dim 16 --downsample 4 --batch-size 8`

rel_err:0.01255

`python -u Burger_example.py --gpu 0 --model ONO2  --n-hidden 128 --n-heads 8 --n-layers 5 --lr 0.001 --use_tb 1 --attn_type nystrom --max_grad_norm 0.1 --orth 1 --psi_dim 16 --downsample 2 --batch-size 8`

rel_err:

`python -u Burger_example.py --gpu 3 --model ONO2  --n-hidden 128 --n-heads 8 --n-layers 5 --lr 0.001 --use_tb 1 --attn_type nystrom --max_grad_norm 0.1 --orth 1 --psi_dim 16 --downsample 1 --batch-size 8`

rel_err:0.01115
