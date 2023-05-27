#python darcy_hzk.py --gpu 1 --n-hidden 96 --n-layers 3 --lr 0.001 --use_tb 1
#python darcy_hzk.py --gpu 0 --model ONO --n-hidden 128 --n-layers 5 --lr 0.001 --use_tb 1

#python darcy_hzk.py --gpu 1 --model ONO2  --n-hidden 96 --n-layers 3 --ortho 0 --lr 0.001 --use_tb 1
#python darcy_hzk.py --gpu 1 --model ONO2  --n-hidden 96 --n-heads 1 --n-layers 3 --ortho 0 --lr 0.001 --use_tb 1
#python darcy_hzk.py --gpu 0 --model ONO2  --n-hidden 96 --n-heads 4 --n-layers 3 --ortho 0 --lr 0.001 --use_tb 1
#python darcy_hzk.py --gpu 1 --model CGPT  --n-hidden 96 --n-heads 1 --n-layers 4 --ortho 0 --lr 0.001 --use_tb 1
#python darcy_hzk.py --gpu 2 --model CGPT  --n-hidden 128 --n-heads 4 --n-layers 4 --ortho 0 --lr 0.001 --use_tb 1
#
#python darcy_hzk.py --gpu 4 --model ONO2  --n-hidden 96 --n-heads 1 --n-layers 4 --ortho 0 --lr 0.001 --use_tb 1

python darcy_hzk.py --gpu 1 --model ONO2  --n-hidden 96 --n-heads 1 --n-layers 4 --ortho 1 --lr 0.001 --use_tb 1


