CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=55555 dist_train.py --dataset half_CROHME