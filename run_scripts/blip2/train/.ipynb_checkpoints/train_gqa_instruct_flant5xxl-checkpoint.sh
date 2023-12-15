# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 
python -m torch.distributed.run --nproc_per_node=8 --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/aurora/gqa_instruct_flant5xxl_aurora.yaml
# python -m torch.distributed.run --nproc_per_node=8 --master_port=25641 train.py --cfg-path lavis/projects/blip2/train/gqa_instruct_ft_flant5xxl_aurora.yaml
