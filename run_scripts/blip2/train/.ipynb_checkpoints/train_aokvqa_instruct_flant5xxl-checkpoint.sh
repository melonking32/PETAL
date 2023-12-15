# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 

python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/aurora/caption_flickr_instruct_flant5xxl_aurora.yaml

python -m torch.distributed.run --nproc_per_node=1  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/aurora/aokvqa_instruct_flant5xxl_aurora.yaml
# python -m torch.distributed.run --nproc_per_node=1  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/aurora/attn_score_flickr.yaml
# python -m torch.distributed.run --nproc_per_node=8  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/ft/aokvqa_instruct_flant5xxl_ft.yaml

# python -m torch.distributed.run --nproc_per_node=8 --master_port=25641  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/head/aokvqa_instruct_flant5xxl_head.yaml

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=7 --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/lora/aokvqa_instruct_flant5xxl_lora.yaml
