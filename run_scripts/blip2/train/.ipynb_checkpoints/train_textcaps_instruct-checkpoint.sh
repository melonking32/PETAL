# CUDA_LAUNCH_BLOCKING=1 
# CUDA_VISIBLE_DEVICES=1,2,3,4,6 
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/aurora/textcaps_instruct_flant5xxl_aurora.yaml
# python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/lora/textcaps_instruct_flant5xxl_lora.yaml
# python -m torch.distributed.run --nproc_per_node=8  --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/ft/textcaps_instruct_flant5xxl_ft.yaml
# python -m torch.distributed.run --nproc_per_node=8  --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/adapter/textcaps_instruct_flant5xxl_adapter.yaml
# python -m torch.distributed.run --nproc_per_node=8  --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/head/textcaps_instruct_flant5xxl_head.yaml