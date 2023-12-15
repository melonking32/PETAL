# CUDA_LAUNCH_BLOCKING=1 
# CUDA_VISIBLE_DEVICES=7
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/aurora/caption_flickr_instruct_flant5xxl_aurora.yaml
# python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/lora/caption_flickr_instruct_flant5xxl_lora.yaml