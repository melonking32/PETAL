CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip2/train/caption_coco_ft_instruct_flant5xl.yaml