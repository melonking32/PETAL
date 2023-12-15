#CUDA_VISIBLE_DEVICES=1,2,3,4,6 
python -m torch.distributed.run --nproc_per_node=8  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/aurora/okvqa_instruct_flant5xl_aurora.yaml
