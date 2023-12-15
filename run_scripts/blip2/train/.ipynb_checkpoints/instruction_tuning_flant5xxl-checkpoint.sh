# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/instruction_tuning.yaml
