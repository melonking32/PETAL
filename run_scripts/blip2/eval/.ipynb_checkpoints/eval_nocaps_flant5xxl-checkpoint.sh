CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip2/eval/nocaps_eval.yaml
