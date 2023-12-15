num=8
mode='aurora_mixture'  # lora ft adapter head bias
python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path lavis/projects/blip2/train/${mode}/aokvqa_instruct_flant5xxl_${mode}.yaml

python -m torch.distributed.run --nproc_per_node=${num}  train.py --cfg-path lavis/projects/blip2/train/${mode}/okvqa_instruct_flant5xxl_${mode}.yaml

python -m torch.distributed.run --nproc_per_node=${num}  train.py --cfg-path lavis/projects/blip2/train/${mode}/gqa_instruct_flant5xxl_${mode}.yaml

python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path lavis/projects/blip2/train/${mode}/textvqa_instruct_flant5xxl_${mode}.yaml

python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path lavis/projects/blip2/train/${mode}/caption_flickr_instruct_flant5xxl_${mode}.yaml

python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path lavis/projects/blip2/train/${mode}/textcaps_instruct_flant5xxl_${mode}.yaml

