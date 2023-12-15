mode='ft'
num=8
python -m torch.distributed.run --nproc_per_node=${num}  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/aokvqa_instruct_flant5xxl_${mode}.yaml

mode='lora'
python -m torch.distributed.run --nproc_per_node=${num}  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/aokvqa_instruct_flant5xxl_${mode}.yaml

mode='adapter'
python -m torch.distributed.run --nproc_per_node=${num}  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/aokvqa_instruct_flant5xxl_${mode}.yaml

mode='aurora'
python -m torch.distributed.run --nproc_per_node=${num}  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/aokvqa_instruct_flant5xxl_${mode}.yaml