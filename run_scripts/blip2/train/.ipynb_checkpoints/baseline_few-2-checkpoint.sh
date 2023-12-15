num=8 
# python -m torch.distributed.run --nproc_per_node=${num} --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/ft/aokvqa_instruct_flant5xxl_ft.yaml

# python -m torch.distributed.run --nproc_per_node=${num}  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/ft/okvqa_instruct_flant5xl_ft.yaml
# python -m torch.distributed.run --nproc_per_node=${num} --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/ft/gqa_instruct_flant5xxl_ft.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/ft/textvqa_instruct_flant5xxl_ft.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/ft/caption_flickr_instruct_flant5xxl_ft.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/ft/textcaps_instruct_flant5xxl_ft.yaml

# mode='adapter'
# python -m torch.distributed.run --nproc_per_node=${num} --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/aokvqa_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=${num}  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/okvqa_instruct_flant5xl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=${num} --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/gqa_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/textvqa_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/caption_flickr_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/textcaps_instruct_flant5xxl_${mode}.yaml


python -m torch.distributed.run --nproc_per_node=${num} --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/lora/aokvqa_instruct_flant5xxl_lora.yaml

python -m torch.distributed.run --nproc_per_node=${num}  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/lora/okvqa_instruct_flant5xl_lora.yaml
# python -m torch.distributed.run --nproc_per_node=${num} --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/lora/gqa_instruct_flant5xxl_lora.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/lora/textvqa_instruct_flant5xxl_lora.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/lora/caption_flickr_instruct_flant5xxl_lora.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/lora/textcaps_instruct_flant5xxl_lora.yaml




mode='aurora'
python -m torch.distributed.run --nproc_per_node=${num} --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/aokvqa_instruct_flant5xxl_${mode}.yaml

python -m torch.distributed.run --nproc_per_node=${num}  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/okvqa_instruct_flant5xl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=${num} --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/gqa_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/textvqa_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/caption_flickr_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/textcaps_instruct_flant5xxl_${mode}.yaml

# mode='aurora_mixture'
# python -m torch.distributed.run --nproc_per_node=${num} --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/aokvqa_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=${num}  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/okvqa_instruct_flant5xl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=${num} --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/gqa_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/textvqa_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/caption_flickr_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=${num} train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/textcaps_instruct_flant5xxl_${mode}.yaml


# python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/aokvqa_instruct_flant5xxl_lora_blip2.yaml

# python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/aokvqa_instruct_flant5xxl_aurora_blip2.yaml

# python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/aokvqa_instruct_flant5xxl_mixture_blip2.yaml


# mode='head'
# python -m torch.distributed.run --nproc_per_node=8 --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/aokvqa_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=8  train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/okvqa_instruct_flant5xl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=8 --master_port=25641 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/gqa_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/textvqa_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/caption_flickr_instruct_flant5xxl_${mode}.yaml

# python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path /mnt/pfs/zhaiyihang/Project/LAVIS/lavis/projects/blip2/train/${mode}/textcaps_instruct_flant5xxl_${mode}.yaml
