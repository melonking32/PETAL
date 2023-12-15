# CUDA_VISIBLE_DEVICES=1,2,3,4,6
# echo '################ coco caption base'
# python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/caption_coco_instruct.yaml
# echo '################ coco caption lora'
# python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/caption_coco_instruct-lora.yaml
echo '################ coco caption aurora'
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/caption_coco_instruct-aurora.yaml