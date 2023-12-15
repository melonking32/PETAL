# CUDA_VISIBLE_DEVICES=1,3,4,6
echo '################ coco caption base'
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/caption_coco_instruct.yaml
echo '################ coco caption lora'
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/caption_coco_instruct-lora.yaml
echo '################ coco caption aurora'
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/caption_coco_instruct-aurora.yaml

echo '################ vqa base'
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/vqav2_instruct_flant5xxl.yaml
echo '################ vqa lora'
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/vqav2_instruct_flant5xxl-lora.yaml
echo '################ vqa aurora'
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/vqav2_instruct_flant5xxl-aurora.yaml