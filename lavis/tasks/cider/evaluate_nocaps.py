from cider import Cider
if __name__ == '__main__':
    # res_path='/blob/v-yihangzhai/LAVIS/lavis/output/BLIP/NoCaps/20230704151/result/val_epochbest.json'  # zero shot
    # res_path='/blob/v-yihangzhai/LAVIS/lavis/output/BLIP/NoCaps_Finetune/20230705050/result/val_epochbest.json' #coco caption+4qa finetune 1epoch
#     res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/BLIP/NoCaps_Finetune/20230712195/result/val_epochbest.json' #coco caption
    # res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBLIP/NoCaps_Finetune/20230720172/result/val_epochbest.json'
   #  res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBlip/NoCaps_Finetune/20230728200/result/val_epochbest.json'
    # res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBlip/NoCaps_Finetune/20230728202/result/val_epochbest.json'
   # res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBlip/NoCaps_Finetune/20230728204/result/val_epochbest.json'
   #  res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBlip/NoCaps_Finetune/20230728205/result/val_epochbest.json'
   # res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBlip/NoCaps_Finetune/20230728211/result/val_epochbest.json'
    # res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBlip/Caption_flickr_instruct/20230825175/result/test_epoch1.json'
    # gts_path='/mnt/pfs/zhaiyihang/Project/Lavis_data/nocaps/nocaps_val_4500_captions.json'
    # import json
    # res_dict={}
    # with open (res_path,'r') as f:
    #     data=json.load(f)
    #     for d in data:
    #         res_dict[d['image_id']]=[d['caption']]

    # gts_dict={}
    # with open (gts_path,'r') as f:
    #     data=json.load(f)
    #     for ann in data['annotations']:
    #         if ann['image_id'] not in gts_dict:
    #             gts_dict[ann['image_id']]=[ann['caption']]
    #         else:
    #             gts_dict[ann['image_id']].append(ann['caption'])
    # eva=Cider(n=4,sigma=6.0)
    # print(eva.compute_score(gts_dict,res_dict))
    # res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBlip/Caption_flickr_instruct/20230909211/result/test_epoch4.json'   #coco caption
    # gts_path='/mnt/pfs/zhaiyihang/Project/Lavis_data/flickr30k/annotations/test.json'
    # import json
    # res_dict={}
    # with open (res_path,'r') as f:
    #     data=json.load(f)
    #     for d in data:
    #         res_dict[d['image_id']]=[d['caption']]
    # print(len(res_dict.keys()))
    # # print(res_dict.keys())
    # gts_dict={}
    # ids=0
    # with open (gts_path,'r') as f:
    #     data=json.load(f)
    #     for d in data:
    #         gts_dict[ids]=d['caption']
    #         ids+=1
    # print(len(gts_dict.keys()))
    # print(gts_dict.keys())


    res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBlip/Caption_flickr_instruct/20230910101/result/val_epoch0.json'
    gts_path='/mnt/pfs/zhaiyihang/Project/Lavis_data/textcaps/annotations/textcaps_val.json'
    import json
    res_dict={}
    with open (res_path,'r') as f:
        data=json.load(f)
        for d in data:
            res_dict[d['image_id']]=[d['caption']]
    print(len(res_dict.keys()))
    # print(res_dict.keys())
    gts_dict={}
    ids=0
    with open (gts_path,'r') as f:
        data=json.load(f)
        for d in data:
            gts_dict[d['image_id']]=d['reference_strs']
            ids+=1
    print(len(gts_dict.keys()))

    
    eva=Cider()
    print(eva.compute_score(gts_dict,res_dict))    
    # print(res_dict[0])
    # print(gts_dict[0])
