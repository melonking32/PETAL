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
    res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBlip/Caption_Textcaps_instruct/20230912053/result/val_epoch4.json'   #coco caption
    gts_path='/mnt/pfs/zhaiyihang/Project/Lavis_data/flickr30k/annotations/test.json'
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
            gts_dict[ids]=d['caption']
            ids+=1
    # print(len(gts_dict.keys()))
    # print(gts_dict.keys())

    res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBlip/Caption_Textcaps_instruct/20230912053/result/val_epoch0.json'
    gts_path='/mnt/pfs/zhaiyihang/Project/Lavis_data/textcaps/annotations/textcaps_val.json'
    import json
    res_dict={}
    with open (res_path,'r') as f:
        data=json.load(f)
        for d in data:
            res_dict[d['image_id']]=[d['caption']]
    print(len(res_dict.keys()))
    # # print(res_dict.keys())
    gts_dict={}
    ids=0
    with open (gts_path,'r') as f:
        data=json.load(f)
        for d in data:
            gts_dict[d['image_id']]=d['reference_strs']
            ids+=1
    print(len(gts_dict.keys()))

    
    eva=Cider()
    print('Cider: ',eva.compute_score(gts_dict,res_dict)[0])    


    
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.bleu_score import corpus_bleu
    from rouge import Rouge
    
    sent_score=0
    corpus_score=0
    bleu_1,bleu_2,bleu_3,bleu_4=0,0,0,0
    Rogue_score_1={'f':0,'p':0,'r':0}
    Rogue_score_2={'f':0,'p':0,'r':0}
    Rogue_score_l={'f':0,'p':0,'r':0}
    
    for k in res_dict.keys():
        candidate=[res_dict[k][0]]
        reference=gts_dict[k]
        # print(candidate,reference)
        rouge = Rouge()
        rouge_sent_score_1={'f':0,'p':0,'r':0}
        rouge_sent_score_2={'f':0,'p':0,'r':0}
        rouge_sent_score_l={'f':0,'p':0,'r':0}
        for re in reference:
            rouge_score = rouge.get_scores(hyps=candidate, refs=[re])
            for kk in rouge_sent_score_1.keys():
                rouge_sent_score_1[kk]+=rouge_score[0]["rouge-1"][kk]/len(reference)
                rouge_sent_score_2[kk]+=rouge_score[0]["rouge-2"][kk]/len(reference)
                rouge_sent_score_l[kk]+=rouge_score[0]["rouge-l"][kk]/len(reference)
        for kk in rouge_sent_score_1.keys():
            Rogue_score_1[kk]+=rouge_sent_score_1[kk]
            Rogue_score_2[kk]+=rouge_sent_score_2[kk]
            Rogue_score_l[kk]+=rouge_sent_score_l[kk]
    for kk in rouge_sent_score_1.keys():
        Rogue_score_1[kk]/=len(res_dict.keys())
        Rogue_score_2[kk]/=len(res_dict.keys())
        Rogue_score_l[kk]/=len(res_dict.keys())
        
    
    for k in res_dict.keys():
        caption_out=res_dict[k][0].split()
        caption_gt=[gt_item.split() for gt_item in gts_dict[k]]
        # print(caption_gt)
        # print(caption_out)
        # print(caption_gt, caption_out)
        score_1 = sentence_bleu(caption_gt, caption_out,weights=(1, 0, 0, 0))
        score_2 = sentence_bleu(caption_gt, caption_out,weights=(0.5, 0.5, 0, 0))
        score_3 = sentence_bleu(caption_gt, caption_out,weights=(0.33, 0.33, 0.33, 0))
        score_4 = sentence_bleu(caption_gt, caption_out,weights=(0.25, 0.25, 0.25, 0.25))
        
        bleu_1+=score_1
        bleu_2+=score_2
        bleu_3+=score_3
        bleu_4+=score_4
    
    print('Cider: ',eva.compute_score(gts_dict,res_dict)[0])    
    print('Bleu_1:',bleu_1/len(res_dict.keys()))
    print('Bleu_2:',bleu_2/len(res_dict.keys()))
    print('Bleu_3:',bleu_3/len(res_dict.keys()))
    print('Bleu_4:',bleu_4/len(res_dict.keys()))
    print('Rogue_1:',Rogue_score_1)
    print('Rogue_2:',Rogue_score_2)
    print('Rogue_l:',Rogue_score_l)
    

        # corpus_score+= corpus_bleu([caption_gt], [caption_out])
        # print(score)
    
    # print(sent_score/len(res_dict.keys()))
    # print(corpus_score/len(res_dict.keys()))
    # print(res_dict[0])
    # print(gts_dict[0])
