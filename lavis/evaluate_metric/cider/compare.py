from cider import Cider
import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

  #coco caption
gts_path='/mnt/pfs/zhaiyihang/Project/Lavis_data/flickr30k/annotations/test.json'
gts_dict={}
ids=0
with open (gts_path,'r') as f:
    data1=json.load(f)
    for d in data1:
        gts_dict[ids]=d['caption']
        ids+=1

res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBlip/Caption_flickr_instruct/20230912230/result/test_epoch4.json'  # vital
res_dict_vital={}
with open (res_path,'r') as f:
    data=json.load(f)
    for d in data:
        res_dict_vital[d['image_id']]=[d['caption']]
# print(len(res_dict.keys()))
# print(res_dict.keys())

res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBlip/Caption_flickr_instruct/20230912131/result/test_epoch4.json'  #ft
res_dict_ft={}
with open (res_path,'r') as f:
    data=json.load(f)
    for d in data:
        res_dict_ft[d['image_id']]=[d['caption']]

res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBlip/Caption_flickr_instruct/20230912130/result/test_epoch4.json'  # lora
res_dict_lora={}
with open (res_path,'r') as f:
    data=json.load(f)
    for d in data:
        res_dict_lora[d['image_id']]=[d['caption']]

res_path='/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/output/InstructBlip/Caption_flickr_instruct/20230912045/result/test_epoch4.json'  # adapter
res_dict_adapter={}
with open (res_path,'r') as f:
    data=json.load(f)
    for d in data:
        res_dict_adapter[d['image_id']]=[d['caption']]

# print(res_dict_vital['caef3d1e97c2cf15'])

eva=Cider()
rouge = Rouge()
# print(eva.compute_score(gts_dict,res_dict_vital))
# print(gts_dict)
# print(res_dict_vital)
for key in gts_dict.keys():
    if '5489602545.jpg' in data1[key]['image']:
        print(data1[key]['image'])
        print(gts_dict[key][0])
        print(res_dict_ft[key])
        print(res_dict_adapter[key])
        print(res_dict_lora[key])
        print(res_dict_vital[key])
        print('###########')
    
    # rouge_score = rouge.get_scores(hyps=[gts_dict[key]], refs=res_dict_lora[key])
    # print(rouge_score)
    # print(cider1,cider2)
