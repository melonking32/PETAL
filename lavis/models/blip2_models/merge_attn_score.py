import torch
score=torch.load('/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/models/blip2_models/attn_score_instruction_enhanced.pt')
with open('/mnt/pfs/zhaiyihang/Project/LAVIS/lavis/models/blip2_models/flickr_image_id_aurora_mixture_instruct','r') as f:
    ids=f.readlines()
print(score.keys())
for i in range(1,19):
    print(score[i])
assert len(score.keys())//18==len(ids)
res={}
for i in range(len(ids)):
    dicts_1={}
    for k in range(3):
        dicts_2={}
        for j in range(6):
            dicts_2['layer'+str(j)]=score[18*i+k*6+j+1]
        dicts_1['Instruct'+str(k)]=dicts_2
    res[ids[i].replace('flickr30k-images/','').replace('\n','')]=dicts_1
print(res.keys())
print(res['1009434119.jpg'])

torch.save(res,'instruction_enhanced_flickr_attention_score.pt')


