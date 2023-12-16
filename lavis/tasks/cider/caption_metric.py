from .cider import Cider
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

def compute_metric(gts_dict,res_dict):
    eva=Cider()
    cider_score=eva.compute_score(gts_dict,res_dict)[0]
    
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
        Rogue_score_1[kk]=round(Rogue_score_1[kk]/len(res_dict.keys())*100,2)
        Rogue_score_2[kk]=round(Rogue_score_2[kk]/len(res_dict.keys())*100,2)
        Rogue_score_l[kk]=round(Rogue_score_l[kk]/len(res_dict.keys())*100,2)
        
    
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

    result_dict={}
    result_dict['Cider']=round(cider_score*100,2)
    result_dict['Bleu_1']=round(bleu_1/len(res_dict.keys())*100,2)
    result_dict['Bleu_2']=round(bleu_2/len(res_dict.keys())*100,2)
    result_dict['Bleu_3']=round(bleu_3/len(res_dict.keys())*100,2)
    result_dict['Bleu_4']=round(bleu_4/len(res_dict.keys())*100,2)
    result_dict['Rogue_1']=Rogue_score_1
    result_dict['Rogue_2']=Rogue_score_1
    result_dict['Rogue_l']=Rogue_score_1

    print(result_dict)
    # print('Cider: ',eva.compute_score(gts_dict,res_dict)[0])    
    # print('Bleu_1:',bleu_1/len(res_dict.keys()))
    # print('Bleu_2:',bleu_2/len(res_dict.keys()))
    # print('Bleu_3:',bleu_3/len(res_dict.keys()))
    # print('Bleu_4:',bleu_4/len(res_dict.keys()))
    # print('Rogue_1:',Rogue_score_1)
    # print('Rogue_2:',Rogue_score_2)
    # print('Rogue_l:',Rogue_score_l)

    return (result_dict)