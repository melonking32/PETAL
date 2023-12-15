"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask

import sys
sys.path.append("/root/paddlejob/workspace/zhaiyihang/Project/LAVIS/lavis/tasks/cider")
from cider import Cider
from caption_metric import compute_metric

@registry.register_task("captioning_textcaps")
class Caption_textcaps_Task(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        # print('samples',samples)
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        img_ids = samples["image_id"]
        for caption, img_id in zip(captions, img_ids):
            results.append({"caption": caption, "image_id": img_id})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        import json
        from nltk.translate.bleu_score import sentence_bleu
        # TODO better way to define this
        with open('/root/paddlejob/workspace/zhaiyihang/Project/Lavis_data/textcaps/annotations/textcaps_val.json') as f:
            gt_caption=json.load(f)
        gts={}
        for g in gt_caption:
            gts[g['image_id']]=g['reference_strs']
        outputs={}
        print(eval_result_file)
        with open(eval_result_file) as f:
            output_caption=json.load(f)
        for o in output_caption:
            outputs[o['image_id']]=o['caption']
        
        assert len(outputs.keys())==len(gts.keys())
        all_score=0
        for k in outputs.keys():
            caption_out=outputs[k].split()
            caption_gt=[gt_item.split() for gt_item in gts[k]]
            # print(caption_gt)
            # print(caption_out)
            score = sentence_bleu(caption_gt, caption_out)
            all_score+=score


        res_path=eval_result_file
        gts_path='/root/paddlejob/workspace/zhaiyihang/Project/Lavis_data/textcaps/annotations/textcaps_val.json'
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
    
        eva_res=compute_metric(gts_dict,res_dict)
        with open(os.path.join(registry.get_path("output_dir").split('2')[0], "evaluate.txt"),'a') as f:
            f.write(json.dumps(eva_res)+'\n')
        # eva=Cider()
        # print(eva.compute_score(gts_dict,res_dict))   

        
        # print('Bleu: ',all_score/len(gts.keys()))
        return {'agg_metrics':eva_res['Cider']}



# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval
