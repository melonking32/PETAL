"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset,VQA_Instruct_Dataset
from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class GQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answers = [ann["answer"]]
        weights = [1]

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }


class GQA_Instruct_Dataset(VQA_Instruct_Dataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        Question=ann["question"]
        question=f'Question: {Question} Short answer:'
        question = self.text_processor(question)
        
        Instruct_1='What objects are in the picture'
        Instruct_2='What are the characteristics of the objects in the picture'
        Instruct_3='What is the relationship between the objects in the picture'

        
        Instruct_1=self.text_processor(Instruct_1)
        Instruct_2=self.text_processor(Instruct_2)
        Instruct_3=self.text_processor(Instruct_3)
        answers = ann["answer"]

        return {
            "image": image,
            "Instruct_1": Instruct_1,
            # "Instruct_1": question,
            "Instruct_2": Instruct_2,
            "Instruct_3": Instruct_3,
            "text_input": question,
            "text_output": answers
        }


class GQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. gqa/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        ## TODO: support inference method == 'ranking'
        answer_list_path = ann_paths[1] if len(ann_paths) > 1 else ''
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        Instruct_1='What objects are in the picture'
        Instruct_2='What are the characteristics of the objects in the picture'
        Instruct_3='What is the relationship between the objects in the picture'

        
        Instruct_1=self.text_processor(Instruct_1)
        Instruct_2=self.text_processor(Instruct_2)
        Instruct_3=self.text_processor(Instruct_3)
        if "answer" in ann:
            # answer is a string
            answer = ann["answer"]
        else:
            answer = None

        return {
            "image": image,
            "text_input": question,
            "Instruct_1": Instruct_1,
            # "Instruct_1": question,
            "Instruct_2": Instruct_2,
            "Instruct_3": Instruct_3,
            "answer": answer,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }
