"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset, VQA_Instruct_Dataset

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


class COCOVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }

class COCOVQA_Instruct_Dataset(VQA_Instruct_Dataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        Question=ann["question"]
        prompts=[f'{Question}',
        f'Question: {Question}',
        f'{Question} A short answer to the question is',
        f'Q: {Question} A:',
        f'Question: {Question} Short answer:',
        f'Given the image, answer the following question with no more than three words. {Question}',
        f'Based on the image, respond to this question with a short answer: {Question}. Answer:',
        f'Use the provided image to answer the question: {Question} Provide your answer as short as possible:',
        f'What is the answer to the following question? "{Question}"',
        f'The question "{Question}" can be answered using the image. A short answer is'
        ]
        import random
        # question=random.choice(prompts)
        question=prompts[4]
        # prompt='Based on the image, respond to this question with a short answer: [Question]. Answer:'
        # question=prompt.replace('[Question]',ann["question"])
        
        question = self.text_processor(question)

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "text_input": question,
            "text_output": answers[0],
        }



class COCOVQA_Instruct_EvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        Question=ann["question"]
        question=f'Question: {Question} Short answer:'
        # question=prompt.replace('[Question]',ann["question"])
        question = self.text_processor(question)

        # Instruct_1='What objects are in the picture'
        # Instruct_2='What are the characteristics of the objects in the picture'
        # Instruct_3='What is the relationship between the objects in the picture'
        Instruct_1='What objects are in the picture'
        Instruct_2='What color are the objects in the picture'
        Instruct_3='What are the characteristics of the objects in the picture'

        
        Instruct_1=self.text_processor(Instruct_1)
        Instruct_2=self.text_processor(Instruct_2)
        Instruct_3=self.text_processor(Instruct_3)
        
        return {
            "image": image,
            "text_input": question,
            "Instruct_1": Instruct_1,
            # "Instruct_1": question,
            "Instruct_2": Instruct_2,
            "Instruct_3": Instruct_3,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }

class OKVQA_Instruct_Dataset(VQA_Instruct_Dataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        Question=ann["question"]
        prompts=[f'{Question}',
        f'Question: {Question}',
        f'{Question} A short answer to the question is',
        f'Q: {Question} A:',
        f'Question: {Question} Short answer:',
        f'Given the image, answer the following question with no more than three words. {Question}',
        f'Based on the image, respond to this question with a short answer: {Question}. Answer:',
        f'Use the provided image to answer the question: {Question} Provide your answer as short as possible:',
        f'What is the answer to the following question? "{Question}"',
        f'The question "{Question}" can be answered using the image. A short answer is'
        ]
        import random
        # question=random.choice(prompts)
        question=prompts[4]
        # question=f'Let’s think step by step, Question: {Question} Short answer:'
        question = self.text_processor(question)
        # print(question)
        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())
        
        # Instruct_1='What objects are in the picture'
        # Instruct_2='What are the characteristics of the objects in the picture'
        # Instruct_3='What is the relationship between the objects in the picture'
        Instruct_1='What objects are in the picture'
        Instruct_2='What color are the objects in the picture'
        Instruct_3='What are the characteristics of the objects in the picture'

        
        Instruct_1=self.text_processor(Instruct_1)
        Instruct_2=self.text_processor(Instruct_2)
        Instruct_3=self.text_processor(Instruct_3)
        
        return {
            "image": image,
            "text_input": question,
            "Instruct_1": Instruct_1,
            # "Instruct_1": question,
            "Instruct_2": Instruct_2,
            "Instruct_3": Instruct_3,
            "text_output": answers[0],
        }

class COCOVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        # Instruct_1='What objects are in the picture'
        # Instruct_2='What are the characteristics of the objects in the picture'
        # Instruct_3='What is the relationship between the objects in the picture'
        Instruct_1='What objects are in the picture'
        Instruct_2='What color are the objects in the picture'
        Instruct_3='What are the characteristics of the objects in the picture'

        
        Instruct_1=self.text_processor(Instruct_1)
        Instruct_2=self.text_processor(Instruct_2)
        Instruct_3=self.text_processor(Instruct_3)
        return {
            "image": image,
            "text_input": question,
            "Instruct_1": Instruct_1,
            # "Instruct_1": question,
            "Instruct_2": Instruct_2,
            "Instruct_3": Instruct_3,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }