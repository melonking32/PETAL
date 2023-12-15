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


class COCO_Question_Dataset(VQA_Instruct_Dataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        Answer=answers[0]
        question=ann["question"]
        prompts=[
            f'Given the image, generate a question whose answer is: {Answer}. Question:',
            f'Based on the image, provide a question with the answer: {Answer}. Question:',
            f'Given the visual representation, create a question for which the answer is "{Answer}".',
            f'From the image provided, craft a question that leads to the reply: {Answer}. Question:',
            f'Considering the picture, come up with a question where the answer is: {Answer}.',
            f'Taking the image into account, generate an question that has the answer: {Answer}. Question:'
        ]
        import random
        prompt=random.choice(prompts)
        prompt = self.text_processor(prompt)
        return {
            "image": image,
            "text_input": prompt,
            "text_output": question,
        }

class OKVQA_Question_Dataset(VQA_Instruct_Dataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        Answer=answers[0]
        question=ann["question"]
        prompts=[
            f'Given the image, generate a question whose answer is: {Answer}. Question:',
            f'Based on the image, provide a question with the answer: {Answer}. Question:',
            f'Given the visual representation, create a question for which the answer is "{Answer}".',
            f'From the image provided, craft a question that leads to the reply: {Answer}. Question:',
            f'Considering the picture, come up with a question where the answer is: {Answer}.',
            f'Taking the image into account, generate an question that has the answer: {Answer}. Question:'
        ]
        import random
        prompt=random.choice(prompts)
        prompt = self.text_processor(prompt)
        return {
            "image": image,
            "text_input": prompt,
            "text_output": question,
        }

class AOKVQA_Question_Dataset(VQA_Instruct_Dataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        answer_key = "direct_answers"

        answer_weight = {}
        for answer in ann[answer_key]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann[answer_key])
            else:
                answer_weight[answer] = 1 / len(ann[answer_key])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        Answer=answers[0]
        question=ann["question"]
        prompts=[
            f'Given the image, generate a question whose answer is: {Answer}. Question:',
            f'Based on the image, provide a question with the answer: {Answer}. Question:',
            f'Given the visual representation, create a question for which the answer is "{Answer}".',
            f'From the image provided, craft a question that leads to the reply: {Answer}. Question:',
            f'Considering the picture, come up with a question where the answer is: {Answer}.',
            f'Taking the image into account, generate an question that has the answer: {Answer}. Question:'
        ]
        import random
        prompt=random.choice(prompts)
        prompt = self.text_processor(prompt)
        return {
            "image": image,
            "text_input": prompt,
            "text_output": question,
        }