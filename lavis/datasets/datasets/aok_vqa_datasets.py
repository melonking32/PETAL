"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from collections import OrderedDict
import json
import os
import torch

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset,VQA_Instruct_Dataset


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "direct_answers": "; ".join(ann["direct_answers"]),
                "choices": "; ".join(ann["choices"]),
                "correct_choice": ann["choices"][ann["correct_choice_idx"]],
                "image": sample["image"],
            }
        )


class AOKVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer_key = "direct_answers"

        answer_weight = {}
        for answer in ann[answer_key]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann[answer_key])
            else:
                answer_weight[answer] = 1 / len(ann[answer_key])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }


class AOKVQAEvalDataset(VQAEvalDataset, __DisplMixin):
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

    def collater(self, samples):
        (
            image_list,
            question_list,
            question_id_list,
            instance_id_list,
            choices_list,
            correct_choice_idx_list,
            direct_answers_list,
        ) = ([], [], [], [], [], [], [])

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            question_id_list.append(sample["question_id"])
            instance_id_list.append(sample["instance_id"])
            choices_list.append(sample["choices"])
            correct_choice_idx_list.append(sample["correct_choice_idx"])
            direct_answers_list.append(sample["direct_answers"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "question_id": question_id_list,
            "instance_id": instance_id_list,
            "choices": choices_list,
            "correct_choice_idx": correct_choice_idx_list,
            "direct_answers": direct_answers_list,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        choices = ann["choices"]
        if "correct_choice_idx" in ann:
            correct_choice_idx = ann["correct_choice_idx"]
        else:
            correct_choice_idx = None

        if "direct_answers" in ann:
            direct_answers = ann["direct_answers"]
        else:
            direct_answers = None

        return {
            "image": image,
            "text_input": question,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "choices": choices,
            "correct_choice_idx": correct_choice_idx,
            "direct_answers": direct_answers,
        }


class AOKVQA_Instruct_Dataset(VQA_Instruct_Dataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        Question=ann["question"]
        # Instructblip 联合finetune
        # prompts=[f'{Question}',
        # f'Question: {Question}',
        # f'{Question} A short answer to the question is',
        # f'Q: {Question} A:',
        # f'Question: {Question} Short answer:',
        # f'Given the image, answer the following question with no more than three words. {Question}',
        # f'Based on the image, respond to this question with a short answer: {Question}. Answer:',
        # f'Use the provided image to answer the question: {Question} Provide your answer as short as possible:',
        # f'What is the answer to the following question? "{Question}"',
        # f'The question "{Question}" can be answered using the image. A short answer is'
        # ]
        # import random
        # question=random.choice(prompts)

        # 单任务Finetune
        question=f'Question: {Question} Short answer:'
        question2=f'Question: {Question} Answer the following question with no more than three words. '
        Instruct_1='What objects are in the picture'
        Instruct_2='What are the characteristics of the objects in the picture'
        Instruct_3='What is the relationship between the objects in the picture'
        # Instruct_1='What objects are in the picture'
        # Instruct_2='What color are the objects in the picture'
        # Instruct_3='What are the characteristics of the objects in the picture'

        
        Instruct_1=self.text_processor(Instruct_1)
        Instruct_2=self.text_processor(Instruct_2)
        Instruct_3=self.text_processor(Instruct_3)
        
        question = self.text_processor(question)
        question2=self.text_processor(question2)
        answer_key = "direct_answers"

        answer_weight = {}
        for answer in ann[answer_key]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann[answer_key])
            else:
                answer_weight[answer] = 1 / len(ann[answer_key])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "text_input": question,
            "Instruct_1": Instruct_1,
            # "Instruct_1": question,
            "Instruct_2": Instruct_2,
            "Instruct_3": Instruct_3,
            'text_input_instruct2':question2,
            "text_output": answers[0],
        }


class AOKVQA_Instruct_EvalDataset(VQAEvalDataset, __DisplMixin):
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

    def collater(self, samples):
        (
            image_list,
            question_list,
            Instruction_list_1,
            Instruction_list_2,
            Instruction_list_3,
            question_id_list,
            instance_id_list,
            choices_list,
            correct_choice_idx_list,
            direct_answers_list,
        ) = ([], [], [], [], [], [], [],[], [],[])

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            Instruction_list_1.append(sample["Instruct_1"])
            Instruction_list_2.append(sample["Instruct_2"])
            Instruction_list_3.append(sample["Instruct_3"])
            
            question_id_list.append(sample["question_id"])
            instance_id_list.append(sample["instance_id"])
            choices_list.append(sample["choices"])
            correct_choice_idx_list.append(sample["correct_choice_idx"])
            direct_answers_list.append(sample["direct_answers"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "Instruct_1": Instruction_list_1,
            "Instruct_2": Instruction_list_2,
            "Instruct_3": Instruction_list_3,
            "question_id": question_id_list,
            "instance_id": instance_id_list,
            "choices": choices_list,
            "correct_choice_idx": correct_choice_idx_list,
            "direct_answers": direct_answers_list,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        Instruct_1='What objects are in the picture'
        Instruct_2='What are the characteristics of the objects in the picture'
        Instruct_3='What is the relationship between the objects in the picture'
        # Instruct_1='What objects are in the picture'
        # Instruct_2='What color are the objects in the picture'
        # Instruct_3='What are the characteristics of the objects in the picture'

        
        Instruct_1=self.text_processor(Instruct_1)
        Instruct_2=self.text_processor(Instruct_2)
        Instruct_3=self.text_processor(Instruct_3)
        
        choices = ann["choices"]
        if "correct_choice_idx" in ann:
            correct_choice_idx = ann["correct_choice_idx"]
        else:
            correct_choice_idx = None

        if "direct_answers" in ann:
            direct_answers = ann["direct_answers"]
        else:
            direct_answers = None

        # return {
        #     "image": image,
        #     "text_input": question,
        #     "question_id": ann["question_id"],
        #     "instance_id": ann["instance_id"]
        # }
        return {
            "image": image,
            "text_input": question,
            "Instruct_1": Instruct_1,
            # "Instruct_1": question,
            "Instruct_2": Instruct_2,
            "Instruct_3": Instruct_3,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "choices": choices,
            "correct_choice_idx": correct_choice_idx,
            "direct_answers": direct_answers,
        }