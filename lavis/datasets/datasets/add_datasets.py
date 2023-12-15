
import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class TextCapsDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        prompts=['A short image caption:',
                'A short image description:',
                'A photo of',
                'An image that shows',
                'Write a short description for the image.',
                "Write a description for the photo: ",
                'Provide a description of what is presented in the photo.',
                'Briefly describe the content of the image.',
                'Can you briefly explain what you see in the image?',
                'Could you use a few words to describe what you perceive in the photo?',
                'Please provide a short depiction of the picture.',
                'Using language, provide a short account of the image.',
                'Use a few words to illustrate what is happening in the picture.'
                ]
        import random
        # prompt=random.choice(prompts)
        prompt=prompts[2]
        prompt = self.text_processor(prompt)
        caption = self.text_processor(ann["caption"])
        Instruct_1='What objects are in the picture'
        Instruct_2='What are the characteristics of the objects in the picture'
        Instruct_3='What is the relationship between the objects in the picture'
        # Instruct_1='What objects are in the picture'
        # Instruct_2='What color are the objects in the picture'
        # Instruct_3='What are the characteristics of the objects in the picture'
        
        Instruct_1=self.text_processor(Instruct_1)
        Instruct_2=self.text_processor(Instruct_2)
        Instruct_3=self.text_processor(Instruct_3)
        return {
            "image": image,
            "text_input":prompt,
            # "Instruct_1": Instruct_1,
            "Instruct_1": prompt,
            "Instruct_2": Instruct_2,
            "Instruct_3": Instruct_3,
            "text_output":caption,
        }
class TextCaps_Eval_Dataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image_path"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        prompts=['A short image caption:',
                'A short image description:',
                'A photo of',
                'An image that shows',
                'Write a short description for the image.',
                "Write a description for the photo: ",
                'Provide a description of what is presented in the photo.',
                'Briefly describe the content of the image.',
                'Can you briefly explain what you see in the image?',
                'Could you use a few words to describe what you perceive in the photo?',
                'Please provide a short depiction of the picture.',
                'Using language, provide a short account of the image.',
                'Use a few words to illustrate what is happening in the picture.'
                ]
        import random
        # prompt=random.choice(prompts)
        prompt=prompts[2]
        prompt = self.text_processor(prompt)
        # caption = self.text_processor(ann["caption"])
        Instruct_1='What objects are in the picture'
        Instruct_2='What are the characteristics of the objects in the picture'
        Instruct_3='What is the relationship between the objects in the picture'
        # Instruct_1='What objects are in the picture'
        # Instruct_2='What color are the objects in the picture'
        # Instruct_3='What are the characteristics of the objects in the picture'

        
        Instruct_1=self.text_processor(Instruct_1)
        Instruct_2=self.text_processor(Instruct_2)
        Instruct_3=self.text_processor(Instruct_3)
        return {
            "image": image,
            "text_input":prompt,
            "Instruct_1": Instruct_1,
            # "Instruct_1": prompt,
            "Instruct_2": Instruct_2,
            "Instruct_3": Instruct_3,
            "image_id": ann['image_id'],
            "index": index,
        }
    
class LLaVADataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        import random
        prompt=ann['input']
        prompt = self.text_processor(prompt)
        caption = self.text_processor(ann["output"])

        return {
            "image": image,
            "text_input":prompt,
            "text_output":caption,
        }


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


class TextVQA_Instruct_Dataset(VQA_Instruct_Dataset, __DisplMixin):
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
        answers = ann["answer"][0]

        return {
            "image": image,
            "text_input": question,
            "Instruct_1": Instruct_1,
            # "Instruct_1": question,
            "Instruct_2": Instruct_2,
            "Instruct_3": Instruct_3,
            "text_output": answers
        }


class TextVQAEvalDataset(VQAEvalDataset, __DisplMixin):
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
            answers = ann["answer"][0]
        else:
            answers = None

        return {
            "image": image,
            "text_input": question,
            "Instruct_1": Instruct_1,
            # "Instruct_1": question,
            "Instruct_2": Instruct_2,
            "Instruct_3": Instruct_3,
            # "direct_answers": answers,
            "answer": answers,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }
