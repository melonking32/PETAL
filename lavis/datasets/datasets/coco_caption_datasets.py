"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset

COCOCapDataset = CaptionDataset


class COCOCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]
        
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
        prompt=prompts[0]
        prompt = self.text_processor(prompt)
        
        Instruct_1='What objects are in the picture'
        Instruct_2='What are the characteristics of the objects in the picture'
        Instruct_3='What is the relationship between the objects in the picture'

        
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
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }


class NoCapsEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["img_id"]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }
        
class NoCapsEval_Instruct_Dataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        prompts="A short image description: "
        prompts = self.text_processor(prompts)
        image = self.vis_processor(image)

        img_id = ann["img_id"]

        return {
            "image": image,
            "text_input":prompts,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }
