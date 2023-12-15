"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.aok_vqa_datasets import AOKVQADataset, AOKVQAEvalDataset,AOKVQA_Instruct_EvalDataset,AOKVQA_Instruct_Dataset
from lavis.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset,OKVQA_Instruct_Dataset,COCOVQA_Instruct_Dataset,COCOVQA_Instruct_EvalDataset
from lavis.datasets.datasets.vg_vqa_datasets import VGVQADataset
from lavis.datasets.datasets.gqa_datasets import GQADataset, GQAEvalDataset
from lavis.datasets.datasets.add_datasets import TextVQA_Instruct_Dataset,TextVQAEvalDataset

from lavis.datasets.datasets.qa_qg_datasets import COCO_Question_Dataset,OKVQA_Question_Dataset,AOKVQA_Question_Dataset
 

@registry.register_builder("coco_vqa_qg")
class COCO_Question_Generate_Builder(BaseDatasetBuilder):
    train_dataset_cls = COCO_Question_Dataset
    eval_dataset_cls = COCOVQA_Instruct_EvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_qg.yaml",
    }

@registry.register_builder("ok_vqa_qg")
class OKVQA_Question_Generate_Builder(BaseDatasetBuilder):
    train_dataset_cls = OKVQA_Question_Dataset
    eval_dataset_cls = COCOVQAEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults_qg.yaml",
    }


@registry.register_builder("aok_vqa_qg")
class AOKVQA_Question_Generate_Builder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQA_Question_Dataset
    eval_dataset_cls = AOKVQA_Instruct_EvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults_qg.yaml"}


@registry.register_builder("textvqa_instruct")
class TextVQA_Instruct_Builder(BaseDatasetBuilder):
    train_dataset_cls = TextVQA_Instruct_Dataset
    eval_dataset_cls = TextVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/textvqa/defaults_textvqa.yaml"}
