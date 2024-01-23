# Parameter-efficient Instruction-enhanced Learning

[Paper](https://arxiv.org/abs/2312.12458) | [arXiv](https://arxiv.org/abs/2312.12458) 

We propose a novel approach to utilize Parameter-Efficient Tuning for generAl-purpose vision-Language models, namely PETAL. PETAL enhances the semantic depth of instructions in two innovative ways: **1)** by introducing adaptive instruction mixture-of-experts (MOEs), and **2)** by fortifying the score-based linkage between parameterefficient tuning and mutual information.

The whole architecture of our PETAL:

<a href="url"><img src="https://github.com/melonking32/PETAL/blob/main/assets/main.pdf" align="center" width="700" ></a>

Case study and the visualization output:

<a href="url"><img src="https://github.com/melonking32/PETAL/blob/main/assets/Case1.jpg" align="center" width="700" ></a>


## Installation

1. First clone the directory.

2. Install dependencies.

First, create a new environment using [conda](https://docs.conda.io/en/latest/miniconda.html) (with python >= 3.7). Then install pytorch and other dependencies as follows 

Install pytorch (replace "cu113" with appropriate cuda version. For example, cuda11.1 will use "cu111"):
```code
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other dependencies. Run the following command:
```code
pip install -r requirements.txt 
```

## Training
```bash
cd PETAL
bash run_scripts/blip2/train/train_aurora_mixture.sh
```

## Evaluation
```bash
bash run_scripts/blip2/eval/eval_aurora_mixture.sh
```
