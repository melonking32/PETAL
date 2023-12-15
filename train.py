import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
#suppress warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)

    model = task.build_model(cfg)
                
    if 'lora' in cfg.model_cfg['arch'] or 'adapter' in cfg.model_cfg['arch'] or 'aurora' in cfg.model_cfg['arch']:
        if not cfg.run_cfg['evaluate']:
            for name, m in model.named_modules():
                if 'lora' in name and 'lora.' not in name:
                    m.init_adapter_weights()
                if 'adapters' in name  and 'adapters.' not in name:
                    for mm in m:
                        mm.init_adapter_weights()

        # print('Initialize adapter')
        for name, parameter in model.named_parameters():
            if 'CP' in name or 'lora' in name or 'adapter' in name or 'Wrapper' in name or 'para_info' in name or 'Expert' in name: 
                parameter.requires_grad = True
            else: 
                parameter.requires_grad = False

    
    elif 'train_mode' in cfg.run_cfg and cfg.run_cfg['train_mode']=='head':
        for name, parameter in model.named_parameters():
            if '11' in name and 'bert' in name and 'intermediate' in name :  
            # if 'CP' in name:  # or adapter
                print(name)
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False
    runner = get_runner_class(cfg)(
         cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
     )
    runner.train()


if __name__ == "__main__":
    main()
