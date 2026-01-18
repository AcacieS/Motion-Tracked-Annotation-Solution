%cd /content/Cutie/

import torch
torch.backends.cudnn.benchmark = True

from omegaconf import open_dict
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from cutie.model.cutie import CUTIE
from cutie.inference.utils.args_utils import get_dataset_cfg

# Initialize Hydra only once
if not GlobalHydra.instance().is_initialized():
    initialize_config_dir(
        version_base="1.3.2",
        config_dir="/content/Cutie/cutie/config",
        job_name="eval_config",
    )

cfg = compose(config_name="eval_config")

with open_dict(cfg):
    cfg["weights"] = "./weights/cutie-base-mega.pth"
    cfg["mem_every"] = cfg.get("mem_every") or 5
    cfg["stagger_updates"] = cfg.get("stagger_updates") or 0

data_cfg = get_dataset_cfg(cfg)

cutie = CUTIE(cfg).cuda()
cutie.eval()
weights = torch.load(cfg.weights, weights_only=True, map_location="cuda")
cutie.load_weights(weights)

print("CUTIE loaded OK:", cfg.weights)
