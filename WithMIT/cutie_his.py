%cd /content/Cutie/

import torch
from omegaconf import open_dict
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from cutie.model.cutie import CUTIE
from cutie.inference.utils.args_utils import get_dataset_cfg

# ---- choose device ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 如果你想强制不用GPU，就取消下一行注释：
# DEVICE = "cpu"

# ---- clear hydra (notebook) ----
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

with torch.inference_mode():
    initialize_config_dir(
        version_base="1.3.2",
        config_dir="/content/Cutie/cutie/config",
        job_name="eval_config",
    )
    cfg = compose(config_name="eval_config")

    with open_dict(cfg):
        cfg["weights"] = "./weights/cutie-base-mega.pth"
        if cfg.get("mem_every", None) is None:
            cfg["mem_every"] = 5
        if cfg.get("stagger_updates", None) is None:
            cfg["stagger_updates"] = 0

    _ = get_dataset_cfg(cfg)

    cutie = CUTIE(cfg).to(DEVICE).eval()
    model_weights = torch.load(cfg.weights, map_location=DEVICE)
    cutie.load_weights(model_weights)

print("CUTIE loaded OK:", cfg.weights, "| device:", DEVICE)