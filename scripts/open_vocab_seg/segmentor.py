from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import configurable
import torch.nn as nn
import torch

backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}

@META_ARCH_REGISTRY.register()
class Segmentor(nn.Module):

    
    def __init__(self,cfg) -> None:
        super().__init__()

        
        backbone_arch = backbone_archs[cfg.MODEL.DINO.SIZE]
        backbone_name = f"dinov2_{backbone_arch}"

        self.backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
