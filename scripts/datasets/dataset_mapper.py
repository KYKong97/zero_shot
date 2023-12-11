import copy
import logging
from typing import Any
import numpy as np
import torch
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.data import detection_utils as utils
from detectron2.structures import BitMasks, Instances

__all__ = ["MaskFormerSemanticDatasetMapper"]

class MaskFormerSemanticDatasetMapper:

    @configurable
    def __init__(self,
                 is_train=True,
                 *,
                 augmentations,
                 image_format,
                 ignore_label,
                 size_divisibility) -> None:
        
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.img_format = image_format

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}"
        )

    @classmethod
    def from_config(cls, cfg, is_train=True):
        if is_train:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
                )
            ]

            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
                    )

                )
            
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs.append(T.RandomFlip())

            dataset_names = cfg.DATASETS.TRAIN
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
            augs = [T.ResizeShortestEdge(min_size, max_size,sample_style)]
            dataset_names = cfg.DATASETS.TEST
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train":is_train,
            "augmentations":augs,
            "image_format":cfg.INPUT.FORMAT,
            "ignore_label":ignore_label,
            "size_divisibility":cfg.INPUT.SIZE_DIVISIBILITY if is_train else -1
        }
        return ret

    def __call__(self, dataset_dict) -> Any:

        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None
        
        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens,aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad Image and segmentation label here
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2,0,1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0]
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(
                    sem_seg_gt,padding_size, value=self.ignore_label
                ).contiguous()
        image_shape = (image.shape[-2],image.shape[-1])

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()
        
        if "annotations" in dataset_dict:
            raise ValueError(
                "Semantic segmentation dataset should not have 'annotations'."
            )

        # Prepare category
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # removed ignore region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)
            
            if len(masks) == 0:
                ## Ignore images has no annotations
                instances.gt_masks = torch.zeros(
                    (0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1])
                )
            else:
                masks = BitMasks(
                    torch.stack(
                        [
                            torch.from_numpy(np.ascontiguousarray(x.copy()))
                            for x in masks
                        ]
                    )
                )

                instances.gt_masks = masks.tensor
            dataset_dict["instances"] = instances
        return dataset_dict
    

