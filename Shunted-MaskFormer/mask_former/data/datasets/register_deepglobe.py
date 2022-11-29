# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

dg_CATEGORIES = [
    {"id": 0, "name": "background"},
    {"id": 1, "name": "built-up"},
    {"id": 2, "name": "farmland"},
    {"id": 3, "name": "meadow"},
    {"id": 4, "name": "forest"},
    {"id": 5, "name": "water"},
    {"id": 6, "name": "bareland"}
]


def _get_gid_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in dg_CATEGORIES]
    assert len(stuff_ids) == 7, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in dg_CATEGORIES]

    ret = {
        "thing_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_gid(root):
    root = os.path.join(root, "deepglobedata")
    meta = _get_gid_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images_detectron2/train", "annotations_detectron2/train"),
        ("test", "images_detectron2/test", "annotations_detectron2/test"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"deepglobe_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=0,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")

register_all_gid(_root)
