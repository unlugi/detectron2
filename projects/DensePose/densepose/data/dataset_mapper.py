# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import logging
from typing import Any, Dict, Tuple
import torch

from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.layers import ROIAlign
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

from .structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData

from cairosvg import svg2svg, svg2png
import xml.etree.cElementTree as ET
import re
import io
import numpy as np
import os
import PIL.Image as Image

# TODO: clean this up
ns = {'svg': "http://www.w3.org/2000/svg",
      'inkscape': "http://www.inkscape.org/namespaces/inkscape"}

BODY_PARTS_FULL = ['HEAD', 'NECK',
              'UPPER_TORSO', 'LOWER_TORSO',
              'L_SHOULDER', 'R_SHOULDER',
              'L_UPPERARM', 'R_UPPERARM',
              'L_ELBOW',    'R_ELBOW',
              'L_FOREARM',  'R_FOREARM',
              'L_WRIST',    'R_WRIST',
              'L_HAND',     'R_HAND',
              'L_THUMB',    'R_THUMB',
              'L_HIP',      'R_HIP',
              'L_UPPERLEG', 'R_UPPERLEG',
              'L_KNEE',     'R_KNEE',
              'L_LOWERLEG', 'R_LOWERLEG',
              'L_ANKLE',    'R_ANKLE',
              'L_FOOT',     'R_FOOT']

JOINTS = ['L_SHOULDER', 'R_SHOULDER',
          'L_ELBOW',    'R_ELBOW',
          'L_WRIST',    'R_WRIST',
          'L_HIP',      'R_HIP',
          'L_KNEE',     'R_KNEE',
          'L_ANKLE',    'R_ANKLE']

LIMBS = ['L_UPPERARM', 'R_UPPERARM',
         'L_FOREARM',  'R_FOREARM',
         'L_HAND',     'R_HAND',
         'L_THUMB',    'R_THUMB',
         'L_UPPERLEG', 'R_UPPERLEG',
         'L_LOWERLEG', 'R_LOWERLEG',
         'L_FOOT',     'R_FOOT']

TORSO = ['UPPER_TORSO', 'LOWER_TORSO']

TOP = ['HEAD', 'NECK']

# TODO: change this dictionary
BODY_PARTS = {'JOINTS': JOINTS,
              'LIMBS': LIMBS,
              'TORSO': TORSO,
              'TOP': TOP}

AUGS_PER_PART = {'JOINTS': ['disappear_random', 'translate', 'jitter'],
                 'LIMBS': ['disappear_occlusion', 'translate', 'jitter'],
                 'TORSO': ['disappear_occlusion', 'translate', 'jitter'],
                 'TOP': ['translate', 'jitter']}

class SVGAugmenter:
    """ Body part-based data augmenter class for Primitive3DHuman svg sketches. """
    def __init__(self, body_parts, augs_per_part,  do_aug, aug_joints, aug_limbs_occ,
                aug_limbs_vis, aug_torso_occ, aug_torso_vis, return_svg=False):

        self.return_svg = return_svg  # If true, returns .svg. Else, returns byte string
        self.body_parts = body_parts
        self.augs_per_part = augs_per_part
        self.do_aug = do_aug
        self.aug_joints = aug_joints
        self.aug_limbs_occ = aug_limbs_occ
        self.aug_limbs_vis = aug_limbs_vis
        self.aug_torso_occ = aug_torso_occ
        self.aug_torso_vis = aug_torso_vis

    def augment(self, filename, save_folder="", im_no=1):

        # TODO: This is too slow bc. all the loops
        # Read the sketch .svg
        tree = ET.parse(filename)
        root = tree.getroot()

        if self.do_aug:
            for i, path in enumerate(root.findall('svg:g/svg:g/', ns)):
                path.attrib['stroke-width'] = "1.9"
                body_part_name = path.attrib['stroke-name']
                body_part_key = ''
                # find body part name in self.body_parts dict and get its key
                for key, value in BODY_PARTS.items():
                    if body_part_name in value:
                        body_part_key = key
                        break
                # use the key to get the augmentation
                augmentations_for_part = self.augs_per_part[body_part_key]

                if path.attrib['stroke-opacity'] == '0.0':
                    continue

                if 'disappear_random' in augmentations_for_part:
                    # Disappearing joints
                    self.disappear(path, prob_vis=self.aug_joints, use_occlusion=False)

                if 'disappear_occlusion' in augmentations_for_part:
                    # Disappear wrt. occlusion - affects only torso and limbs atm.
                    if body_part_key == 'TORSO':
                        if not(self.aug_torso_vis == 0 and self.aug_torso_occ == 0):
                            self.disappear(path, prob_vis=self.aug_torso_vis, prob_occ=self.aug_torso_occ, use_occlusion=True)
                    elif body_part_key == 'LIMBS':
                        self.disappear(path, prob_vis=self.aug_limbs_vis, prob_occ=self.aug_limbs_occ, use_occlusion=True)

                if 'translate' in augmentations_for_part:
                    # Translate body parts globally - all body parts? probability? how much
                    # TODO: translate parts with multiple paths together?
                    self.translate_path_global(path)

                if 'jitter' in augmentations_for_part:
                    # Local path jitter
                    self.jitter_path_local(path)

                if 'squish' in augmentations_for_part:
                    # Squish joints (Optional)
                    self.squish_joints(path)

        if self.return_svg:
            # Saves the augmentation to a file
            image = svg2png(bytestring=ET.tostring(root), background_color='rgb(255,255,255)',
                            write_to=os.path.join(save_folder,  im_no + '.png'))
        else:
            # Returns bytestring
            image = svg2png(bytestring=ET.tostring(root), background_color='rgb(255,255,255)')
            #image = Image.open(io.BytesIO(image))
            #image.show()
            #return io.BytesIO(image)
            return Image.open(io.BytesIO(image))

    def disappear(self, path, body_parts=[], prob_vis=0.5, prob_occ=0.0, use_occlusion=False):

        # TODO: Hiding based on occlusion changed. All paths are either 1 or 0.
        if use_occlusion:
            per_node_occlusion = [int(o) for o in path.attrib['stroke-occlusion'].split(' ')[1:-1]]
            # Count occluded points
            #num_occluded_points = len(per_node_occlusion) - np.count_nonzero(per_node_occlusion)
            #prob = num_occluded_points / len(per_node_occlusion)
            if np.count_nonzero(per_node_occlusion) == len(per_node_occlusion):
                # means all 1's
                prob = prob_vis
            else:
                # means all zeros
                prob = prob_occ
        else:
            prob = prob_vis

        # Draw a sample from the uniform distribution
        do = np.random.uniform(0, 1.0) < prob
        if do:
            path.attrib['stroke-opacity'] = "0.0"

    def translate_path_global(self, path, body_parts=[], prob=0.4):

        # Draw a sample from the uniform distribution
        do = np.random.uniform(0, 1) < prob
        if do:  # TODO: whole part vs path in a part?
            old_path_d = path.attrib['d']
            old_path_xy = [float(o) for o in re.split(' |, ', old_path_d)[2:-1]]

            n_node = len(old_path_xy) // 2  # Number of nodes in the path
            translate_xy = np.random.uniform(-3, 3, 2)  # Generate random translations in xy-axes
            new_path_xy = np.asarray(old_path_xy).reshape(n_node, 2) + np.repeat(translate_xy[None, :], n_node, axis=0)

            path.attrib['d'] = ' M ' + "".join('{:.3f}, {:.3f} '.format(x, y) for x, y in new_path_xy)

    def jitter_path_local(self, path, prob=0.5):
        # Draw a sample from the uniform distribution
        #do = np.random.uniform(0, 1) < prob
        #if do:

        old_path_d = path.attrib['d']
        old_path_xy = [float(o) for o in re.split(' |, ', old_path_d)[2:-1]]

        n_xy = len(old_path_xy) // 2
        n_params = np.ceil(n_xy / 10).astype(int)  # We want each sub path to be 5 nodes in length
        path_jitter_xy = np.random.normal(0, 0.7, (n_params, 2))
        path_jitter_xy = np.repeat(path_jitter_xy, 10, axis=0)[:n_xy]
        new_path_xy = np.asarray(old_path_xy).reshape(n_xy, 2) + path_jitter_xy

        path.attrib['d'] = ' M ' + "".join('{:.3f}, {:.3f} '.format(x, y) for x, y in new_path_xy)

    def squish_joints(self, body_parts=[]):
        # TODO: rotate and add anisotropic scale on the joints for squishy effect.
        ...


def build_augmentation(cfg, is_train):
    logger = logging.getLogger(__name__)
    result = utils.build_augmentation(cfg, is_train)
    if is_train:
        random_rotation = T.RandomRotation(
            cfg.INPUT.ROTATION_ANGLES, expand=False, sample_style="choice"
        )
        result.append(random_rotation)
        logger.info("DensePose-specific augmentation used in training: " + str(random_rotation))
    return result


class DatasetMapper:
    """
    A customized version of `detectron2.data.DatasetMapper`
    """

    def __init__(self, cfg, is_train=True):
        self.augmentation = build_augmentation(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = (
            cfg.MODEL.MASK_ON or (
                cfg.MODEL.DENSEPOSE_ON
                and cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS)
        )
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.densepose_on   = cfg.MODEL.DENSEPOSE_ON
        assert not cfg.MODEL.LOAD_PROPOSALS, "not supported yet"
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.densepose_on:
            densepose_transform_srcs = [
                MetadataCatalog.get(ds).densepose_transform_src
                for ds in cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
            ]
            assert len(densepose_transform_srcs) > 0
            # TODO: check that DensePose transformation data is the same for
            # all the datasets. Otherwise one would have to pass DB ID with
            # each entry to select proper transformation data. For now, since
            # all DensePose annotated data uses the same data semantics, we
            # omit this check.
            densepose_transform_data_fpath = PathManager.get_local_path(densepose_transform_srcs[0])
            self.densepose_transform_data = DensePoseTransformData.load(
                densepose_transform_data_fpath
            )

        self.is_train = is_train

        # TODO: Configs for svg augmentations
        self.load_svg = cfg.INPUT.SVG.LOAD_SVG
        self.svg_augment = SVGAugmenter(body_parts=BODY_PARTS, augs_per_part=AUGS_PER_PART,
                                        do_aug=cfg.INPUT.SVG.AUG,
                                        aug_joints=cfg.INPUT.SVG.AUG_PROB_JOINTS,
                                        aug_limbs_occ=cfg.INPUT.SVG.AUG_PROB_LIMBS_OCC,
                                        aug_limbs_vis=cfg.INPUT.SVG.AUG_PROB_LIMBS_VIS,
                                        aug_torso_occ=cfg.INPUT.SVG.AUG_PROB_TORSO_OCC,
                                        aug_torso_vis=cfg.INPUT.SVG.AUG_PROB_TORSO_VIS,
                                        return_svg=False)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # TODO: modifying this
        if self.load_svg: # load svg
            image = self.svg_augment.augment(dataset_dict["file_name"][:-3] + 'svg')
            image = utils.convert_PIL_to_numpy(image, self.img_format)
        else: # load raster image
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)

        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.augmentation, image)
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        for anno in dataset_dict["annotations"]:
            if not self.mask_on:
                anno.pop("segmentation", None)
            if not self.keypoint_on:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        # USER: Don't call transpose_densepose if you don't need
        annos = [
            self._transform_densepose(
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                ),
                transforms,
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        if self.mask_on:
            self._add_densepose_masks_as_segmentation(annos, image_shape)

        instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        densepose_annotations = [obj.get("densepose") for obj in annos]
        if densepose_annotations and not all(v is None for v in densepose_annotations):
            instances.gt_densepose = DensePoseList(
                densepose_annotations, instances.gt_boxes, image_shape
            )

        dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]
        return dataset_dict

    def _transform_densepose(self, annotation, transforms):
        if not self.densepose_on:
            return annotation

        # Handle densepose annotations
        is_valid, reason_not_valid = DensePoseDataRelative.validate_annotation(annotation)
        if is_valid:
            densepose_data = DensePoseDataRelative(annotation, cleanup=True)
            densepose_data.apply_transform(transforms, self.densepose_transform_data)
            annotation["densepose"] = densepose_data
        else:
            # logger = logging.getLogger(__name__)
            # logger.debug("Could not load DensePose annotation: {}".format(reason_not_valid))
            DensePoseDataRelative.cleanup_annotation(annotation)
            # NOTE: annotations for certain instances may be unavailable.
            # 'None' is accepted by the DensePostList data structure.
            annotation["densepose"] = None
        return annotation

    def _add_densepose_masks_as_segmentation(
        self, annotations: Dict[str, Any], image_shape_hw: Tuple[int, int]
    ):
        for obj in annotations:
            if ("densepose" not in obj) or ("segmentation" in obj):
                continue
            # DP segmentation: torch.Tensor [S, S] of float32, S=256
            segm_dp = torch.zeros_like(obj["densepose"].segm)
            segm_dp[obj["densepose"].segm > 0] = 1
            segm_h, segm_w = segm_dp.shape
            bbox_segm_dp = torch.tensor((0, 0, segm_h - 1, segm_w - 1), dtype=torch.float32)
            # image bbox
            x0, y0, x1, y1 = (
                v.item() for v in BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
            )
            segm_aligned = (
                ROIAlign((y1 - y0, x1 - x0), 1.0, 0, aligned=True)
                .forward(segm_dp.view(1, 1, *segm_dp.shape), bbox_segm_dp)
                .squeeze()
            )
            image_mask = torch.zeros(*image_shape_hw, dtype=torch.float32)
            image_mask[y0:y1, x0:x1] = segm_aligned
            # segmentation for BitMask: np.array [H, W] of np.bool
            obj["segmentation"] = image_mask >= 0.5
