# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from sam2.build_sam import build_sam2_video_predictor
import cv2
import math
import gc
torch.cuda.empty_cache()
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random

from tmp_module.tmp_v5 import box_prompt_predictor
from tmp_module.kalman_refine_module import BoxKalman


# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask_img = Image.open(path)
    palette = mask_img.getpalette()
    mask_array = np.array(mask_img).astype(np.uint8)
    return mask_array, palette


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if allow_missing and not os.path.exists(input_mask_path): # allow_missing允许没有输入帧
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette

def mask_to_box(mask):
    # find contours by using opencv
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # just contour was not corrected(Scattered holes), we need better method to find the most big contour
    if contours is None or len(contours) == 0:
        # print("No object found in the mask")
        mask_box = [0,0,0,0]
        return mask_box

    if contours: # find the max box
        x_list = []
        y_list = []
        x2_list = []
        y2_list = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_list.append(x)
            y_list.append(y)
            x2 = x+w
            y2 = y+h
            x2_list.append(x2)
            y2_list.append(y2)
        x = min(x_list)
        y = min(y_list)
        x2 = max(x2_list)
        y2 = max(y2_list)
        mask_box = [x, y, x2, y2]
        return mask_box


def evaluateJF(all_res_masks, all_gt_masks, all_void_masks):
    j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
    j_metrics_res = db_eval_iou(all_res_masks, all_gt_masks, all_void_masks)
    f_metrics_res = db_eval_boundary(all_res_masks, all_gt_masks, all_void_masks)
    return j_metrics_res, f_metrics_res

def db_eval_iou(segmentation, annotation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j

def db_eval_boundary(segmentation, annotation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res

def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap

# Data required for constructing prediction boxes
def construct_optim_data_reverse(frame_idx, obj_id, inference_state, local_rank, local_rate, middle_rate, backtrack_range):
    frame_idx_list_local = []
    frame_idx_list_middle = []
    his_img_emb_local = []
    his_img_emb_middle = []
    his_img_emb_large = []
    box_tensor_list_local = []
    box_tensor_list_middle = []

    if frame_idx < 48:
        start = frame_idx - 8
        end = frame_idx
        cyc_num = 0
        for frame_ind in range(end, start, -1):
            exist = 0
            if inference_state['mask_to_box'][frame_ind][obj_id]['mask_box'][0] == 0 and inference_state['mask_to_box'][frame_ind][obj_id]['mask_box'][2] == 0:
                start_add = frame_ind - backtrack_range
                end_add = frame_ind - 1
                for recent_idx in range(end_add, start_add, -1):
                    if recent_idx < 0:
                        continue
                    if inference_state['mask_to_box'][recent_idx][obj_id]['mask_box'][0] != 0 or inference_state['mask_to_box'][recent_idx][obj_id]['mask_box'][2] != 0:
                        if cyc_num < 4:
                            frame_idx_list_local.append(recent_idx)
                            frame_idx_list_middle.append(recent_idx)
                        else:
                            frame_idx_list_local.append(recent_idx)
                        exist = 1
                        break
                    else:
                        continue
                if exist == 0:
                    return exist, None, None, None, None, None, None, None, None, None
            else:
                if cyc_num < 4:
                    frame_idx_list_local.append(frame_ind)
                    frame_idx_list_middle.append(frame_ind)
                else:
                    frame_idx_list_local.append(frame_ind)
            cyc_num += 1

    else:
        start_local = frame_idx - local_rate*8
        end_local = frame_idx
        for frame_ind in range(end_local, start_local, -local_rate):
            exist = 0
            if inference_state['mask_to_box'][frame_ind][obj_id]['mask_box'][0] == 0 and inference_state['mask_to_box'][frame_ind][obj_id]['mask_box'][2] == 0:
                start_add = frame_ind - backtrack_range
                end_add = frame_ind - 1
                for recent_idx in range(end_add, start_add, -1):
                    if inference_state['mask_to_box'][recent_idx][obj_id]['mask_box'][0] != 0 or inference_state['mask_to_box'][recent_idx][obj_id]['mask_box'][2] != 0:
                        frame_idx_list_local.append(recent_idx)
                        exist = 1
                        break
                    else:
                        continue
                if exist == 0:
                    return exist, None, None, None, None, None, None, None, None, None
            else:
                frame_idx_list_local.append(frame_ind)

        # middle sampling
        start_middle = frame_idx - middle_rate * 4
        end_middle = frame_idx
        for frame_ind in range(end_middle, start_middle, -middle_rate):
            exist = 0
            if inference_state['mask_to_box'][frame_ind][obj_id]['mask_box'][0] == 0 and inference_state['mask_to_box'][frame_ind][obj_id]['mask_box'][2] == 0:
                start_add = frame_ind - backtrack_range
                end_add = frame_ind - 1
                for recent_idx in range(end_add, start_add, -1):
                    if inference_state['mask_to_box'][recent_idx][obj_id]['mask_box'][0] != 0 or inference_state['mask_to_box'][recent_idx][obj_id]['mask_box'][2] != 0:
                        frame_idx_list_middle.append(recent_idx)
                        exist = 1
                        break
                    else:
                        continue
                if exist == 0:
                    return exist, None, None, None, None, None, None, None, None, None
            else:
                frame_idx_list_middle.append(frame_ind)

    assert len(frame_idx_list_local) == 8
    for idx in frame_idx_list_local:
        his_img_emb_local.append(inference_state['backbone_out'][idx]['backbone_fpn_original'][2])
        box_tensor_list_local.append(inference_state['box_prompt_feature'][idx][obj_id])
    his_img_emb_local_list = torch.cat(his_img_emb_local[0:8], dim=0).to(local_rank)
    assert len(his_img_emb_local_list) == 8
    his_box_emb_list_local = torch.cat(box_tensor_list_local[0:8], dim=0).to(local_rank)
    assert len(his_box_emb_list_local) == 8

    assert len(frame_idx_list_middle) == 4
    for idx in frame_idx_list_middle:
        his_img_emb_middle.append(inference_state['backbone_out'][idx]['backbone_fpn_original'][1])
        box_tensor_list_middle.append(inference_state['box_prompt_feature'][idx][obj_id])
    his_img_emb_middle_list = torch.cat(his_img_emb_middle[0:4], dim=0).to(local_rank)
    assert len(his_img_emb_middle_list) == 4
    his_box_emb_list_middle = torch.cat(box_tensor_list_middle[0:4], dim=0).to(local_rank)
    assert len(his_box_emb_list_middle) == 4

    his_img_emb_large.append(inference_state['backbone_out'][frame_idx_list_local[0]]['backbone_fpn_original'][0])
    his_img_emb_large_list = torch.cat(his_img_emb_large[0:1], dim=0).to(local_rank)
    his_box_emb_list_large = torch.cat(box_tensor_list_local[0:1], dim=0).to(local_rank)
    assert len(his_box_emb_list_large) == 1
    assert len(his_img_emb_large_list) == 1
    # vision_pos_emb
    img_pos_enc_local = inference_state['backbone_out'][frame_idx]['vision_pos_enc'][2]
    img_pos_enc_middle = inference_state['backbone_out'][frame_idx]['vision_pos_enc'][1]
    img_pos_enc_large = inference_state['backbone_out'][frame_idx]['vision_pos_enc'][0]
    exist = 1
    return exist, his_img_emb_local_list, his_img_emb_middle_list, his_img_emb_large_list, img_pos_enc_local, img_pos_enc_middle, img_pos_enc_large, his_box_emb_list_local, his_box_emb_list_middle, his_box_emb_list_large


def extend_train_output(train_output, extend_rate, inference_state, device):
    x = train_output[2] - train_output[0]
    y = train_output[3] - train_output[1]
    x_expand = x * extend_rate / 2
    y_expand = y * extend_rate / 2
    x1 = train_output[0] - x_expand
    x2 = train_output[2] + x_expand
    y1 = train_output[1] - y_expand
    y2 = train_output[3] + y_expand
    if x1 < 0:
        x1 = 0
    if x2 > inference_state['video_width']:
        x2 = inference_state['video_width']
    if y1 < 0:
        y1 = 0
    if y2 > inference_state['video_height']:
        y2 = inference_state['video_height']
    extend_box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=device, requires_grad=True)
    return extend_box

def sample_in_annulus(box_small, box_big):
    # sampling the points
    sample_points = []
    S = (box_big[2]-box_big[0]) * (box_big[3]-box_big[1]) - (box_small[2]-box_small[0]) * (box_small[3]-box_small[1])
    num_points = int(S/3000)
    if num_points > 6:
        num_points = 6
    while len(sample_points) < num_points:
        x = int(random.uniform(box_big[0], box_big[2]))
        y = int(random.uniform(box_big[1], box_big[3]))
        if not (box_small[0] <= x <= box_small[2] and box_small[1] <= y <= box_small[3]):
            sample_points.append((x, y))
    return np.array(sample_points)

def transfer_point_inputs(predict_box, inference_state, local_rank):
    """
        Process the point prompt into the format required by the SAM2 model
    """
    points = torch.zeros(0, 2, dtype=torch.float32).to(local_rank)
    labels = torch.zeros(0, dtype=torch.int32).to(local_rank)
    if points.dim() == 2:
        points = points.unsqueeze(0)  # add batch dimension
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)  # add batch dimension

    box_coords = predict_box.reshape(1, 2, 2)
    box_labels = torch.tensor([2, 3], dtype=torch.int32, device=local_rank)
    box_labels = box_labels.reshape(1, 2)
    points = torch.cat([box_coords, points], dim=1)
    labels = torch.cat([box_labels, labels], dim=1)
    video_H = inference_state['video_height']
    video_W = inference_state['video_width']
    points = points / torch.tensor([video_W, video_H]).to(points.device)
    points = points * 1024  # self.img_size
    points = points.to(local_rank)
    labels = labels.to(local_rank)
    return points, labels

def construct_point_inputs(frame_idx, obj_ids, inference_state, local_rank):
    """
        Process the box prompts as the point prompts into the format required by the SAM2 model
    """
    box_prompt_coords = []
    box_prompt_labels = []
    need_prompt = 0
    need_prompt_list = []

    if inference_state['box_prompt_per_frame'].get(frame_idx+1) is not None:
        for i, obj_id in enumerate(obj_ids):
            if inference_state['box_prompt_per_frame'][frame_idx+1]['box_optim'].get(obj_id) is not None:
                need_prompt = 2
            else: # no obj
                need_prompt = 1
            if need_prompt == 2:
                box_optim = inference_state['box_prompt_per_frame'][frame_idx + 1]['box_optim'][obj_id]
                box_coord, box_label = transfer_point_inputs(box_optim, inference_state, local_rank)
                box_prompt_coords.append(box_coord)
                box_prompt_labels.append(box_label)
            if need_prompt == 1:
                box_coord = torch.zeros(1, 2, 2, device=local_rank)
                box_label = -torch.ones(1, 2, dtype=torch.int32, device=local_rank)
                box_prompt_coords.append(box_coord)
                box_prompt_labels.append(box_label)
            need_prompt_list.append(need_prompt)

        box_pormpt_per_frame = torch.cat(box_prompt_coords, dim=0).to(local_rank)
        box_labels_per_frame = torch.cat(box_prompt_labels, dim=0).to(local_rank)
        inference_state['box_prompt_per_frame'][frame_idx+1]['point_coords'] = box_pormpt_per_frame
        inference_state['box_prompt_per_frame'][frame_idx+1]['point_labels'] = box_labels_per_frame
        inference_state['box_prompt_per_frame'][frame_idx + 1]['real_need_box'] = need_prompt_list
        point_inputs = inference_state['box_prompt_per_frame'][frame_idx+1]
    else:
        point_inputs = None
        need_prompt_list = None

    return point_inputs


def put_per_obj_mask(object_ids_set, current_masks, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for object_id in object_ids_set:
        object_mask = current_masks[object_id]['output_mask']
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask

def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)

def save_masks_to_dir(
    object_ids_set,
    output_mask_dir,
    video_name,
    frame_name,
    current_masks,
    height,
    width,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, "mask_result"), exist_ok=True)
    output_mask = put_per_obj_mask(object_ids_set, current_masks, height, width)
    output_mask_path = os.path.join(
        output_mask_dir, "mask_result", f"{frame_name}.png"
    )
    save_ann_png(output_mask_path, output_mask, output_palette)

def visual_result(
    object_ids_set,
    video_name,
    out_frame_idx,
    output_mask_dir,
    frame_name,
    video_dir,
    inference_state,
    current_masks,
):
    """
        Save masks to a directory as PNG files with box prompts.
    """
    os.makedirs(os.path.join(output_mask_dir, video_name, "visual_result"), exist_ok=True)
    original_imgae_path = os.path.join(video_dir, frame_name) + ".jpg"
    original_image = matplotlib.image.imread(original_imgae_path)
    mask_imgae_path = os.path.join(output_mask_dir, video_name, "mask_result", frame_name) + ".png"
    mask_image = matplotlib.image.imread(mask_imgae_path)
    # alpha_layer = np.repeat(mask_image[:, :, np.newaxis], 3, axis=2) * 0.5

    # Box prompt visualization
    fig, ax = plt.subplots()
    ax.imshow(original_image)
    if inference_state['mask_to_box'].get(out_frame_idx) is not None:
        for object_id in object_ids_set:
            if inference_state['mask_to_box'][out_frame_idx].get(object_id) is not None:
                mask_box_coord = inference_state['mask_to_box'][out_frame_idx][object_id]['mask_box']
                if mask_box_coord[0] != 0 or mask_box_coord[2] != 0:
                    mask_point_corner = (mask_box_coord[0], mask_box_coord[1])
                    mask_box_width = mask_box_coord[2] - mask_box_coord[0]
                    mask_box_height = mask_box_coord[3] - mask_box_coord[1]
                    rect_mask_box = matplotlib.patches.Rectangle(mask_point_corner, mask_box_width, mask_box_height, linewidth=1, edgecolor='yellow', facecolor='none')
                    ax.add_patch(rect_mask_box)
    ax.imshow(mask_image, alpha=0.5)
    plt.axis('off')  # 不显示坐标轴
    output_mask_path = os.path.join(
        output_mask_dir, video_name, "visual_result", f"{frame_name}.png"
    )
    plt.savefig(output_mask_path)
    plt.close()


def construct_his_box(out_frame_idx, out_obj_id, inference_state):
    history_box = []
    start = out_frame_idx - 4
    for frame_ind in range(start, out_frame_idx+1, 1):
        if inference_state['mask_to_box'].get(frame_ind) is not None:
            if inference_state['mask_to_box'][frame_ind].get(out_obj_id) is not None:
                frame_box = inference_state['mask_to_box'][frame_ind][out_obj_id]['mask_box']
                if frame_box[0] == 0 and frame_box[2] == 0:
                    continue
                else:
                    history_box.append(inference_state['mask_to_box'][frame_ind][out_obj_id]['mask_box'])
            else:
                continue
        else:
            continue
    return history_box

def eiou_loss(box1, box2, alpha=1, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
    w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = torch.pow(inter / (union + eps), alpha)
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal squared
    rho2 = (((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4) ** alpha  # center dist ** 2
    rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
    rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
    cw2 = torch.pow(cw ** 2 + eps, alpha)
    ch2 = torch.pow(ch ** 2 + eps, alpha)
    eiou_loss = 1 - iou + rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2
    return eiou_loss[0], iou[0]  # EIou


def get_box_properties(box):
    """
        Analyze the box and identify the width, height, and centroid
    """
    x1, y1, x2, y2 = box
    b_w = x2-x1
    b_h = y2-y1
    b_cx = x1 + b_w/2
    b_cy = y1 + b_h/2
    return b_cx, b_cy, b_w, b_h

def get_mask_properties(mask):
    """
        Analyze the mask and identify the centroid
    """
    rows, cols = np.where(mask)
    if len(rows) == 0 or len(cols) == 0:
        return None
    center_y = np.mean(rows)
    center_x = np.mean(cols)
    return center_x, center_y

def need_box_judge(c_mask_ratio, c_w, c_h, c_cx, c_cy, den_c, h_w, h_h, h_cx, h_cy, den_h):
    if c_mask_ratio > 0.005:
        if abs(h_w - c_w) > h_w / 3:
            return 1
        if abs(h_h - c_h) > h_h / 3:
            return 1
        if h_w != 0 or h_h != 0:
            d_center = ((c_cx - h_cx) ** 2 + (c_cy - h_cy) ** 2) ** 0.5
            if d_center > max(h_w, h_h):
                return 1
    if abs(den_c - den_h) > min(den_c, den_h):
        return 1
    else:
        return 0

def re_seg_big(c_w, c_h, s_c_mask, h_w, h_h, s_h_mask):
    if abs(c_w - h_w) > h_w * 1 / 3:
        return 1
    elif abs(c_h - h_h) > h_h * 1 / 3:
        return 1
    elif abs(s_c_mask - s_h_mask) > 0.3 * s_h_mask:
        return 1
    else:
        return 0

def re_seg_small(den_c, s_c_mask, den_h, s_h_mask):
    if abs(den_c - den_h) > 0.5 * den_h:
        return 1
    elif abs(s_c_mask - s_h_mask) > 0.5 * s_h_mask:
        return 1
    else:
        return 0

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference(
        local_rank,
        predictor,
        box_prompt_predictor,
        base_video_dir,
        input_mask_dir,
        output_mask_dir,
        video_name,
        extend_rate_box,
        local_rate,
        middle_rate,
        backtrack_range,
        score_thresh=0.0,
        use_all_masks=False,
        per_obj_png_file=False,
    ):

    """Run VOS inference on a single video with the given predictor."""
    # load the video frames and initialize the inference state on this video
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".bmp"] and '.' not in os.path.splitext(p)[0]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


    inference_state = predictor.init_state(
        video_name=video_name, video_path=video_dir, async_loading_frames=False
    )
    # inference_state['device'] = local_rank
    height = inference_state['video_height']
    width = inference_state['video_width']
    S_img = height * width
    input_palette = None  # input_palette

    # fetch mask inputs from input_mask_dir (either only mask for the first frame, or all available masks)
    if not use_all_masks:
        # use only the first video's ground-truth mask as the input mask
        input_frame_inds = [0]
    else:
        # use all mask files available in the input_mask_dir as the input masks
        if not per_obj_png_file:
            input_frame_inds = [
                idx
                for idx, name in enumerate(frame_names)
                if os.path.exists(
                    os.path.join(input_mask_dir, video_name, f"{name}.png")
                )
            ]
        else:
            input_frame_inds = [
                idx
                for object_name in os.listdir(os.path.join(input_mask_dir, video_name))
                for idx, name in enumerate(frame_names)
                if os.path.exists(
                    os.path.join(input_mask_dir, video_name, object_name, f"{name}.png")
                )
            ]
        # check and make sure we got at least one input frame
        if len(input_frame_inds) == 0:
            raise RuntimeError(
                f"In {video_name=}, got no input masks in {input_mask_dir=}. "
                "Please make sure the input masks are available in the correct format."
            )
        input_frame_inds = sorted(set(input_frame_inds))

    # add those input masks to SAM 2 inference state before propagation
    object_ids_set = None
    for input_frame_idx in input_frame_inds: # 这里input_frame_inds=【0】
        try:
            per_obj_input_mask, input_palette = load_masks_from_dir(
                input_mask_dir=input_mask_dir,
                video_name=video_name,
                frame_name=frame_names[input_frame_idx],
                per_obj_png_file=per_obj_png_file,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"In {video_name=}, failed to load input mask for frame {input_frame_idx=}. "
                "Please add the `--track_object_appearing_later_in_video` flag "
                "for VOS datasets that don't have all objects to track appearing "
                "in the first frame (such as LVOS or YouTube-VOS)."
            ) from e
        # get the list of object ids to track from the first input frame
        if object_ids_set is None:
            object_ids_set = set(per_obj_input_mask)
        for object_id, object_mask in per_obj_input_mask.items():
            # check and make sure no new object ids appear only in later frames
            if object_id not in object_ids_set:
                raise RuntimeError(
                    f"In {video_name=}, got a new {object_id=} appearing only in a "
                    f"later {input_frame_idx=} (but not appearing in the first frame). "
                    "Please add the `--track_object_appearing_later_in_video` flag "
                    "for VOS datasets that don't have all objects to track appearing "
                    "in the first frame (such as LVOS or YouTube-VOS)."
                )
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=object_mask,
            )

            if np.sum(object_mask)/S_img < 0.0001:
                mask_obj = object_mask.astype(np.uint8) * 255
                mask_box = mask_to_box(mask_obj)
                box = np.array(mask_box, dtype=np.float32)
                center_x, center_y = get_mask_properties(object_mask)
                points = np.array([[center_x, center_y]], dtype=np.float32)
                # for labels, `1` means positive click and `0` means negative click
                labels = np.array([1], np.int32)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                    box=box,
                )

    # check and make sure we have at least one object to track
    if object_ids_set is None or len(object_ids_set) == 0:
        raise RuntimeError(
            f"In {video_name=}, got no object ids on {input_frame_inds=}. "
            "Please add the `--track_object_appearing_later_in_video` flag "
            "for VOS datasets that don't have all objects to track appearing "
            "in the first frame (such as LVOS or YouTube-VOS)."
        )

    output_video_file_dir = os.path.join(output_mask_dir, video_name)
    os.makedirs(output_video_file_dir, exist_ok=True)
    output_video_file = os.path.join(output_video_file_dir, f"{video_name}.txt")
    with open(output_video_file, 'a+', encoding='utf-8') as file:
        file.write(f"video_name: {val_video_name}; frame_num: {inference_state['num_frames']}\n")
    file.close()

    J_Mean = []
    F_Mean = []
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}

    for out_frame_idx, out_obj_ids, out_mask_logit in predictor.propagate_in_video_perobj(
            inference_state,
            local_rank,
    ):
        eps = 1e-7
        per_obj_output_mask = {
            out_obj_id: (out_mask_logit[i] > score_thresh).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        per_obj_anno_mask, anno_palette = load_masks_from_dir(
            input_mask_dir=input_mask_dir,
            video_name=video_name,
            frame_name=frame_names[out_frame_idx],
            per_obj_png_file=per_obj_png_file,
        )

        current_masks = {}
        box_prompt = {}

        for i, out_obj_id in enumerate(inference_state['obj_ids']):
            current_masks[out_obj_id] = {}
            if per_obj_output_mask.get(out_obj_id) is not None:
                mask_obj = per_obj_output_mask[out_obj_id].squeeze().astype(np.uint8) * 255  # suit for cv2.read
                current_masks[out_obj_id]['output_mask'] = per_obj_output_mask[out_obj_id].squeeze()
                s_mask = np.sum(per_obj_output_mask[out_obj_id])
                current_masks[out_obj_id]['s_mask'] = s_mask
                current_masks[out_obj_id]['mask_ratio'] = s_mask / S_img
                mask_box = mask_to_box(mask_obj)
                if mask_box[0] ==0 and mask_box[2] == 0:
                    current_masks[out_obj_id]['mask_box'] = [0, 0, 0, 0]
                else:
                    current_masks[out_obj_id]['mask_box'] = mask_box
                    mask_box_tensor = torch.tensor(mask_box).unsqueeze(0).unsqueeze(0).to(local_rank)
                    box_prompt[out_obj_id] = predictor.sam_prompt_encoder._embed_boxes(mask_box_tensor)
            else:
                current_masks[out_obj_id]['s_mask'] = 0
                current_masks[out_obj_id]['mask_ratio'] = 0
                current_masks[out_obj_id]['mask_box'] = [0, 0, 0, 0]

        if out_frame_idx == 0:
            if per_obj_anno_mask.get(out_obj_id) is not None:
                masks_void = np.zeros_like(per_obj_anno_mask[out_obj_id])
                j_metrics_res, f_metrics_res = evaluateJF(per_obj_output_mask[out_obj_id].squeeze(0), per_obj_anno_mask[out_obj_id], masks_void)
                J_Mean.append(j_metrics_res)
                F_Mean.append(f_metrics_res)
                with open(output_video_file, 'a+', encoding='utf-8') as file:
                    file.write(f"frame_num: {out_frame_idx}; obj: {out_obj_id};  JF:{(j_metrics_res + f_metrics_res) / 2} J:{j_metrics_res}; F:{f_metrics_res}\n")


        if inference_state['box_prompt_per_frame'].get(out_frame_idx) is None:
            inference_state['box_prompt_per_frame'][out_frame_idx] = {}
        if inference_state['box_prompt_per_frame'].get(out_frame_idx + 1) is None:
            inference_state['box_prompt_per_frame'][out_frame_idx + 1] = {}
            inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box'] = {}
            inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box_optim'] = {}
            inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box_kalman'] = {}
        inference_state['box_prompt_feature'][out_frame_idx] = box_prompt
        inference_state['mask_to_box'][out_frame_idx] = current_masks


        if out_frame_idx > 0:
            next_need_box_list = []
            need_re_seg_list = []
            real_re_seg_list = []
            history_masks = inference_state['mask_to_box'][out_frame_idx - 1]
            for idx, out_obj_id in enumerate(out_obj_ids):
                c_cx, c_cy, c_w, c_h = get_box_properties(current_masks[out_obj_id]['mask_box'])
                h_cx, h_cy, h_w, h_h = get_box_properties(history_masks[out_obj_id]['mask_box'])
                # c_scale_ratio = c_w / (c_h + eps)
                # h_scale_ratio = h_w / (h_h + eps)
                # d_center = ((c_cx - h_cx) ** 2 + (c_cy - h_cy) ** 2) ** 0.5
                s_c_mask = current_masks[out_obj_id]['s_mask']
                s_h_mask = history_masks[out_obj_id]['s_mask']
                # The density value of the target
                den_c = s_c_mask / (c_w * c_h + eps)
                den_h = s_h_mask / (h_w * h_h + eps)
                c_mask = current_masks[out_obj_id]['output_mask']
                h_mask = history_masks[out_obj_id]['output_mask']
                re_seg = 0
                if current_masks[out_obj_id]['mask_ratio'] > 0.005 and history_masks[out_obj_id]['mask_ratio'] > 0.005:
                    re_seg = re_seg_big(c_w, c_h, s_c_mask, h_w, h_h, s_h_mask)
                    if re_seg == 1:
                        intersection_mask = np.logical_and(c_mask, h_mask)
                        if np.sum(intersection_mask) > 0:
                            center_x, center_y = get_mask_properties(intersection_mask)
                            current_masks[out_obj_id]['point_prompt'] = [[center_x, center_y]]
                            points = np.array([[center_x, center_y]], dtype=np.float32)
                            labels = np.array([1], np.int32)
                            out_frame_idx, obj_ids, video_res_masks = predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=out_frame_idx,
                                obj_id=out_obj_id,
                                points=points,
                                labels=labels,
                            )
                            re_per_obj_output_mask = {
                                out_obj_id: (video_res_masks[i] > score_thresh).cpu().numpy()
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }
                            # current_masks[out_obj_id]['original_output_mask'] = current_masks[out_obj_id]['output_mask']
                            current_masks[out_obj_id]['output_mask'] = re_per_obj_output_mask[out_obj_id].squeeze()
                            real_re_seg_list.append(1)

                        else:
                            center_x, center_y = get_mask_properties(c_mask)
                            current_masks[out_obj_id]['point_prompt'] = [[center_x, center_y]]
                            points = np.array([[center_x, center_y]], dtype=np.float32)
                            labels = np.array([1], np.int32)
                            out_frame_idx, obj_ids, video_res_masks = predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=out_frame_idx,
                                obj_id=out_obj_id,
                                points=points,
                                labels=labels,
                            )
                            re_per_obj_output_mask = {
                                out_obj_id: (video_res_masks[i] > score_thresh).cpu().numpy()
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }
                            # current_masks[out_obj_id]['original_output_mask'] = current_masks[out_obj_id]['output_mask']
                            current_masks[out_obj_id]['output_mask'] = re_per_obj_output_mask[out_obj_id].squeeze()
                            real_re_seg_list.append(1)

                else:
                    if s_c_mask > 0 and s_h_mask > 0:
                        re_seg = re_seg_small(den_c, s_c_mask, den_h, s_h_mask)
                        if re_seg == 1:
                            if den_c > 0.2:
                                center_x, center_y = get_mask_properties(c_mask)
                                current_masks[out_obj_id]['point_prompt'] = [[center_x, center_y]]
                                points = np.array([[center_x, center_y]], dtype=np.float32)
                                labels = np.array([1], np.int32)
                                out_frame_idx, obj_ids, video_res_masks = predictor.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=out_frame_idx,
                                    obj_id=out_obj_id,
                                    points=points,
                                    labels=labels,
                                )
                                re_per_obj_output_mask = {
                                    out_obj_id: (video_res_masks[i] > score_thresh).cpu().numpy()
                                    for i, out_obj_id in enumerate(out_obj_ids)
                                }
                                current_masks[out_obj_id]['output_mask'] = re_per_obj_output_mask[out_obj_id].squeeze()
                                real_re_seg_list.append(1)

                            else:
                                if s_c_mask > 0:
                                    mask_uint8 = c_mask.astype(np.uint8) * 255
                                    kernel_size = 5
                                    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size, kernel_size))
                                    eroded_mask = cv2.erode(mask_uint8, kernel_ellipse, iterations=1)
                                    max_points = max(1, s_c_mask // 100)
                                    if max_points > 3:
                                        max_points = 3
                                    corners = cv2.goodFeaturesToTrack(
                                        image=eroded_mask,
                                        maxCorners=max_points,  # Maximum number of corner points
                                        qualityLevel=0.05,
                                        minDistance=20,
                                        blockSize=3,
                                        useHarrisDetector=False,
                                        k=0.04
                                    )
                                    if corners is not None:
                                        corners = corners.reshape(-1, 2)
                                        current_masks[out_obj_id]['point_prompt'] = corners
                                        #(row, col) -> (y, x)
                                        # corners_int = np.fliplr(corners).astype(int)
                                        labels = np.ones(len(corners), np.int32)
                                        out_frame_idx, obj_ids, video_res_masks = predictor.add_new_points_or_box(
                                            inference_state=inference_state,
                                            frame_idx=out_frame_idx,
                                            obj_id=out_obj_id,
                                            points=corners,
                                            labels=labels,
                                        )
                                        re_per_obj_output_mask = {
                                            out_obj_id: (video_res_masks[i] > score_thresh).cpu().numpy()
                                            for i, out_obj_id in enumerate(out_obj_ids)
                                        }
                                        # current_masks[out_obj_id]['original_output_mask'] = current_masks[out_obj_id]['output_mask']
                                        current_masks[out_obj_id]['output_mask'] = re_per_obj_output_mask[out_obj_id].squeeze()
                                        real_re_seg_list.append(1)

                need_re_seg_list.append(re_seg)
                if len(real_re_seg_list) < out_obj_id:
                    real_re_seg_list.append(0)
                if re_seg == 1:
                    if real_re_seg_list[idx] == 1:
                        mask_obj = current_masks[out_obj_id]['output_mask'].astype(np.uint8) * 255
                        mask_box = mask_to_box(mask_obj)
                        if mask_box[0] != 0 and mask_box[2] != 0:
                            current_masks[out_obj_id]['mask_box'] = mask_box
                            mask_box_tensor = torch.tensor(mask_box).unsqueeze(0).unsqueeze(0).to(local_rank)
                            box_prompt[out_obj_id] = predictor.sam_prompt_encoder._embed_boxes(mask_box_tensor)
                            c_cx, c_cy, c_w, c_h = get_box_properties(mask_box)
                            s_c_mask = np.sum(current_masks[out_obj_id]['output_mask'])
                            current_masks[out_obj_id]['s_mask'] = s_c_mask
                            current_masks[out_obj_id]['mask_ratio'] = s_c_mask / S_img
                            den_c = s_c_mask / (c_w * c_h + eps)

                if per_obj_anno_mask.get(out_obj_id) is not None:
                    masks_void = np.zeros_like(per_obj_anno_mask[out_obj_id])
                    # j_metrics_res, f_metrics_res = evaluateJF(per_obj_output_mask[out_obj_id].squeeze(0), per_obj_anno_mask[out_obj_id], masks_void)
                    j_metrics_res, f_metrics_res = evaluateJF(current_masks[out_obj_id]['output_mask'], per_obj_anno_mask[out_obj_id], masks_void)
                    J_Mean.append(j_metrics_res)
                    F_Mean.append(f_metrics_res)
                    with open(output_video_file, 'a+', encoding='utf-8') as file:
                        file.write(f"frame_num: {out_frame_idx}; obj: {out_obj_id};  JF:{(j_metrics_res+f_metrics_res)/2} J:{j_metrics_res}; F:{f_metrics_res}\n")
                    file.close()

                # need_box = 0
                need_box = need_box_judge(current_masks[out_obj_id]['mask_ratio'], c_w, c_h, c_cx, c_cy, den_c, h_w, h_h, h_cx, h_cy, den_h)
                next_need_box_list.append(need_box)

                if need_box == 1:
                    c_scale_ratio = c_w / (c_h + eps)
                    if out_frame_idx > 2:
                        history_boxes = construct_his_box(out_frame_idx, out_obj_id, inference_state)
                        tracker = BoxKalman(dt=1.0)
                        success = tracker.initialize(history_boxes)
                        if success:
                            k_cx, k_cy, k_w, k_h = tracker.predict()
                            k_box = [k_cx - k_w / 2, k_cy - k_h / 2, k_cx + k_w / 2, k_cy + k_h / 2]
                            k_box_tensor = torch.tensor(k_box, dtype=torch.float32).to(local_rank)
                            d_kc = ((k_cx - c_cx) ** 2 + (k_cy - c_cy) ** 2) ** 0.5
                            if current_masks[out_obj_id]['mask_ratio'] > 0.005:
                                k_scale_ratio = k_w / (k_h + eps)
                                if k_scale_ratio <= 2 * c_scale_ratio and k_scale_ratio >= 0.5 * c_scale_ratio:
                                    d_obj = (c_w ** 2 + c_h ** 2) ** 0.5
                                    if d_kc <= d_obj:
                                        inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box_kalman'][out_obj_id] = k_box_tensor
                            else:
                                if current_masks[out_obj_id]['s_mask'] > 0:
                                    if d_kc <= 2 * max(c_w, c_h):
                                        inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box_kalman'][out_obj_id] = k_box_tensor
                                else:
                                    inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box_kalman'][out_obj_id] = k_box_tensor

                    if out_frame_idx > 7 and out_frame_idx < inference_state['num_frames'] - 1:
                        image = inference_state['images'][out_frame_idx + 1].to(local_rank).float().unsqueeze(0)
                        backbone_out = predictor.forward_image(image)
                        next_img_large = backbone_out['backbone_fpn_original'][0]
                        exist, his_img_emb_local_list, his_img_emb_middle_list, his_img_emb_large_list, \
                        img_pos_enc_local, img_pos_enc_middle, img_pos_enc_large, \
                        his_box_emb_list_local, his_box_emb_list_middle, his_box_emb_list_large = construct_optim_data_reverse(
                            out_frame_idx, out_obj_id, inference_state, local_rank, local_rate, middle_rate, backtrack_range)
                        if exist == 1:
                            box_prompt_predictor.eval()
                            predict_box = box_prompt_predictor(image_embedding_local=his_img_emb_local_list,
                                                               image_embedding_middle=his_img_emb_middle_list,
                                                               image_embedding_large=his_img_emb_large_list,
                                                               next_img_large=next_img_large,
                                                               image_pe_local=img_pos_enc_local,
                                                               image_pe_middle=img_pos_enc_middle,
                                                               image_pe_large=img_pos_enc_large,
                                                               box_embedding_local=his_box_emb_list_local,
                                                               box_embedding_middle=his_box_emb_list_middle,
                                                               box_embedding_large=his_box_emb_list_large,
                                                               num_box=8,
                                                               sam_prompt_encoder=predictor.sam_prompt_encoder,
                                                               local_rank=local_rank
                                                               ).squeeze()  # 前向传播
                            # predict_box = torch.tensor(predict_box, dtype=torch.float32).to(local_rank)
                            p_box = predict_box.float().to('cpu').numpy()
                            p_cx, p_cy, p_w, p_h = get_box_properties(p_box)
                            d_pc = ((p_cx - c_cx) ** 2 + (p_cy - c_cy) ** 2) ** 0.5
                            if current_masks[out_obj_id]['mask_ratio'] > 0.005:
                                p_scale_ratio = p_w / (p_h + eps)
                                if p_scale_ratio <= 2 * c_scale_ratio and p_scale_ratio >= 0.5 * c_scale_ratio:
                                    d_obj = (c_w ** 2 + c_h ** 2) ** 0.5
                                    if d_pc <= d_obj:
                                        inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box'][out_obj_id] = predict_box
                            else:
                                if current_masks[out_obj_id]['s_mask'] > 0:
                                    if d_pc <= 2 * max(c_w, c_h):
                                        inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box'][out_obj_id] = predict_box
                    if inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box_kalman'].get(out_obj_id) is not None:
                        kalman_box = inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box_kalman'][out_obj_id]
                        if inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box'].get(out_obj_id) is not None:
                            tmp_box = inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box'][out_obj_id]
                            k_cx, k_cy, k_w, k_h = get_box_properties(kalman_box)
                            p_cx, p_cy, p_w, p_h = get_box_properties(tmp_box)
                            eiou, iou = eiou_loss(tmp_box, kalman_box)
                            if iou > 0.6: # 两者预测都很好，融合
                                f_cx = (k_cx + p_cx) / 2
                                f_cy = (k_cy + p_cy) / 2
                                f_w = (k_w + p_w) / 2
                                f_h = (k_h + p_h) / 2
                                fp_box = [f_cx - f_w / 2, f_cy - f_h / 2, f_cx + f_w / 2, f_cy + f_h / 2]
                                fp_box2 = torch.tensor(fp_box, dtype=torch.float32).to(local_rank)
                                inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box_optim'][out_obj_id] = fp_box2
                                continue
                            else:
                                continue
                        else:
                            if len(history_boxes) > 4:
                                inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box_optim'][out_obj_id] = kalman_box
                                continue
                            else:
                                continue
                    else:
                        if inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box'].get(out_obj_id) is not None:
                            if s_h_mask > 0 and s_c_mask > 0:
                                tmp_box = inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box'][out_obj_id]
                                inference_state['box_prompt_per_frame'][out_frame_idx + 1]['box_optim'][out_obj_id] = tmp_box
                                continue
                            else:
                                continue
                        else:
                            continue

            current_masks['need_box'] = next_need_box_list
            current_masks['need_re_seg'] = need_re_seg_list
            current_masks['real_re_seg'] = real_re_seg_list

        inference_state['box_prompt_feature'][out_frame_idx] = box_prompt
        inference_state['mask_to_box'][out_frame_idx] = current_masks
        # Generate box prompts for the next frame
        construct_point_inputs(out_frame_idx, out_obj_ids, inference_state, local_rank)

        # save the results
        save_masks_to_dir(
                object_ids_set = object_ids_set,
                output_mask_dir=output_video_file_dir,
                video_name=video_name,
                frame_name=frame_names[out_frame_idx],
                current_masks=current_masks,
                height=height,
                width=width,
                output_palette=output_palette,
            )

        visual_result(
                object_ids_set = object_ids_set,
                video_name= video_name,
                out_frame_idx=out_frame_idx,
                output_mask_dir=output_mask_dir,
                frame_name=frame_names[out_frame_idx],
                video_dir=video_dir,
                inference_state=inference_state,
                current_masks=current_masks,
            )

    Jm = sum(J_Mean)/len(J_Mean)
    Fm = sum(F_Mean)/len(F_Mean)
    JF = (Jm + Fm)/2
    print(f"video:{video_name}, JF{JF}, J{Jm},F{Fm}")

    return Jm, Fm, JF



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./checkpoints/sam2.1_hiera_b+.pt",
        help="path to the SAM 2 model checkpoint",
    )

    parser.add_argument( # ./DAVIS-2017-trainval-480p/DAVIS/JPEGImages/480p
        "--base_video_dir",
        type=str,
        required=True,
        help="directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument( # ./DAVIS-2017-trainval-480p/DAVIS/Annotations/480p
        "--input_mask_dir",
        type=str,
        required=True,
        help="directory containing input masks (as PNG files) of each video",
    )
    parser.add_argument(
        "--val_video_list_file",
        type=str,
        default=None,
        help="text file containing the list of video names to run VOS prediction on",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as update_sign = 0 PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--use_all_masks",
        action="store_true",
        help="whether to use all available PNG files in input_mask_dir "
        "(default without this flag: just the first PNG file as input to the SAM 2 model; "
        "usually we don't need this flag, since semi-supervised VOS evaluation usually takes input from the first frame only)",
    )
    parser.add_argument(
        "--per_obj_png_file",
        action="store_true",
        help="whether use separate per-object PNG files for input and output masks "
        "(default without this flag: all object masks are packed into a single PNG file on each frame following DAVIS format; "
        "note that the SA-V dataset stores each object mask as an individual PNG file and requires this flag)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    parser.add_argument(
        "--track_object_appearing_later_in_video",
        action="store_true",
        help="whether to track objects that appear later in the video (i.e. not on the first frame; "
        "some VOS datasets like LVOS or YouTube-VOS don't have all objects appearing in the first frame)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./train_all/checkpoints/0716_V5_val_8979.pth",
        help="path to the box predictor model checkpoint",
    )
    parser.add_argument(
        "--extend_box",
        type=float,
        default=0.0,
        help="extend rate of the box",
    )

    parser.add_argument(
        "--local_rate",
        type=int,
        required=True,
        help="local_sample_rate",
    )

    parser.add_argument(
        "--middle_rate",
        type=int,
        required=True,
        help="middle_sample_rate",
    )

    parser.add_argument(
        "--local_weight",
        type=float,
        required=True,
        help="local weight of the multi-scale feature fusion",
    )
    parser.add_argument(
        "--middle_weight",
        type=float,
        required=True,
        help="middle weight of the multi-scale feature fusion",
    )

    parser.add_argument(
        "--large_weight",
        type=float,
        required=True,
        help="large weight of the multi-scale feature fusion",
    )

    parser.add_argument(
        "--backtrack_range",
        type=int,
        required=True,
        help="backtrack range of history frame when object occluded",
    )

    args = parser.parse_args()

    # if we use per-object PNG files, they could possibly overlap in inputs and outputs
    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false" if args.per_obj_png_file else "true")
    ]

    local_rank = 'cuda:0'
    # local_rank = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(local_rank)

    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
    ).to(local_rank)
    predictor.eval()

    box_prompt_predictor = box_prompt_predictor(
        embedding_dim=256,
        num_heads=8,
        output_dim=2,
        num_box=8,
        local_weight=args.local_weight,
        middle_weight=args.middle_weight,
        large_weight=args.large_weight,
    ).to(local_rank)
    model_path = args.model_path

    # box_prompt_predictor.load_state_dict(torch.load(model_path, map_location=local_rank))
    box_prompt_predictor.load_state_dict(torch.load(model_path))
    box_prompt_predictor.to(local_rank)
    box_prompt_predictor.eval()

    print("模型加载完成！")
    if args.use_all_masks:
        print("using all available masks in input_mask_dir as input to the SAM 2 model")
    else:
        print("using only the first frame's mask in input_mask_dir as input to the SAM 2 model")
    if args.val_video_list_file is not None:
        with open(args.val_video_list_file, "r") as f:
            val_video_names = [l.strip() for l in f.readlines()]
        print(f"running VOS prediction on {len(val_video_names)}")
    else:
        video_names = [
            p
            for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        ]
        print(f"running VOS prediction on {len(video_names)} videos:\n{video_names}")

    # output file
    now = datetime.now()
    output_file_path_dir = os.path.join(args.output_mask_dir, now.strftime("%Y-%m-%d %H:%M")) # "%Y-%m-%d %H:%M:%S"
    os.makedirs(output_file_path_dir, exist_ok=True)
    output_file_path = os.path.join(output_file_path_dir, now.strftime("%Y-%m-%d %H:%M")) + ".txt"
    val_jf_result_list = []
    val_jm_result_list = []
    val_fm_result_list = []

    print(f"Start Valing! ")
    print(f"running VOS prediction on {len(val_video_names)} val_videos:\n{val_video_names}")

    for n_val_video, val_video_name in enumerate(val_video_names):
        print(f"\n{n_val_video + 1}/{int(len(val_video_names))} - running on {val_video_name}")
        Jm, Fm, JF = vos_inference(
            local_rank=local_rank,
            predictor=predictor,
            box_prompt_predictor=box_prompt_predictor,
            base_video_dir=args.base_video_dir,
            input_mask_dir=args.input_mask_dir,
            output_mask_dir=output_file_path_dir,
            video_name=val_video_name,
            extend_rate_box=args.extend_box,
            local_rate=args.local_rate,
            middle_rate=args.middle_rate,
            backtrack_range=args.backtrack_range,
            score_thresh=args.score_thresh,
            use_all_masks=args.use_all_masks,
            per_obj_png_file=args.per_obj_png_file,
        )

        with open(output_file_path, 'a+', encoding='utf-8') as file:
            file.write(f"video_name: {val_video_name}; J&F:{JF}; Jm:{Jm}; Fm:{Fm}\n")
        file.close()
        val_jf_result_list.append(JF)
        val_jm_result_list.append(Jm)
        val_fm_result_list.append(Fm)

    # total JF score
    JF_final = sum(val_jf_result_list) / len(val_jf_result_list)
    Jm_final = sum(val_jm_result_list) / len(val_jm_result_list)
    Fm_final = sum(val_fm_result_list) / len(val_fm_result_list)
    print(f"jf:{JF_final}j{Jm_final}f{Fm_final}")

    with open(output_file_path, 'a+', encoding='utf-8') as file:
        file.write("*" * 30 + "\n")
        file.write(f"jf_val:{JF_final}; jm_val:{Jm_final}; fm_val:{Fm_final}; \n")
        file.write("*" * 30 + "\n")
    file.close()

    gc.collect()
    torch.cuda.empty_cache()

    print("Val Finished!")

