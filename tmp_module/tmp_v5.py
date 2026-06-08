# Copyright (c) SJTU, Z.Wu.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/jone-Wu/LSP-SAM2_v2
import warnings
import contextlib

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, Type

"""
    25.10.12  V5
    8xlocal + 4xmiddle + 1xlarge
    next_img_feature_scale: large
    dropout: Attn：0.1、MLP：0.0
    cls token initialization：0
"""

from sam2.utils.misc import get_sdpa_settings
warnings.simplefilter(action="ignore", category=FutureWarning)
# Check whether Flash Attention is available (and use it by default)
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()
# A fallback setting to allow all available kernels if Flash Attention fails
ALLOW_ALL_KERNELS = False

def sdp_kernel_context(dropout_p):
    """
    Get the context for the attention scaled dot-product kernel. We use Flash Attention
    by default, but fall back to all available kernels if Flash Attention fails.
    """
    if ALLOW_ALL_KERNELS:
        return contextlib.nullcontext()

    return torch.backends.cuda.sdp_kernel(
        enable_flash=USE_FLASH_ATTN,
        # if Flash attention kernel is off, then math kernel needs to be enabled
        enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
        enable_mem_efficient=OLD_GPU,
    )

class box_prompt_predictor(nn.Module):
    def __init__(self,
                 # depth: int,
                 embedding_dim: int,
                 num_heads: int,
                 num_box: int,
                 local_weight: float,
                 middle_weight: float,
                 large_weight: float,
                 activation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2,
                 output_dim = 2
                 ) -> None:

        super(box_prompt_predictor, self).__init__()
        # self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_box =num_box
        # weighting factor
        self.local_weight = local_weight
        self.middle_weight = middle_weight
        self.large_weight = large_weight
        self.cross_attn_box_to_image_large = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_large = nn.LayerNorm(embedding_dim)
        self.cross_attn_box_to_image_middle = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_middle = nn.LayerNorm(embedding_dim)
        self.cross_attn_box_to_image_local = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_local = nn.LayerNorm(embedding_dim)
        self.self_attn_local = Attention(embedding_dim, num_heads)
        self.norm_local2 = nn.LayerNorm(embedding_dim)
        self.self_attn_middle = Attention(embedding_dim, num_heads)
        self.norm_middle2 = nn.LayerNorm(embedding_dim)
        self.norm_result = nn.LayerNorm(embedding_dim)
        self.cross_attn_box_to_image2 = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_predict = nn.LayerNorm(embedding_dim)
        self.final_decoder = MLP(dim_input=512, hidden_size=256, hidden_size2=64, dim_output=4)

    def forward(self,
                image_embedding_local: Tensor,
                image_embedding_middle: Tensor,
                image_embedding_large: Tensor,
                next_img_large,
                image_pe_local: Tensor,
                image_pe_middle: Tensor,
                image_pe_large: Tensor,
                box_embedding_local: Tensor,
                box_embedding_middle: Tensor,
                box_embedding_large: Tensor,
                num_box: int,
                sam_prompt_encoder,
                local_rank
                ) -> Tuple[Tensor, Tensor]:

        # bs, c, h, w = image_embedding.shape
        # bs_b, n_b, c_b = box_embedding.shape
        image_embedding_local = image_embedding_local.flatten(2).permute(0, 2, 1)
        image_embedding_middle = image_embedding_middle.flatten(2).permute(0, 2, 1)
        image_embedding_large = image_embedding_large.flatten(2).permute(0, 2, 1)
        next_img_large = next_img_large.flatten(2).permute(0, 2, 1)
        image_pe_local = image_pe_local.flatten(2).permute(0, 2, 1)
        image_pe_middle = image_pe_middle.flatten(2).permute(0, 2, 1)
        image_pe_large = image_pe_large.flatten(2).permute(0, 2, 1)

        # fusion embedding local x8
        fusion_features_local_list = []
        num_local = image_embedding_local.shape[0]
        for i in range(num_local):
            queries_local = box_embedding_local[i, :, :].unsqueeze(dim=0)
            kv_local = (image_embedding_local[i, :, :].unsqueeze(dim=0) + image_pe_local)
            fusion_features_local1 = self.cross_attn_box_to_image_local(q=queries_local, k=kv_local, v=kv_local)
            fusion_features_local2 = queries_local + fusion_features_local1
            fusion_features_local = self.norm_local(fusion_features_local2)
            fusion_features_local_list.append(fusion_features_local)
        fusion_features_local_list = torch.cat(fusion_features_local_list, dim=0)

        # middle x4
        fusion_features_middle_list = []
        num_middle = image_embedding_middle.shape[0]
        # box_embedding_middle = box_embedding[bs_b-4:bs_b, :, :]
        for i in range(num_middle):
            queries_middle = box_embedding_middle[i, :, :].unsqueeze(dim=0)
            kv_middle = image_embedding_middle[i, :, :].unsqueeze(dim=0) + image_pe_middle
            fusion_features_middle1 = self.cross_attn_box_to_image_middle(q=queries_middle, k=kv_middle, v=kv_middle)
            fusion_features_middle2 = queries_middle + fusion_features_middle1
            fusion_features_middle = self.norm_middle(fusion_features_middle2)
            fusion_features_middle_list.append(fusion_features_middle)
        fusion_features_middle_list = torch.cat(fusion_features_middle_list, dim=0)

        # large x1
        queries_large = box_embedding_large[0, :, :].unsqueeze(dim=0)
        kv_large = image_embedding_large[0, :, :].unsqueeze(dim=0) + image_pe_large
        fusion_features_large1 = self.cross_attn_box_to_image_large(q=queries_large, k=kv_large, v=kv_large)
        fusion_features_large2 = queries_large + fusion_features_large1
        result_large = self.norm_large(fusion_features_large2)

        cls_coords = torch.tensor([0, 0, 0, 0]).to(local_rank)
        with torch.no_grad():
            cls_token = sam_prompt_encoder._embed_boxes(cls_coords)
        self_attn_qkv_local = torch.cat((cls_token, fusion_features_local_list), dim=0)
        result_local_list = self.self_attn_local(q=self_attn_qkv_local, k=self_attn_qkv_local, v=self_attn_qkv_local)
        result_local = self.norm_local2(result_local_list[0].unsqueeze(dim=0)) # （1, 2, 256）

        self_attn_qkv_middle = torch.cat((cls_token, fusion_features_middle_list), dim=0)
        result_middle_list = self.self_attn_middle(q=self_attn_qkv_middle, k=self_attn_qkv_middle, v=self_attn_qkv_middle)
        result_middle = self.norm_middle2(result_middle_list[0].unsqueeze(dim=0)) # （1, 2, 256）

        result = self.local_weight * result_local + self.middle_weight * result_middle + self.large_weight * result_large
        result2 = self.norm_result(result)

        latest_image_embedding = next_img_large + image_pe_large
        box_predict1 = self.cross_attn_box_to_image2(q=result2, k=latest_image_embedding, v=latest_image_embedding)
        box_predict2 = box_predict1 + result2
        box_predict = self.norm_predict(box_predict2).view(1,-1)
        final_box_predict = self.final_decoder(box_predict)

        return final_box_predict


class Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.1,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            # Fall back to all kernels if the Flash attention kernel fails
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class MLP(nn.Module):
    def __init__(self, dim_input, hidden_size, hidden_size2, dim_output):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim_input, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, dim_output)
        # self.active = nn.Hardswish()

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.act1(out1)
        out3 = self.fc2(out2)
        out4 = self.act2(out3)
        out = self.fc3(out4)
        # return self.active(out)
        return out











