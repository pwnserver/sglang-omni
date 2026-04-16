"""Vendor wrapper for sglang.srt.layers.*

Centralize third-party imports and apply monkey patches here.

Patches applied (sm_121a / GB10 Blackwell):
  sgl_kernel ships pre-compiled CUDA kernels that do NOT include sm_121a.
  On unsupported GPUs we monkey-patch MultiPlatformOp.__init_subclass__ so
  that every subclass (RMSNorm, RotaryEmbedding, …) dispatches to its
  forward_native instead of forward_cuda.  This is a single global fix
  that covers all current and future MultiPlatformOp-based layers.
These patches can be removed once sgl_kernel ships sm_121a binaries.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import torch

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global MultiPlatformOp patch: force forward_native on unsupported GPUs
# ---------------------------------------------------------------------------
# sgl_kernel 0.3.x pre-compiled CUDA kernels cover up to sm_120.
# sm_121 (GB10 Blackwell) is NOT included → every sgl_kernel op raises
# "no kernel image is available for execution on the device".
#
# MultiPlatformOp.__init__ sets self._forward_method = self.forward_cuda
# on CUDA devices.  We patch it so that on unsupported GPUs it falls back
# to self.forward_native (pure PyTorch, correct, slightly slower).
_SGL_KERNEL_MAX_SM = 120


def _device_needs_native_fallback() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    return sm > _SGL_KERNEL_MAX_SM


_NEEDS_NATIVE = _device_needs_native_fallback()

if _NEEDS_NATIVE:
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

    _orig_mp_init = MultiPlatformOp.__init__

    def _patched_mp_init(self, *args, **kwargs):
        _orig_mp_init(self, *args, **kwargs)
        # After original __init__ sets _forward_method to forward_cuda,
        # override it with forward_native if available.
        if hasattr(self, "forward_native"):
            self._forward_method = self.forward_native

    MultiPlatformOp.__init__ = _patched_mp_init
    _logger.info(
        "MultiPlatformOp patched: forward_cuda → forward_native "
        "(sgl_kernel lacks sm_%d%d support)",
        *torch.cuda.get_device_capability(),
    )

from sgl_kernel import top_k_top_p_sampling_from_probs
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding, get_rope
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

# ---------------------------------------------------------------------------
# RMSNorm.forward_with_allreduce_fusion monkey-patch
# ---------------------------------------------------------------------------
_orig_forward_with_allreduce_fusion = RMSNorm.forward_with_allreduce_fusion


def _patched_forward_with_allreduce_fusion(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if residual is not None:
        from sglang.srt.distributed import (
            get_tensor_model_parallel_world_size,
            tensor_model_parallel_all_reduce,
        )
        from sglang.srt.layers.flashinfer_comm_fusion import (
            flashinfer_allreduce_residual_rmsnorm,
        )

        if get_tensor_model_parallel_world_size() > 1:
            fused_result = flashinfer_allreduce_residual_rmsnorm(
                input_tensor=x,
                residual=residual,
                weight=self.weight,
                eps=self.variance_epsilon,
            )
            if fused_result[0] is not None:
                return fused_result

            x = tensor_model_parallel_all_reduce(x)
            return self.forward(
                x,
                residual,
                post_residual_addition=post_residual_addition,
                **kwargs,
            )

    return self.forward(
        x,
        residual,
        post_residual_addition=post_residual_addition,
        **kwargs,
    )


RMSNorm.forward_with_allreduce_fusion = _patched_forward_with_allreduce_fusion

__all__ = [
    "RadixAttention",
    "VocabParallelEmbedding",
    "MRotaryEmbedding",
    "get_rope",
    "get_layer_id",
    "RMSNorm",
    "SiluAndMul",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "ReplicatedLinear",
    "RowParallelLinear",
    "TopK",
    "get_moe_a2a_backend",
    "should_use_flashinfer_cutlass_moe_fp4_allgather",
    "get_moe_impl_class",
    "RoutingMethodType",
    "get_attention_tp_rank",
    "get_attention_tp_size",
    "QuantizationConfig",
    "LayerCommunicator",
    "LayerScatterModes",
    "FusedMoE",
    "top_k_top_p_sampling_from_probs",
]
