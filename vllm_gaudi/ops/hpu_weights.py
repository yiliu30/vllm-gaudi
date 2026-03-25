import torch

import vllm.model_executor.model_loader.utils as hpu_utils
import vllm.model_executor.model_loader.base_loader as base_loader
import vllm.model_executor.parameter as vllm_parameter
import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.attention import (Attention, MLAAttention)
from vllm.model_executor.model_loader.reload import set_torchao_reload_attrs


def hpu_process_weights_after_loading(model, model_config, target_device):
    """Gaudi override: accept device strings (e.g., "hpu")."""
    target_device = torch.device(target_device)
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            #with device_loading_context(module, target_device):
            quant_method.process_weights_after_loading(module)

    # Initialize post-load attention weights for both Attention and MLA.
    for _, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)) and hasattr(module, "process_weights_after_loading"):
            #with device_loading_context(module, target_device):
            module.process_weights_after_loading(model_config.dtype)

    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)


hpu_utils.process_weights_after_loading = hpu_process_weights_after_loading
base_loader.process_weights_after_loading = hpu_process_weights_after_loading

# ---------------------------------------------------------------------------
# Fix: per-channel FP8 weight-scale loading
#
# Some FP8 checkpoints (e.g. Qwen3-MoE) store per-channel scales as 1-D
# tensors per expert (shape [N]), while vLLM allocates the parameter buffer
# with a trailing size-1 dim ([N, 1]) for kernel compatibility.
# The upstream copy_ calls assert identical shapes and therefore crash.
# We patch both the FusedMoE._load_per_channel_weight_scale method and the
# _ColumnvLLMParameter / BasevLLMParameter load helpers to use reshape()
# instead of a hard shape assertion.  reshape() is a no-op when shapes already
# match and raises on genuine element-count mismatches.
# ---------------------------------------------------------------------------


def _hpu_load_per_channel_weight_scale(self, expert_data, shard_dim, shard_id, loaded_weight, tp_rank):
    if shard_id == "w2":
        expert_data.copy_(loaded_weight.reshape(expert_data.shape))
    elif shard_id in ("w1", "w3"):
        # Align ndim so that _load_w13's narrow() and copy_() work when
        # loaded_weight is 1D but expert_data has a trailing size-1 dim.
        while loaded_weight.ndim < expert_data.ndim:
            loaded_weight = loaded_weight.unsqueeze(-1)
        self._load_w13(
            shard_id=shard_id,
            shard_dim=shard_dim,
            loaded_weight=loaded_weight,
            expert_data=expert_data,
            tp_rank=tp_rank,
        )


fused_moe_layer.FusedMoE._load_per_channel_weight_scale = _hpu_load_per_channel_weight_scale


def _hpu_assert_and_load(self, loaded_weight: torch.Tensor):
    self.data.copy_(loaded_weight.reshape(self.data.shape))


def _hpu_load_column_parallel_weight(self, loaded_weight: torch.Tensor):
    shard_size = self.data.shape[self.output_dim]
    loaded_weight = loaded_weight.narrow(self.output_dim, self.tp_rank * shard_size, shard_size)
    self.data.copy_(loaded_weight.reshape(self.data.shape))


def _hpu_load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
    from vllm.model_executor.parameter import PackedColumnParameter, PackedvLLMParameter
    shard_offset = kwargs.get("shard_offset")
    shard_size = kwargs.get("shard_size")
    if (isinstance(self, (PackedColumnParameter, PackedvLLMParameter)) and self.packed_dim == self.output_dim):
        shard_size, shard_offset = self.adjust_shard_indexes_for_packing(shard_offset=shard_offset,
                                                                         shard_size=shard_size)
    param_data = self.data
    param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
    loaded_weight = loaded_weight.narrow(self.output_dim, self.tp_rank * shard_size, shard_size)
    param_data.copy_(loaded_weight.reshape(param_data.shape))


def _hpu_load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
    from vllm.model_executor.parameter import PackedColumnParameter, PackedvLLMParameter
    shard_offset = kwargs.get("shard_offset")
    shard_size = kwargs.get("shard_size")
    shard_id = kwargs.get("shard_id")
    num_heads = kwargs.get("num_heads")
    if (isinstance(self, (PackedColumnParameter, PackedvLLMParameter)) and self.output_dim == self.packed_dim):
        shard_size, shard_offset = self.adjust_shard_indexes_for_packing(shard_offset=shard_offset,
                                                                         shard_size=shard_size)
    param_data = self.data
    shard_id = self.tp_rank if shard_id == "q" else self.tp_rank // num_heads
    param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
    loaded_weight = loaded_weight.narrow(self.output_dim, shard_id * shard_size, shard_size)
    param_data.copy_(loaded_weight.reshape(param_data.shape))


def _hpu_load_row_parallel_weight(self, loaded_weight: torch.Tensor):
    shard_size = self.data.shape[self.input_dim]
    loaded_weight = loaded_weight.narrow(self.input_dim, self.tp_rank * shard_size, shard_size)
    if len(loaded_weight.shape) == 0:
        loaded_weight = loaded_weight.reshape(1)
    self.data.copy_(loaded_weight.reshape(self.data.shape))


_ColumnvLLMParameter = vllm_parameter._ColumnvLLMParameter
_ColumnvLLMParameter.load_column_parallel_weight = _hpu_load_column_parallel_weight
_ColumnvLLMParameter.load_merged_column_weight = _hpu_load_merged_column_weight
_ColumnvLLMParameter.load_qkv_weight = _hpu_load_qkv_weight

vllm_parameter.BasevLLMParameter._assert_and_load = _hpu_assert_and_load
vllm_parameter.RowvLLMParameter.load_row_parallel_weight = _hpu_load_row_parallel_weight
