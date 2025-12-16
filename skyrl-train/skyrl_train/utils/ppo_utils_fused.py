"""PPO loss functions using fused kernels for better performance.

This module provides drop-in replacements for the loss functions in ppo_utils.py
that use torch.compile for automatic kernel fusion and improved performance.
"""

from typing import Optional, Tuple

import torch
from omegaconf import DictConfig

from skyrl_train.kernels.fused_ppo_loss import fused_ppo_loss
from skyrl_train.utils.ppo_utils import (
    PolicyLossType,
    masked_mean,
    reduce_loss,
    register_policy_loss,
)


@register_policy_loss("ppo_fused")
def ppo_policy_loss_fused(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Fused PPO policy loss using torch.compile.
    
    This is a drop-in replacement for ppo_policy_loss that uses a fused kernel
    for better performance. It supports:
    - Regular PPO clipping
    - Dual clip
    - TIS (truncated importance sampling)
    
    Args:
        log_probs: Log probabilities from current policy
        old_log_probs: Log probabilities from old policy
        advantages: Advantage estimates
        config: Training configuration
        loss_mask: Mask for valid tokens
        rollout_logprobs: Log probs from rollout policy (for TIS)
    
    Returns:
        loss: Reduced loss value
        clip_ratio: Fraction of tokens that were clipped
    """
    assert config.policy_loss_type in ["regular", "dual_clip", "ppo_fused"], \
        "loss_type must be 'regular', 'dual_clip', or 'ppo_fused'"
    
    loss_reduction = config.loss_reduction
    assert loss_reduction in [
        "token_mean",
        "sequence_mean",
        "seq_mean_token_sum_norm",
    ], "loss_reduction must be either 'token_mean', 'sequence_mean', or 'seq_mean_token_sum_norm'"
    
    # Determine if we should use dual_clip
    use_dual_clip = config.policy_loss_type == "dual_clip"
    clip_ratio_c = config.get("clip_ratio_c", 3.0) if use_dual_clip else 3.0
    
    # Determine if we should use TIS
    use_tis = config.get("use_tis", False)
    tis_imp_ratio_cap = config.get("tis_imp_ratio_cap", 10.0) if use_tis else 10.0
    
    # Call fused kernel
    per_token_loss, clip_ratio = fused_ppo_loss(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        loss_mask=loss_mask,
        eps_clip_low=config.eps_clip_low,
        eps_clip_high=config.eps_clip_high,
        use_dual_clip=use_dual_clip,
        clip_ratio_c=clip_ratio_c,
        use_tis=use_tis,
        rollout_logprobs=rollout_logprobs,
        tis_imp_ratio_cap=tis_imp_ratio_cap,
    )
    
    # Reduce loss according to config
    loss = reduce_loss(per_token_loss, loss_mask, loss_reduction, config.max_seq_len)
    
    return loss, clip_ratio


# Convenience function to check if fused kernels are available
def is_fused_kernel_available() -> bool:
    """Check if CUDA is available for fused kernels (uses torch.compile, no external deps needed)."""
    return torch.cuda.is_available()


# Convenience function to get the best available PPO loss function
def get_ppo_loss_function(use_fused: bool = True):
    """
    Get the best available PPO loss function.
    
    Args:
        use_fused: Whether to prefer fused kernel if available
    
    Returns:
        The loss function to use
    """
    if use_fused and is_fused_kernel_available():
        return ppo_policy_loss_fused
    else:
        from skyrl_train.utils.ppo_utils import ppo_policy_loss
        return ppo_policy_loss
