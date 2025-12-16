"""Fused PPO loss kernel using torch.compile for performance optimization."""

import torch


@torch.compile
def fused_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    eps_clip_low: float = 0.2,
    eps_clip_high: float = 0.2,
    use_dual_clip: bool = False,
    clip_ratio_c: float = 3.0,
    use_tis: bool = False,
    rollout_logprobs: torch.Tensor | None = None,
    tis_imp_ratio_cap: float = 10.0,
) -> tuple[torch.Tensor, float]:
    """
    Fused PPO loss computation using torch.compile.
    
    This fuses all PPO loss operations for better performance:
    - torch.compile automatically fuses operations into efficient kernels
    - Avoids materializing unnecessary intermediate tensors
    - Better memory bandwidth utilization
    - No external dependencies (Triton, etc.)
    
    Args:
        log_probs: [batch_size, seqlen] log probabilities from current policy
        old_log_probs: [batch_size, seqlen] log probabilities from old policy
        advantages: [batch_size, seqlen] advantage estimates
        loss_mask: [batch_size, seqlen] mask for valid tokens
        eps_clip_low: Lower clipping threshold (default: 0.2)
        eps_clip_high: Upper clipping threshold (default: 0.2)
        use_dual_clip: Whether to use dual clipping (default: False)
        clip_ratio_c: Dual clip ratio constant (default: 3.0)
        use_tis: Whether to use truncated importance sampling (default: False)
        rollout_logprobs: [batch_size, seqlen] log probs from rollout policy (required if use_tis=True)
        tis_imp_ratio_cap: Maximum TIS importance ratio (default: 10.0)
    
    Returns:
        loss: Per-token losses [batch_size, seqlen]
        clip_ratio: Fraction of tokens that were clipped (scalar)
    """
    # Compute log_ratio with safe clipping to avoid overflow
    log_ratio = torch.clamp(log_probs - old_log_probs, -20.0, 20.0)
    
    # Compute ratio = exp(log_ratio)
    ratio = torch.exp(log_ratio)
    
    # Compute surr1 = ratio * advantages
    surr1 = ratio * advantages
    
    # Compute clipped ratio and surr2
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip_low, 1.0 + eps_clip_high)
    surr2 = clipped_ratio * advantages
    
    # Compute loss = -min(surr1, surr2)
    loss = -torch.min(surr1, surr2)
    
    # Track clipping for metrics
    is_clipped = (-surr2 > -surr1).float()
    
    # Apply dual_clip if enabled
    if use_dual_clip:
        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, loss)
        loss = torch.where(advantages < 0.0, clip_pg_losses2, loss)
    
    # Apply TIS (truncated importance sampling) if enabled
    if use_tis:
        assert rollout_logprobs is not None, "rollout_logprobs required when use_tis=True"
        tis_log_ratio = torch.clamp(old_log_probs - rollout_logprobs, -20.0, 20.0)
        tis_imp_ratio = torch.clamp(torch.exp(tis_log_ratio), max=tis_imp_ratio_cap)
        loss = loss * tis_imp_ratio
    
    # Apply mask
    loss = torch.where(loss_mask != 0.0, loss * loss_mask, torch.zeros_like(loss))
    
    # Compute clip ratio for monitoring
    clip_ratio = (is_clipped * loss_mask).sum() / loss_mask.sum()
    
    return loss, clip_ratio.item()
