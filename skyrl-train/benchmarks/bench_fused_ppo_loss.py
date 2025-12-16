"""Benchmark fused PPO loss kernel vs reference implementation."""

import time
import torch

from skyrl_train.kernels.fused_ppo_loss import fused_ppo_loss


def reference_ppo_loss(
    log_probs,
    old_log_probs,
    advantages,
    loss_mask,
    eps_clip_low=0.2,
    eps_clip_high=0.2,
    use_dual_clip=False,
    clip_ratio_c=3.0,
    use_tis=False,
    rollout_logprobs=None,
    tis_imp_ratio_cap=10.0,
):
    """Reference implementation matching skyrl_train/utils/ppo_utils.py"""
    log_ratio = log_probs - old_log_probs
    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
    ratio = torch.exp(log_ratio)
    
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - eps_clip_low, 1 + eps_clip_high) * advantages
    loss = -torch.min(surr1, surr2)
    
    clip_ratio = ((-surr2 > -surr1).float() * loss_mask).sum() / loss_mask.sum()
    
    if use_dual_clip:
        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, loss)
        loss = torch.where(advantages < 0, clip_pg_losses2, loss)
    
    if use_tis:
        tis_log_ratio = old_log_probs - rollout_logprobs
        tis_log_ratio = torch.clamp(tis_log_ratio, -20.0, 20.0)
        tis_imp_ratio = torch.exp(tis_log_ratio)
        tis_imp_ratio = torch.clamp(tis_imp_ratio, max=tis_imp_ratio_cap)
        loss = loss * tis_imp_ratio
    
    loss = torch.where(loss_mask != 0.0, loss * loss_mask, torch.zeros_like(loss))
    
    return loss, clip_ratio.item()


def benchmark_ppo_loss(batch_size=128, seq_len=512, num_runs=100, warmup=10):
    """Benchmark PPO loss computation."""
    print(f"\nBenchmarking with batch_size={batch_size}, seq_len={seq_len}")
    print("=" * 70)
    
    # Generate random data
    torch.manual_seed(42)
    log_probs = torch.randn(batch_size, seq_len, device='cuda')
    old_log_probs = torch.randn(batch_size, seq_len, device='cuda')
    advantages = torch.randn(batch_size, seq_len, device='cuda')
    loss_mask = torch.ones(batch_size, seq_len, device='cuda')
    rollout_logprobs = torch.randn(batch_size, seq_len, device='cuda')
    
    # Test configurations
    configs = [
        ("Basic PPO", {"use_dual_clip": False, "use_tis": False}),
        ("PPO + Dual Clip", {"use_dual_clip": True, "use_tis": False}),
        ("PPO + TIS", {"use_dual_clip": False, "use_tis": True, "rollout_logprobs": rollout_logprobs}),
        ("PPO + Dual Clip + TIS", {"use_dual_clip": True, "use_tis": True, "rollout_logprobs": rollout_logprobs}),
    ]
    
    for config_name, config_kwargs in configs:
        print(f"\n{config_name}:")
        print("-" * 70)
        
        # Warmup
        for _ in range(warmup):
            _ = reference_ppo_loss(log_probs, old_log_probs, advantages, loss_mask, **config_kwargs)
            _ = fused_ppo_loss(log_probs, old_log_probs, advantages, loss_mask, **config_kwargs)
        torch.cuda.synchronize()
        
        # Benchmark reference implementation
        start = time.perf_counter()
        for _ in range(num_runs):
            ref_loss, ref_clip_ratio = reference_ppo_loss(
                log_probs, old_log_probs, advantages, loss_mask, **config_kwargs
            )
        torch.cuda.synchronize()
        ref_time = (time.perf_counter() - start) / num_runs
        
        # Benchmark fused implementation
        start = time.perf_counter()
        for _ in range(num_runs):
            fused_loss, fused_clip_ratio = fused_ppo_loss(
                log_probs, old_log_probs, advantages, loss_mask, **config_kwargs
            )
        torch.cuda.synchronize()
        fused_time = (time.perf_counter() - start) / num_runs
        
        # Verify correctness
        max_diff = (ref_loss - fused_loss).abs().max().item()
        
        print(f"  Reference time:  {ref_time*1000:.3f} ms")
        print(f"  Fused time:      {fused_time*1000:.3f} ms")
        print(f"  Speedup:         {ref_time/fused_time:.2f}x")
        print(f"  Max diff:        {max_diff:.2e}")
        print(f"  Correct:         {max_diff < 1e-5}")


def benchmark_memory():
    """Compare memory usage between implementations."""
    print("\n" + "=" * 70)
    print("Memory Analysis")
    print("=" * 70)
    
    batch_size, seq_len = 128, 512
    element_size = 4  # float32
    
    # Reference creates multiple intermediate tensors
    intermediates = [
        ("log_ratio", batch_size * seq_len * element_size),
        ("ratio", batch_size * seq_len * element_size),
        ("surr1", batch_size * seq_len * element_size),
        ("surr2", batch_size * seq_len * element_size),
        ("clipped_ratio", batch_size * seq_len * element_size),
    ]
    
    total_intermediate = sum(size for _, size in intermediates)
    
    print(f"\nReference implementation intermediate tensors:")
    for name, size in intermediates:
        print(f"  {name:20s}: {size / 1e6:.2f} MB")
    print(f"  {'Total':20s}: {total_intermediate / 1e6:.2f} MB")
    
    print(f"\nFused implementation:")
    print(f"  No intermediate tensors materialized")
    print(f"  Memory reduction: {total_intermediate / 1e6:.2f} MB saved")


def benchmark_backward():
    """Benchmark backward pass."""
    print("\n" + "=" * 70)
    print("Backward Pass Benchmark")
    print("=" * 70)
    
    batch_size, seq_len = 128, 512
    num_runs = 50
    warmup = 10
    
    torch.manual_seed(42)
    log_probs_ref = torch.randn(batch_size, seq_len, device='cuda', requires_grad=True)
    old_log_probs = torch.randn(batch_size, seq_len, device='cuda')
    advantages = torch.randn(batch_size, seq_len, device='cuda')
    loss_mask = torch.ones(batch_size, seq_len, device='cuda')
    
    log_probs_fused = log_probs_ref.clone().detach().requires_grad_(True)
    
    # Warmup
    for _ in range(warmup):
        ref_loss, _ = reference_ppo_loss(log_probs_ref, old_log_probs, advantages, loss_mask)
        ref_loss.sum().backward()
        log_probs_ref.grad = None
        
        fused_loss, _ = fused_ppo_loss(log_probs_fused, old_log_probs, advantages, loss_mask)
        fused_loss.sum().backward()
        log_probs_fused.grad = None
    torch.cuda.synchronize()
    
    # Benchmark reference
    start = time.perf_counter()
    for _ in range(num_runs):
        ref_loss, _ = reference_ppo_loss(log_probs_ref, old_log_probs, advantages, loss_mask)
        ref_loss.sum().backward()
        log_probs_ref.grad = None
    torch.cuda.synchronize()
    ref_time = (time.perf_counter() - start) / num_runs
    
    # Benchmark fused
    start = time.perf_counter()
    for _ in range(num_runs):
        fused_loss, _ = fused_ppo_loss(log_probs_fused, old_log_probs, advantages, loss_mask)
        fused_loss.sum().backward()
        log_probs_fused.grad = None
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / num_runs
    
    print(f"\nForward + Backward:")
    print(f"  Reference time:  {ref_time*1000:.3f} ms")
    print(f"  Fused time:      {fused_time*1000:.3f} ms")
    print(f"  Speedup:         {ref_time/fused_time:.2f}x")


if __name__ == "__main__":
    print("Fused PPO Loss Kernel Benchmark")
    print("=" * 70)
    
    # Test different batch sizes
    configs = [
        (32, 256),    # Small
        (64, 512),    # Medium
        (128, 512),   # Large (typical)
        (256, 512),   # Very large
    ]
    
    for batch_size, seq_len in configs:
        benchmark_ppo_loss(batch_size, seq_len, num_runs=100)
    
    benchmark_memory()
    benchmark_backward()
