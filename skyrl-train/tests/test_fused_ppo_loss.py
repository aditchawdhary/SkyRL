"""Tests for fused PPO loss kernel."""

import pytest
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
    # Safe exp with clipping
    log_ratio = log_probs - old_log_probs
    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
    ratio = torch.exp(log_ratio)
    
    # PPO clipped loss
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - eps_clip_low, 1 + eps_clip_high) * advantages
    loss = -torch.min(surr1, surr2)
    
    # Clip ratio for monitoring
    clip_ratio = ((-surr2 > -surr1).float() * loss_mask).sum() / loss_mask.sum()
    
    # Dual clip
    if use_dual_clip:
        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, loss)
        loss = torch.where(advantages < 0, clip_pg_losses2, loss)
    
    # TIS
    if use_tis:
        tis_log_ratio = old_log_probs - rollout_logprobs
        tis_log_ratio = torch.clamp(tis_log_ratio, -20.0, 20.0)
        tis_imp_ratio = torch.exp(tis_log_ratio)
        tis_imp_ratio = torch.clamp(tis_imp_ratio, max=tis_imp_ratio_cap)
        loss = loss * tis_imp_ratio
    
    # Apply mask
    loss = torch.where(loss_mask != 0.0, loss * loss_mask, torch.zeros_like(loss))
    
    return loss, clip_ratio.item()


class TestFusedPPOLoss:
    """Test suite for fused PPO loss kernel."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        torch.manual_seed(42)
        batch_size, seq_len = 4, 16
        
        log_probs = torch.randn(batch_size, seq_len, device='cuda')
        old_log_probs = torch.randn(batch_size, seq_len, device='cuda')
        advantages = torch.randn(batch_size, seq_len, device='cuda')
        loss_mask = torch.ones(batch_size, seq_len, device='cuda')
        # Mask out some positions
        loss_mask[:, -4:] = 0.0
        
        return log_probs, old_log_probs, advantages, loss_mask
    
    def test_basic_ppo_loss(self, sample_data):
        """Test basic PPO loss without dual_clip or TIS."""
        log_probs, old_log_probs, advantages, loss_mask = sample_data
        
        # Fused kernel
        fused_loss, fused_clip_ratio = fused_ppo_loss(
            log_probs, old_log_probs, advantages, loss_mask
        )
        
        # Reference implementation
        ref_loss, ref_clip_ratio = reference_ppo_loss(
            log_probs, old_log_probs, advantages, loss_mask
        )
        
        # Check correctness
        assert torch.allclose(fused_loss, ref_loss, rtol=1e-5, atol=1e-6)
        assert abs(fused_clip_ratio - ref_clip_ratio) < 1e-5
    
    def test_dual_clip(self, sample_data):
        """Test PPO loss with dual_clip enabled."""
        log_probs, old_log_probs, advantages, loss_mask = sample_data
        
        fused_loss, fused_clip_ratio = fused_ppo_loss(
            log_probs, old_log_probs, advantages, loss_mask,
            use_dual_clip=True, clip_ratio_c=3.0
        )
        
        ref_loss, ref_clip_ratio = reference_ppo_loss(
            log_probs, old_log_probs, advantages, loss_mask,
            use_dual_clip=True, clip_ratio_c=3.0
        )
        
        assert torch.allclose(fused_loss, ref_loss, rtol=1e-5, atol=1e-6)
        assert abs(fused_clip_ratio - ref_clip_ratio) < 1e-5
    
    def test_tis(self, sample_data):
        """Test PPO loss with TIS (truncated importance sampling)."""
        log_probs, old_log_probs, advantages, loss_mask = sample_data
        rollout_logprobs = torch.randn_like(log_probs)
        
        fused_loss, fused_clip_ratio = fused_ppo_loss(
            log_probs, old_log_probs, advantages, loss_mask,
            use_tis=True, rollout_logprobs=rollout_logprobs, tis_imp_ratio_cap=10.0
        )
        
        ref_loss, ref_clip_ratio = reference_ppo_loss(
            log_probs, old_log_probs, advantages, loss_mask,
            use_tis=True, rollout_logprobs=rollout_logprobs, tis_imp_ratio_cap=10.0
        )
        
        assert torch.allclose(fused_loss, ref_loss, rtol=1e-5, atol=1e-6)
        assert abs(fused_clip_ratio - ref_clip_ratio) < 1e-5
    
    def test_dual_clip_and_tis(self, sample_data):
        """Test PPO loss with both dual_clip and TIS enabled."""
        log_probs, old_log_probs, advantages, loss_mask = sample_data
        rollout_logprobs = torch.randn_like(log_probs)
        
        fused_loss, fused_clip_ratio = fused_ppo_loss(
            log_probs, old_log_probs, advantages, loss_mask,
            use_dual_clip=True, clip_ratio_c=3.0,
            use_tis=True, rollout_logprobs=rollout_logprobs, tis_imp_ratio_cap=10.0
        )
        
        ref_loss, ref_clip_ratio = reference_ppo_loss(
            log_probs, old_log_probs, advantages, loss_mask,
            use_dual_clip=True, clip_ratio_c=3.0,
            use_tis=True, rollout_logprobs=rollout_logprobs, tis_imp_ratio_cap=10.0
        )
        
        assert torch.allclose(fused_loss, ref_loss, rtol=1e-5, atol=1e-6)
        assert abs(fused_clip_ratio - ref_clip_ratio) < 1e-5
    
    def test_custom_clip_thresholds(self, sample_data):
        """Test with custom clipping thresholds."""
        log_probs, old_log_probs, advantages, loss_mask = sample_data
        
        fused_loss, _ = fused_ppo_loss(
            log_probs, old_log_probs, advantages, loss_mask,
            eps_clip_low=0.1, eps_clip_high=0.3
        )
        
        ref_loss, _ = reference_ppo_loss(
            log_probs, old_log_probs, advantages, loss_mask,
            eps_clip_low=0.1, eps_clip_high=0.3
        )
        
        assert torch.allclose(fused_loss, ref_loss, rtol=1e-5, atol=1e-6)
    
    def test_extreme_values(self):
        """Test numerical stability with extreme values."""
        torch.manual_seed(42)
        batch_size, seq_len = 2, 8
        
        # Create extreme log prob differences
        log_probs = torch.tensor([[50.0, -50.0, 0.0, 10.0, -10.0, 25.0, -25.0, 0.0]] * batch_size, device='cuda')
        old_log_probs = torch.tensor([[-50.0, 50.0, 0.0, -10.0, 10.0, -25.0, 25.0, 0.0]] * batch_size, device='cuda')
        advantages = torch.randn(batch_size, seq_len, device='cuda')
        loss_mask = torch.ones(batch_size, seq_len, device='cuda')
        
        fused_loss, _ = fused_ppo_loss(log_probs, old_log_probs, advantages, loss_mask)
        ref_loss, _ = reference_ppo_loss(log_probs, old_log_probs, advantages, loss_mask)
        
        # Should not have NaN or Inf
        assert torch.all(torch.isfinite(fused_loss))
        assert torch.allclose(fused_loss, ref_loss, rtol=1e-5, atol=1e-6)
    
    def test_gradient_flow(self, sample_data):
        """Test that gradients flow correctly through the fused kernel."""
        log_probs, old_log_probs, advantages, loss_mask = sample_data
        log_probs = log_probs.requires_grad_(True)
        advantages = advantages.requires_grad_(True)
        
        fused_loss, _ = fused_ppo_loss(log_probs, old_log_probs, advantages, loss_mask)
        total_loss = fused_loss.sum()
        total_loss.backward()
        
        # Gradients should exist and be finite
        assert log_probs.grad is not None
        assert torch.all(torch.isfinite(log_probs.grad))
        assert advantages.grad is not None
        assert torch.all(torch.isfinite(advantages.grad))
    
    def test_large_batch(self):
        """Test with realistic large batch size."""
        torch.manual_seed(42)
        batch_size, seq_len = 128, 512
        
        log_probs = torch.randn(batch_size, seq_len, device='cuda')
        old_log_probs = torch.randn(batch_size, seq_len, device='cuda')
        advantages = torch.randn(batch_size, seq_len, device='cuda')
        loss_mask = torch.ones(batch_size, seq_len, device='cuda')
        
        fused_loss, fused_clip_ratio = fused_ppo_loss(
            log_probs, old_log_probs, advantages, loss_mask
        )
        
        ref_loss, ref_clip_ratio = reference_ppo_loss(
            log_probs, old_log_probs, advantages, loss_mask
        )
        
        assert torch.allclose(fused_loss, ref_loss, rtol=1e-5, atol=1e-6)
        assert abs(fused_clip_ratio - ref_clip_ratio) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
