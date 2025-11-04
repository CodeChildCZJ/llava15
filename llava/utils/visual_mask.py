# llava/utils/visual_mask.py
import torch

def build_visual_attention_mask(
    pad_mask_2d: torch.Tensor,          # (B, K) 1=valid, 0=pad
    is_image_mask_2d: torch.Tensor,     # (B, K) True for image embeddings
    past_key_values,                    # None for prefill; not None for generate
    mask_visual_first: bool,            # 屏蔽 prefill 最后一个 token 对图像的注意力
    mask_visual_pred: bool,              # 屏蔽 generate 每步对图像的注意力
    pure_text: bool = False             # 如果是纯文本，则全程屏蔽图像 token
) -> torch.Tensor:
    """
    返回 (B, 1, Q, K) 的 mask：
      1.0 → 可见（keep）
      0.0 → 屏蔽（mask）
    Hugging Face 会在 SDPA 内部反转并加 -∞。
    """
    assert pad_mask_2d.dim() == 2 and is_image_mask_2d.shape == pad_mask_2d.shape
    B, K = pad_mask_2d.shape
    device = pad_mask_2d.device

    key_ok = pad_mask_2d[:, None, None, :].to(torch.bool)  # (B,1,1,K)
    allow = key_ok.clone()

    if past_key_values is None:
        # ===== Prefill =====
        L = K
        causal = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool))[None, None, :, :]
        allow = causal & key_ok  # (B,1,L,L)

        if mask_visual_first:
            # 找 最后 一个 有效 query
            valid_counts = pad_mask_2d.long().sum(dim=1)  # (B,)
            last_idx = (valid_counts - 1).clamp(min=0)
            q_sel = (torch.arange(L, device=device)[None, :] == last_idx[:, None])  # (B,L)
            block = ~(q_sel[:, None, :, None] & is_image_mask_2d[:, None, None, :])
            allow = allow & block
    else:
        # ===== Generate =====
        valid_counts = pad_mask_2d.long().sum(dim=1)
        last_idx = (valid_counts - 1).clamp(min=0)
        keys_allowed = (torch.arange(K, device=device)[None, :] <= last_idx[:, None])
        allow = keys_allowed[:, None, None, :].to(torch.bool) & key_ok
        if mask_visual_pred:
            allow = allow & (~is_image_mask_2d[:, None, None, :])

    if pure_text:
        # 所有 query 都不看图像 token
        allow = allow & (~is_image_mask_2d[:, None, None, :])

    # 输出 float 型 1/0 mask
    return allow
