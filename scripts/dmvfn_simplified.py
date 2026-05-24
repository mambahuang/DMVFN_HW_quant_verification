#!/usr/bin/env python3
"""
dmvfn_simplified.py — Forward-path variants for DMVFN quality analysis.
========================================================================

Five modes expose different levels of the HW pipeline:

  'orig'        : DMVFN(training=False) — Bernoulli routing + per-block
                  progressive warp+blend.  Paper's published result.
  'no_routing'  : Same as orig but ref=[1]*9 (all blocks always active).
  'hw_faithful' : Routing-aware iterative forward → SUM ΔF^i → ONE final warp.
                  Matches what HW computes from dump data.
  'hw_algo'     : No routing, per-block warp+blend (fp32 approximation of RTL).
  'hw_quant'    : Full fixed-point simulation matching sim_dmvfn.py / RTL:
                    weight × 256 INT MAC (ConvTranspose2d, INT64 acc)
                    Q11.10 flow (FLOW_SHIFT=6, FRAC_ONE=1024)
                    fixed-point bilinear warp (border clamp)
                    linear sigmoid (MASK_SHIFT=8)
                  Output is rounded to uint8 then converted back to float.
                  Use this to get SW-equivalent HW PSNR/SSIM/LPIPS.

Usage
-----
  # A/B compare all modes:
  python scripts/dmvfn_simplified.py \\
      --val_dataset KittiValDataset \\
      --load_path   pretrained_models/dmvfn_kitti.pkl \\
      --n_samples   20

  # hw_quant only (faster):
  python scripts/dmvfn_simplified.py ... --modes hw_quant orig

  # hw_quant full test (same format as test.py):
  python scripts/test_hw_quant.py \\
      --load_path pretrained_models/dmvfn_kitti.pkl \\
      --val_datasets KittiValDataset
"""

import os
import sys
import math
import random
import argparse
import importlib

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils


# ---------------------------------------------------------------------------
# Helper: put project root on sys.path and import the repo's warp() verbatim
# ---------------------------------------------------------------------------
def _import_repo_warp():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.dirname(here), here, os.getcwd()):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    from model.arch import warp as _warp
    return _warp


# ===========================================================================
# The simplified forward paths
# ===========================================================================
class DMVFN_Simplified(nn.Module):
    """
    Wraps a real DMVFN (loaded with pretrained weights).  Replaces forward()
    with one of three explicit variants so you can pinpoint PSNR losses.
    """

    # 'orig'       — full eval forward, with Bernoulli routing (reproduces arch.py).
    # 'no_routing' — eval forward but ref=[1]*9 (all blocks active).
    # 'hw_algo'    — iterative per-block forward (like test.py fixed_point_forward,
    #                but float warp instead of fp warp), then SUM ΔF^i across
    #                blocks and final-warp once.  Matches HW's summation algorithm.
    # 'hw_faithful' — routing-aware iterative forward to build L10 state, but SUM
    #                ΔF^i across all blocks and final-warp once, matching what HW
    # 'hw_quant'   — full fixed-point simulation in numpy/torch matching sim_dmvfn.py:
    #                weight*256 INT MAC, Q11.10 flow, fixed-point bilinear warp,
    #                linear sigmoid (MASK_SHIFT). Output is uint8→float, no routing.
    MODE = 'hw_faithful'

    SCALE_LIST = [4, 4, 4, 2, 2, 2, 1, 1, 1]
    NUM_BLOCKS = 9

    def __init__(self, dmvfn_orig):
        super().__init__()
        self.dmvfn = dmvfn_orig
        self.blocks = nn.ModuleList([
            getattr(dmvfn_orig, f'block{i}') for i in range(self.NUM_BLOCKS)
        ])
        self._warp = _import_repo_warp()

    # Class-level routing_ref override: set to a list of 9 ints (0/1) to pin routing.
    # None means recompute via Bernoulli (seed=42 matching test.py).
    ROUTING_REF = None

    @torch.no_grad()
    def forward(self, x, scale=None, training=False):
        """
        x: (B, 6, H, W) in [0, 1]
        Returns list of one (B, 3, H, W) tensor (to match arch.py eval output shape).
        """
        scale = scale or self.SCALE_LIST

        if self.MODE == 'orig':
            merged = self.dmvfn(x, scale=scale, training=False)
            return merged if len(merged) > 0 else [x[:, :3]]

        if self.MODE == 'no_routing':
            return self._forward_no_routing(x, scale)

        if self.MODE == 'hw_algo':
            return self._forward_hw_algo(x, scale)

        if self.MODE == 'hw_faithful':
            return self._forward_hw_faithful(x, scale)

        if self.MODE == 'hw_quant':
            return self._forward_hw_quant(x, scale, routing_ref=self.ROUTING_REF)

        raise ValueError(f"Unknown MODE: {self.MODE}")

    # ----------------------------------------------------------------
    def _forward_no_routing(self, x, scale):
        """Original eval forward but ref is forced to [1]*9."""
        B, _, H, W = x.shape
        dev = x.device
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        flow = torch.zeros(B, 4, H, W, device=dev)
        mask = torch.zeros(B, 1, H, W, device=dev)
        warped_img0 = img0.clone()
        warped_img1 = img1.clone()

        for i in range(self.NUM_BLOCKS):
            x_in = torch.cat((img0, img1, warped_img0, warped_img1, mask), 1)
            flow_d, mask_d = self.blocks[i](x_in, flow, scale=scale[i])
            flow = flow + flow_d
            mask = mask + mask_d
            warped_img0 = self._warp(img0, flow[:, :2])
            warped_img1 = self._warp(img1, flow[:, 2:4])

        m = torch.sigmoid(mask)
        merged = warped_img0 * m + warped_img1 * (1.0 - m)
        return [torch.clamp(merged, 0.0, 1.0)]

    # ----------------------------------------------------------------
    def _forward_hw_faithful(self, x, scale):
        """
        Faithfully reproduces what the HW actually computes when fed dump data
        from test.py's routing-enabled forward:

        1. Run arch.py's eval path (with Bernoulli routing) to produce the
           per-block L10 inputs — exactly as test.py does.  Blocks skipped by
           routing contribute zero ΔF/Δmask (matching the all-zero FM dump).
        2. For each executed block, run only lastconv (TConv) to get ΔF^i.
        3. SUM all ΔF^i and Δmask^i (including zeros from skipped blocks).
        4. ONE final warp + blend on the summed flow/mask.

        This should match 'orig' PSNR very closely (< 0.01 dB difference,
        only from Bernoulli randomness).  Any gap between this and your HW
        integer sim is purely quantization + tiling error.
        """
        from model.arch import RoundSTE
        B, _, H, W = x.shape
        dev = x.device
        img0 = x[:, :3]
        img1 = x[:, 3:6]

        # --- Step 1: routing ref (use class-level override if set) ---
        if self.ROUTING_REF is not None:
            ref_list_f = [int(r) for r in self.ROUTING_REF]
        else:
            routing_vector = self.dmvfn.routing(x[:, :6]).reshape(B, -1)
            routing_vector = torch.sigmoid(self.dmvfn.l1(routing_vector))
            routing_vector = routing_vector / (routing_vector.sum(1, True) + 1e-6) * 4.5
            routing_vector = torch.clamp(routing_vector, 0, 1)
            ref_t = RoundSTE.apply(routing_vector)
            ref_list_f = [int(ref_t[0, i].item()) for i in range(self.NUM_BLOCKS)]

        # --- Step 2: iterative forward (respecting routing) to build L10 state ---
        flow_feed = torch.zeros(B, 4, H, W, device=dev)
        mask_feed = torch.zeros(B, 1, H, W, device=dev)
        warped_img0 = img0.clone()
        warped_img1 = img1.clone()

        # HW accumulator: sum of ΔF from all blocks (zeros for skipped)
        flow_sum = torch.zeros(B, 4, H, W, device=dev)
        mask_sum = torch.zeros(B, 1, H, W, device=dev)

        for i in range(self.NUM_BLOCKS):
            if ref_list_f[i]:
                x_in = torch.cat((img0, img1, warped_img0, warped_img1, mask_feed), 1)
                flow_d, mask_d = self.blocks[i](x_in, flow_feed, scale=scale[i])

                # HW accumulator
                flow_sum = flow_sum + flow_d
                mask_sum = mask_sum + mask_d

                # Update feed state (only when block is active, matching arch.py eval)
                flow_feed = flow_feed + flow_d
                mask_feed = mask_feed + mask_d
                warped_img0 = self._warp(img0, flow_feed[:, :2])
                warped_img1 = self._warp(img1, flow_feed[:, 2:4])
            # else: block skipped, no state change, no ΔF contribution

        # --- Step 3: ONE final warp + blend (HW behavior) ---
        w0_final = self._warp(img0, flow_sum[:, :2])
        w1_final = self._warp(img1, flow_sum[:, 2:4])
        m_final = torch.sigmoid(mask_sum)
        merged = w0_final * m_final + w1_final * (1.0 - m_final)
        return [torch.clamp(merged, 0.0, 1.0)]

    # ----------------------------------------------------------------
    def _forward_hw_algo(self, x, scale):
        """
        The HW's algorithmic path.

        Same iterative forward as test.py's fixed_point_forward, which gives
        us each block's L10 input (implicitly, we don't need to hook it —
        just re-run the forward but discard the per-block refinements and
        use only the summed ΔF).

        Trick: to match the HW's 'sum of 9 ΔF's, final warp once' behaviour,
        we accumulate flow_d/mask_d separately and do the final warp+blend
        just once at the end.  The per-block warped_img0/1 that feeds each
        block still has to be generated iteratively (that's what the dump
        captures).
        """
        B, _, H, W = x.shape
        dev = x.device
        img0 = x[:, :3]
        img1 = x[:, 3:6]

        # State used to feed MVFBs — iterative (matches test.py)
        flow_feed = torch.zeros(B, 4, H, W, device=dev)
        mask_feed = torch.zeros(B, 1, H, W, device=dev)
        warped_img0 = img0.clone()
        warped_img1 = img1.clone()

        # Summation of ΔF^i and Δmask^i — this is the HW-side sum
        flow_sum = torch.zeros(B, 4, H, W, device=dev)
        mask_sum = torch.zeros(B, 1, H, W, device=dev)

        for i in range(self.NUM_BLOCKS):
            x_in = torch.cat((img0, img1, warped_img0, warped_img1, mask_feed), 1)
            flow_d, mask_d = self.blocks[i](x_in, flow_feed, scale=scale[i])

            # HW's accumulator
            flow_sum = flow_sum + flow_d
            mask_sum = mask_sum + mask_d

            # Feed state for next block (same as test.py fixed_point_forward)
            flow_feed = flow_feed + flow_d
            mask_feed = mask_feed + mask_d
            warped_img0 = self._warp(img0, flow_feed[:, :2])
            warped_img1 = self._warp(img1, flow_feed[:, 2:4])

        # HW's final blend (ONE warp on the summed flow, not per-block warps)
        w0_final = self._warp(img0, flow_sum[:, :2])
        w1_final = self._warp(img1, flow_sum[:, 2:4])
        # HW uses linear sigmoid approximation: (mask >> MASK_SHIFT) + 512, clamped to [0, 1024]
        # Matches sim_dmvfn.py: mask_sh = sm_val >> MASK_SHIFT; mask_q = clamp(mask_sh + 512, 0, 1024)
        MASK_SHIFT = 8
        m_final = torch.clamp(mask_sum / (1 << MASK_SHIFT) + 0.5, 0.0, 1.0)
        merged   = w0_final * m_final + w1_final * (1.0 - m_final)
        return [torch.clamp(merged, 0.0, 1.0)]
    
    def _forward_hw_actual_per_layer(self, x, scale):
        """
        這是在模擬『硬體每一層都進行 Warp + Blend』的真實行為。
        對標你的 RTL：每過一個 Block 就出一個新的 warped_img 和 merged 結果。
        """
        B, _, H, W = x.shape
        dev = x.device
        img0 = x[:, :3]
        img1 = x[:, 3:6]

        # 初始化狀態
        flow = torch.zeros(B, 4, H, W, device=dev)
        mask = torch.zeros(B, 1, H, W, device=dev)
        warped_img0 = img0.clone()
        warped_img1 = img1.clone()
        
        # 用來存放每一層產出的融合影像 (如果你硬體每層都有輸出點)
        layer_outputs = []

        for i in range(self.NUM_BLOCKS):
            # 1. 準備輸入特徵 (17 channels)
            x_in = torch.cat((img0, img1, warped_img0, warped_img1, mask), 1)
            
            # 2. MVFB 計算殘差 (Delta Flow / Delta Mask)
            flow_d, mask_d = self.blocks[i](x_in, flow, scale=scale[i])

            # 3. 更新累加器 (Accumulator)
            flow = flow + flow_d
            mask = mask + mask_d

            # 4. 立即進行 Warp (這對應你硬體裡的 Warping Engine)
            warped_img0 = self._warp(img0, flow[:, :2])
            warped_img1 = self._warp(img1, flow[:, 2:4])

            # 5. 立即進行 Blend (這對應你硬體裡的 Alpha Blending 單元)
            m = torch.sigmoid(mask)
            merged = warped_img0 * m + warped_img1 * (1.0 - m)
            layer_outputs.append(torch.clamp(merged, 0.0, 1.0))

        # 返回最後一層的結果 (或是回傳整個 list 視你的評測需求而定)
        return [layer_outputs[-1]]

    # ----------------------------------------------------------------
    # hw_quant fixed-point helpers (mirrors sim_dmvfn.py exactly)
    # ----------------------------------------------------------------
    _FRAC_ONE     = 1 << 10              # 1024  (COORD_FRAC_W=10)
    _FLOW_SHIFT   = 6                    # 16 - COORD_FRAC_W: raw TConv acc >> 6 → Q11.10
    _MASK_SHIFT   = 8                    # matches sim_dmvfn.py / RTL config (city_config mask_shift=8)
    _WEIGHT_SCALE = 256                  # Q8.8 weight quantization
    _FM_SCALE     = 256                  # Q8.8 FM quantization
    _OC_USE       = 5                    # flow_x0, flow_y0, flow_x1, flow_y1, mask

    @staticmethod
    def _bilinear_fp_border(img_np, coords_x, coords_y):
        """
        img_np   : (H, W, 3) float32 [0,1]
        coords_x : (H_out, W_out) float32 pixel coordinates (may be OOB)
        coords_y : (H_out, W_out) float32 pixel coordinates (may be OOB)
        Returns  : (H_out, W_out, 3) float32, border clamped (align_corners=True)
        """
        H, W, _ = img_np.shape
        cx = np.clip(coords_x, 0.0, W - 1.0)
        cy = np.clip(coords_y, 0.0, H - 1.0)
        x0 = np.floor(cx).astype(np.int32)
        y0 = np.floor(cy).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, W - 1)
        y1 = np.clip(y0 + 1, 0, H - 1)
        ax = (cx - x0)[..., None]   # fractional weight
        ay = (cy - y0)[..., None]
        tl = img_np[y0, x0]
        tr = img_np[y0, x1]
        bl = img_np[y1, x0]
        br = img_np[y1, x1]
        return (tl * (1 - ax) * (1 - ay) +
                tr *      ax  * (1 - ay) +
                bl * (1 - ax) *      ay  +
                br *      ax  *      ay)

    @staticmethod
    def _bilinear_int_border(img_u8, cx_q, cy_q, frac_one=1024):
        """
        Q11.10 integer bilinear warp matching RTL / sim_dmvfn.py exactly.

        img_u8 : (H, W, 3) uint8
        cx_q   : (H_out, W_out) int64  Q11.10 signed pixel coordinates
        cy_q   : (H_out, W_out) int64  Q11.10 signed pixel coordinates
        Returns: (H_out, W_out, 3) float32 in [0, 1]
        """
        H, W = img_u8.shape[:2]
        frac_mask = frac_one - 1

        # Border clamp in Q11.10 space
        cx_c = np.clip(cx_q, 0, (W - 1) * frac_one).astype(np.int64)
        cy_c = np.clip(cy_q, 0, (H - 1) * frac_one).astype(np.int64)

        xi = (cx_c >> 10).astype(np.int32)   # integer pixel
        yi = (cy_c >> 10).astype(np.int32)
        alpha = (cx_c & frac_mask).astype(np.int64)   # fractional bits
        beta  = (cy_c & frac_mask).astype(np.int64)
        ia = frac_one - alpha
        ib = frac_one - beta

        xi1 = np.clip(xi + 1, 0, W - 1)
        yi1 = np.clip(yi + 1, 0, H - 1)

        p00 = img_u8[yi,  xi ].astype(np.int64)   # (H_out, W_out, 3)
        p10 = img_u8[yi,  xi1].astype(np.int64)
        p01 = img_u8[yi1, xi ].astype(np.int64)
        p11 = img_u8[yi1, xi1].astype(np.int64)

        ia3  = ia[..., None];  ib3  = ib[..., None]
        al3  = alpha[..., None]; be3 = beta[..., None]

        s = p00 * ia3 * ib3 + p10 * al3 * ib3 + p01 * ia3 * be3 + p11 * al3 * be3
        # round-to-nearest, shift by 2*COORD_FRAC_W=20
        out_u8 = ((s + (1 << (2 * 10 - 1))) >> (2 * 10)) & 0xFF
        return out_u8.astype(np.float32) / 255.0

    def _forward_hw_quant(self, x, scale, routing_ref=None):
        """
        Fixed-point simulation matching sim_dmvfn.py / RTL.

        Mirrors test.py exactly: routing is computed first (same Bernoulli
        sampling), then only active blocks run their forward pass and contribute
        non-zero FM.  Skipped blocks contribute zero FM/weight — identical to
        the all-zero stim dumps written by test.py for routing-skipped blocks.

        Pipeline:
          1. Compute routing ref (same as arch.py eval / test.py)
             OR use caller-supplied routing_ref (list of 9 ints, 0/1) to match
             the exact Bernoulli outcome recorded in routing_ref.txt by test.py.
          2. Iterative fp32 forward (routing-aware) with hooks → per-block L10 FM
             Skipped blocks: FM forced to zero (matches all-zero stim dump)
          3. Quantise FM × 256, weight × 256 → INT64 ConvTranspose2d MAC
             Skipped blocks: zero FM × any weight = zero contribution
          4. Sum acc_flow (4ch) and acc_mask (1ch) across all 9 blocks
          5. Flow: floor(acc >> FLOW_SHIFT) → Q11.10 integer coords
          6. Q11.10 integer bilinear warp with border clamp (matches RTL)
          7. Linear sigmoid (integer): mask_q = clip(acc>>MASK_SHIFT + 512, 0, 1024)
          8. Integer blend → uint8 → return as float32 [0,1] tensor (1,3,H,W)
        """
        from model.arch import RoundSTE

        B, _, H, W = x.shape
        assert B == 1, "hw_quant mode supports batch_size=1 only"

        FONE   = self._FRAC_ONE         # 1024
        FSHIFT = self._FLOW_SHIFT       # 6
        MSHIFT = self._MASK_SHIFT       # 8
        WSCALE = self._WEIGHT_SCALE     # 256
        FSCALE = self._FM_SCALE         # 256
        OC     = self._OC_USE           # 5

        img0_fp = x[0, :3]  # (3, H, W) float [0,1]
        img1_fp = x[0, 3:6]

        # ── Step 1: routing ref ──
        if routing_ref is not None:
            # Use caller-supplied ref (from routing_ref.txt) — exact match with test.py dump
            ref_list = [int(r) for r in routing_ref]
            print(f"  [hw_quant] Using supplied routing_ref: {ref_list}")
        else:
            # Recompute via Bernoulli — seed must match test.py's torch.manual_seed(42)
            routing_vector = self.dmvfn.routing(x[:, :6]).reshape(B, -1)
            routing_vector = torch.sigmoid(self.dmvfn.l1(routing_vector))
            routing_vector = routing_vector / (routing_vector.sum(1, True) + 1e-6) * 4.5
            routing_vector = torch.clamp(routing_vector, 0, 1)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            ref_tensor = RoundSTE.apply(routing_vector)   # (1, NUM_BLOCKS) binary
            ref_list = [int(ref_tensor[0, i].item()) for i in range(self.NUM_BLOCKS)]
            print(f"  [hw_quant] Computed routing_ref: {ref_list}")

        # ── Step 2: routing-aware iterative fp32 forward, collect L10 FMs ──
        # Hooks capture the lastconv *input* FM for each block that actually runs.
        # Skipped blocks never fire the hook → their FM stays None → treated as zero.
        l10_fms = {}   # bidx → (IC, FM_H, FM_W) float32 cpu tensor, or None

        def _make_hook(i):
            def _h(_m, inp, _o):
                l10_fms[i] = inp[0][0].detach().cpu()
            return _h

        hooks = [self.blocks[i].lastconv.register_forward_hook(_make_hook(i))
                 for i in range(self.NUM_BLOCKS)]

        flow_feed = torch.zeros(1, 4, H, W, device=x.device)
        mask_feed = torch.zeros(1, 1, H, W, device=x.device)
        warped0   = img0_fp.unsqueeze(0).clone()
        warped1   = img1_fp.unsqueeze(0).clone()

        for i in range(self.NUM_BLOCKS):
            if ref_list[i] < 1:
                # Block skipped by routing — no forward, FM = zero (matches stim)
                continue
            x_in = torch.cat((img0_fp.unsqueeze(0), img1_fp.unsqueeze(0),
                               warped0, warped1, mask_feed), 1)
            flow_d, mask_d = self.blocks[i](x_in, flow_feed, scale=scale[i])
            flow_feed = flow_feed + flow_d
            mask_feed = mask_feed + mask_d
            warped0 = self._warp(img0_fp.unsqueeze(0), flow_feed[:, :2])
            warped1 = self._warp(img1_fp.unsqueeze(0), flow_feed[:, 2:4])

        for h in hooks:
            h.remove()

        # ── Step 3: quantise FMs and weights, integer MAC, accumulate ──
        # acc_flow/acc_mask live in image space (H×W), same as TConv output.
        # Skipped blocks (FM = None) contribute zero — same as all-zero stim dump.
        acc_flow = np.zeros((4, H, W), dtype=np.int64)
        acc_mask = np.zeros((1, H, W), dtype=np.int64)

        for i in range(self.NUM_BLOCKS):
            if i not in l10_fms:
                # Block skipped by routing: zero FM → zero TConv output → skip MAC
                continue
            fm_fp = l10_fms[i].numpy()          # (IC, FM_H, FM_W) float32
            fm_q  = np.round(fm_fp * FSCALE).astype(np.int64)
            FM_H_i, FM_W_i = fm_fp.shape[1], fm_fp.shape[2]

            # ConvTranspose2d weight shape is (IC, OC, KH, KW) — opposite of Conv2d
            w_fp  = (self.blocks[i].lastconv.weight
                     .detach().cpu().numpy())    # (IC_full, OC_full, KH, KW)
            # take only OC_USE output channels (flow×4 + mask×1)
            w_fp  = w_fp[:, :OC]                # (IC_full, 5, KH, KW)
            w_q   = np.round(w_fp * WSCALE).astype(np.int64)

            # bias
            bias_np = None
            if self.blocks[i].lastconv.bias is not None:
                bias_np = (self.blocks[i].lastconv.bias
                           .detach().cpu().numpy()[:OC])  # (5,)
                bias_q  = np.round(bias_np * WSCALE).astype(np.int64)

            # TConv2d: stride=2, kernel 4×4
            # Output size: OH = (FM_H-1)*2+4, OW = (FM_W-1)*2+4
            # OH is always H+2, OW is always W+2 (one extra pixel each side).
            # Symmetric crop of 1 pixel each side gives exactly H×W.
            KH, KW = w_q.shape[2], w_q.shape[3]
            STRIDE = 2
            OH = (FM_H_i - 1) * STRIDE + KH   # = H + (KH - STRIDE) = H + 2
            OW = (FM_W_i - 1) * STRIDE + KW   # = W + (KW - STRIDE) = W + 2
            raw = np.zeros((OC, OH, OW), dtype=np.int64)

            IC_actual = fm_q.shape[0]
            for ky in range(KH):
                for kx in range(KW):
                    # w_q: (IC_full, OC, KH, KW); fm_q: (IC, FM_H, FM_W)
                    w_slice = w_q[:IC_actual, :, ky, kx]  # (IC_actual, OC)
                    contrib = np.einsum('io,ihw->ohw',
                                        w_slice, fm_q[:IC_actual])
                    raw[:, ky:ky + FM_H_i*STRIDE:STRIDE,
                           kx:kx + FM_W_i*STRIDE:STRIDE] += contrib

            if bias_np is not None:
                raw += (bias_q * WSCALE)[:, None, None]

            raw = np.clip(raw, -(1 << 31), (1 << 31) - 1)

            # Symmetric crop: remove (KH-STRIDE)//2 = 1 pixel on each edge
            pad_h = (OH - H) // 2   # always 1
            pad_w = (OW - W) // 2   # always 1
            raw = raw[:, pad_h:pad_h+H, pad_w:pad_w+W]

            acc_flow += raw[:4]
            acc_mask += raw[4:5]

        acc_flow = np.clip(acc_flow, -(1 << 31), (1 << 31) - 1)
        acc_mask = np.clip(acc_mask, -(1 << 31), (1 << 31) - 1)

        # ── Step 5: fixed-point flow → Q11.10 integer coordinates ──
        # acc >> FLOW_SHIFT gives Q11.10; pixel_base in Q11.10 = coord * FONE
        def _fp_coord_q(acc_ch, pixel_base_q):
            # floor(acc >> FLOW_SHIFT) → Q11.10 flow offset
            flow_q = (acc_ch >> FSHIFT).astype(np.int64)
            return (pixel_base_q + flow_q).astype(np.int64)

        yy_q, xx_q = [a.astype(np.int64) * FONE
                      for a in np.mgrid[0:H, 0:W]]

        cx0_q = _fp_coord_q(acc_flow[0], xx_q)   # img0 X coord in Q11.10
        cy0_q = _fp_coord_q(acc_flow[1], yy_q)   # img0 Y
        cx1_q = _fp_coord_q(acc_flow[2], xx_q)   # img1 X
        cy1_q = _fp_coord_q(acc_flow[3], yy_q)   # img1 Y

        # ── Step 6: Q11.10 integer bilinear warp (exact match with sim_dmvfn / RTL) ──
        img0_u8 = np.clip(np.round(img0_fp.permute(1, 2, 0).cpu().numpy() * 255), 0, 255).astype(np.uint8)
        img1_u8 = np.clip(np.round(img1_fp.permute(1, 2, 0).cpu().numpy() * 255), 0, 255).astype(np.uint8)

        warped0_np = self._bilinear_int_border(img0_u8, cx0_q, cy0_q, FONE)
        warped1_np = self._bilinear_int_border(img1_u8, cx1_q, cy1_q, FONE)

        # ── Step 7: linear sigmoid approximation (integer, matches RTL) ──
        # RTL: mask_sh = sm_val >> MASK_SHIFT; mask_q = clamp(mask_sh + 512, 0, 1024)
        mask_sh = (acc_mask[0] >> MSHIFT).astype(np.int64)
        mask_q  = np.clip(mask_sh + (FONE >> 1), 0, FONE).astype(np.int64)
        comp_q  = FONE - mask_q                             # (H, W) int64

        # ── Step 8: integer blend → uint8 → float tensor (exact RTL match) ──
        # _bilinear_int_border returns float32 [0,1]; convert back to uint8 for integer blend
        w0_u8 = np.clip(np.round(warped0_np * 255.0), 0, 255).astype(np.int64)
        w1_u8 = np.clip(np.round(warped1_np * 255.0), 0, 255).astype(np.int64)

        # RTL: blR = ((wR0 * mask_q + wR1 * comp) >> COORD_FRAC_W) & 0xFF
        mq3   = mask_q[..., None]    # (H, W, 1) broadcast over 3 channels
        cq3   = comp_q[..., None]
        blend_int = (w0_u8 * mq3 + w1_u8 * cq3) >> 10   # >> COORD_FRAC_W=10
        blended_u8  = np.clip(blend_int, 0, 255).astype(np.uint8)
        blended_f32 = blended_u8.astype(np.float32) / 255.0

        out = torch.from_numpy(blended_f32).permute(2, 0, 1).unsqueeze(0).to(x.device)  # (1,3,H,W)
        return [out]


# ===========================================================================
# Drop-in Model replacement (same public surface as model.model.Model)
# ===========================================================================
class Model_Simplified:
    def __init__(self, local_rank=-1, resume_path=None, resume_epoch=0,
                 load_path=None, training=False):
        from model.model import Model as _OriginalModel
        assert not training, "Model_Simplified is eval-only."
        assert load_path is not None
        self._orig = _OriginalModel(load_path=load_path, training=False)
        self._simp = DMVFN_Simplified(self._orig.dmvfn)
        dev = next(self._orig.dmvfn.parameters()).device
        self._simp = self._simp.to(dev)
        self._simp.eval()

    @torch.no_grad()
    def eval(self, imgs, name='CityValDataset',
             scale_list=(4, 4, 4, 2, 2, 2, 1, 1, 1)):
        """Mirrors model.model.Model.eval() exactly; uses self._simp underneath."""
        scale_list = list(scale_list)
        b, n, c, h, w = imgs.shape
        preds = []
        if name == 'CityValDataset':
            assert n == 14
            img0, img1 = imgs[:, 2], imgs[:, 3]
            for _ in range(5):
                merged = self._simp(torch.cat((img0, img1), 1),
                                    scale=scale_list, training=False)
                pred = merged[-1] if len(merged) > 0 else img0
                preds.append(pred)
                img0, img1 = img1, pred
        elif name in ('KittiValDataset', 'DavisValDataset'):
            assert n == 9
            img0, img1 = imgs[:, 2], imgs[:, 3]
            for _ in range(5):
                merged = self._simp(torch.cat((img0, img1), 1),
                                    scale=scale_list, training=False)
                pred = merged[-1] if len(merged) > 0 else img0
                preds.append(pred)
                img0, img1 = img1, pred
        elif name == 'VimeoValDataset':
            assert n == 3
            merged = self._simp(torch.cat((imgs[:, 0], imgs[:, 1]), 1),
                                scale=scale_list, training=False)
            pred = merged[-1] if len(merged) > 0 else imgs[:, 0]
            preds.append(pred)
        elif name == 'single_test':
            merged = self._simp(imgs[0], scale=scale_list, training=False)
            return merged[-1] if len(merged) > 0 else imgs[:, 0]
        else:
            raise ValueError(f"Unknown dataset: {name}")
        return torch.stack(preds, 1)

    def device(self):
        self._orig.device()


# ===========================================================================
# CLI: report PSNR / SSIM / LPIPS for all modes, plus cross comparisons
# ===========================================================================
def _cli_compare():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_dataset', default='KittiValDataset',
                    choices=['CityValDataset', 'KittiValDataset',
                             'DavisValDataset', 'VimeoValDataset'])
    ap.add_argument('--load_path', required=True)
    ap.add_argument('--n_samples', type=int, default=20)
    ap.add_argument('--save_imgs', action='store_true')
    ap.add_argument('--modes', nargs='+',
                    default=['orig', 'no_routing', 'hw_faithful', 'hw_algo', 'hw_quant'],
                    help='Subset of modes to run')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    sys.path.insert(0, here)

    from torch.utils.data import DataLoader
    from pytorch_msssim import ssim as _ssim
    import lpips as _lpips_lib

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    DSet = getattr(importlib.import_module('dataset.dataset'), args.val_dataset)
    val_loader = DataLoader(DSet(), batch_size=1, num_workers=1, pin_memory=False)

    model_simp = Model_Simplified(load_path=args.load_path, training=False)
    loss_fn_alex = _lpips_lib.LPIPS(net='alex').to(device)

    ALL_MODES = tuple(args.modes)

    if args.save_imgs:
        save_dir = "output_samples"
        os.makedirs(save_dir, exist_ok=True)
        print(f"[*] Images will be saved to: {save_dir}/")

    print("=" * 75)
    print(f" dmvfn_simplified  |  dataset={args.val_dataset}  samples={args.n_samples}")
    print(f" load_path={args.load_path}")
    print(f" modes: {ALL_MODES}")
    print("=" * 75)
    print(" Mode legend:")
    print("   orig        : DMVFN(training=False), Bernoulli routing + per-block warp")
    print("   no_routing  : same as orig but ref=[1]*9 (all blocks active)")
    print("   hw_faithful : routing-aware iterative forward + SUM ΔF + ONE final warp")
    print("   hw_algo     : no-routing iterative forward + per-block warp+blend")
    print("   hw_quant    : full fixed-point sim (weight×256 INT MAC, Q11.10 flow,")
    print("                 fixed-pt bilinear, linear sigmoid MASK_SHIFT=7)")
    print("=" * 75)

    # Accumulators: per-mode lists of per-prediction scalars
    psnr_acc  = {m: [] for m in ALL_MODES}
    ssim_acc  = {m: [] for m in ALL_MODES}
    lpips_acc = {m: [] for m in ALL_MODES}

    # Cross-mode PSNR comparisons (hw_quant vs others)
    xpsnr = {f'hw_quant_vs_{m}': [] for m in ALL_MODES if m != 'hw_quant'}

    for idx, (data_gpu, _) in enumerate(val_loader):
        if idx >= args.n_samples:
            break
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.

        preds_by_mode = {}
        for m in ALL_MODES:
            DMVFN_Simplified.MODE = m
            # hw_quant is CPU-heavy; keep on CPU then move to device for metrics
            pred = model_simp.eval(data_gpu, args.val_dataset)[0]
            preds_by_mode[m] = pred.to(device)

        gt = data_gpu[0]
        n_pred = preds_by_mode[ALL_MODES[0]].shape[0]

        for j in range(n_pred):
            gt_idx = 2 if args.val_dataset == 'VimeoValDataset' else (4 + j)
            target = gt[gt_idx]          # (3, H, W) float [0,1]
            t4 = target.unsqueeze(0)     # (1, 3, H, W)

            for m in ALL_MODES:
                p = preds_by_mode[m][j]
                p4 = p.unsqueeze(0)

                mse = torch.mean((target - p) ** 2).item()
                psnr_acc[m].append(-10 * math.log10(max(mse, 1e-12)))
                ssim_acc[m].append(
                    float(_ssim(t4, p4, data_range=1.0, size_average=False)))
                lpips_acc[m].append(
                    float(loss_fn_alex(
                        (t4 - 0.5) * 2.0,
                        (p4 - 0.5) * 2.0)))

            # Cross comparisons: hw_quant vs every other mode
            if 'hw_quant' in ALL_MODES:
                for m in ALL_MODES:
                    if m == 'hw_quant':
                        continue
                    key = f'hw_quant_vs_{m}'
                    mse_x = torch.mean(
                        (preds_by_mode['hw_quant'][j] - preds_by_mode[m][j]) ** 2
                    ).item()
                    xpsnr[key].append(-10 * math.log10(max(mse_x, 1e-12)))

            if args.save_imgs:
                vutils.save_image(target, f"{save_dir}/s{idx:03d}_f{j}_GT.png")
                for m in ALL_MODES:
                    vutils.save_image(preds_by_mode[m][j],
                                      f"{save_dir}/s{idx:03d}_f{j}_{m}.png")

        if (idx + 1) % 5 == 0 or idx + 1 == args.n_samples:
            n = len(psnr_acc[ALL_MODES[0]])
            parts = [f"{m}={sum(psnr_acc[m])/n:.2f}dB" for m in ALL_MODES]
            print(f"  [{idx+1:3d}/{args.n_samples}]  " + "  ".join(parts))

    n = max(len(psnr_acc[ALL_MODES[0]]), 1)
    print()
    print("=" * 75)
    print(f" RESULTS  ({n} predictions,  dataset={args.val_dataset})")
    print("=" * 75)
    print(f"  {'Mode':<14}  {'PSNR':>8}  {'SSIM':>8}  {'LPIPS':>8}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*8}")
    for m in ALL_MODES:
        p = sum(psnr_acc[m])  / n
        s = sum(ssim_acc[m])  / n
        l = sum(lpips_acc[m]) / n
        print(f"  {m:<14}  {p:8.4f}  {s:8.4f}  {l:8.4f}")
    print()

    if 'hw_quant' in ALL_MODES and xpsnr:
        print("  hw_quant cross-mode PSNR (higher = closer to that mode):")
        for key, vals in xpsnr.items():
            if vals:
                print(f"    {key:<30} {sum(vals)/len(vals):8.3f} dB")
        print()

    print("  Interpretation:")
    print("  • hw_quant vs GT  = SW model of your RTL; gap vs orig = total quantisation loss.")
    print("  • hw_quant vs hw_faithful = pure quantisation error (warp + MAC + sigmoid).")
    print("  • hw_faithful vs orig     = algorithmic diff (sum-of-ΔF vs progressive warp).")
    print("=" * 75)


if __name__ == '__main__':
    _cli_compare()