import os
import cv2
import sys
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from utils.util import *
from model.model import Model
import model.arch as dmvfn_arch

os.makedirs('./data', exist_ok=True)

# ==========================================
# Force all 9 blocks active in inference mode
# ==========================================
# DMVFN uses adaptive routing (RoundSTE Bernoulli): in training=False each
# block i only runs when ref[0,i]==1.  For stimulus generation we need ALL
# blocks to execute, so we replace RoundSTE with an identity that always
# returns 1.  This patch must be applied before Model() is constructed.
class _ForceOne(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, x):
        return torch.ones_like(x)
    @staticmethod
    def backward(_ctx, grad):
        return grad

dmvfn_arch.RoundSTE = _ForceOne


# ==========================================
# Fixed-point serialization helpers
# ==========================================
def _q(val):
    """Float → Q8.8 integer."""
    return int(round(float(val) * 256))


def _pad_weight_conv2d(w_np, target_oc, target_ic):
    """(OC, IC, KH, KW) → zero-padded to (target_OC, target_IC, KH, KW)."""
    oc, ic, kh, kw = w_np.shape
    out = np.zeros((target_oc, target_ic, kh, kw), dtype=w_np.dtype)
    out[:min(oc, target_oc), :min(ic, target_ic)] = w_np[:target_oc, :target_ic]
    return out


def _pad_weight_tconv2d(w_np, target_ic, target_oc):
    """(IC, OC, KH, KW) → zero-padded to (target_IC, target_OC, KH, KW)."""
    ic, oc, kh, kw = w_np.shape
    out = np.zeros((target_ic, target_oc, kh, kw), dtype=w_np.dtype)
    out[:min(ic, target_ic), :min(oc, target_oc)] = w_np[:target_ic, :target_oc]
    return out


def write_weight_conv2d_txt(filepath, w_int):
    """Write (OC, IC, KH, KW) weight in KK-major order expected by gen_stim."""
    n_oc, n_ic, kh, kw = w_int.shape
    with open(filepath, 'w') as f:
        for k in range(kh * kw):
            ky, kx = k // kw, k % kw
            for oc in range(n_oc):
                f.write(" ".join(str(int(w_int[oc, ic, ky, kx]))
                                 for ic in range(n_ic)) + "\n")
    print(f"  [Conv2d  ] {os.path.basename(filepath)}: OC={n_oc} IC={n_ic} K={kh}x{kw}")


def write_weight_tconv2d_txt(filepath, w_int):
    """Write (IC, OC, KH, KW) weight in KK-major order expected by gen_stim."""
    n_ic, n_oc, kh, kw = w_int.shape
    with open(filepath, 'w') as f:
        for k in range(kh * kw):
            ky, kx = k // kw, k % kw
            for oc in range(n_oc):
                f.write(" ".join(str(int(w_int[ic, oc, ky, kx]))
                                 for ic in range(n_ic)) + "\n")
    print(f"  [TConv2d ] {os.path.basename(filepath)}: IC={n_ic} OC={n_oc} K={kh}x{kw}")


# ==========================================
# Weight dump: LAYER_DUMP_MAP
# ==========================================
# Maps all 11 MVFB_LAYERS_FULL layers to block submodules.
# ic_slice / oc_slice: how many weight channels to take (rest zero-padded).
#
#   Phase 9  (blocks 0-2): num_feature=160
#   Phase 10 (blocks 3-5): num_feature=80
#   Phase 11 (blocks 6-8): num_feature=44
#
# Slice sizes are aligned to PE_NUM=4 / MACPE=4 granularity.
# Layers where actual channel count < slice (e.g. conv1 at nf=44) are zero-padded;
# zero-padded channels produce zero output and do not affect arithmetic correctness.

LAYER_DUMP_MAP = [
    # (li, name,           accessor,                    is_tconv, ic_sl, oc_sl)
    (0,  "conv0_0",      lambda b: b.conv0[0][0],      False,  8, 16),
    (1,  "conv0_1",      lambda b: b.conv0[1][0],      False, 16, 16),
    (2,  "convblock_0",  lambda b: b.convblock[0][0],  False, 16, 16),
    (3,  "convblock_1",  lambda b: b.convblock[1][0],  False, 16, 16),
    (4,  "convblock_2",  lambda b: b.convblock[2][0],  False, 16, 16),
    (5,  "conv_sq",      lambda b: b.conv_sq[0],       False, 16, 16),
    (6,  "conv1",        lambda b: b.conv1[0][0],      False,  8,  8),
    (7,  "convblock1_0", lambda b: b.convblock1[0][0], False,  8,  8),
    (8,  "convblock1_1", lambda b: b.convblock1[0][0], False,  8,  8),
    (9,  "convblock1_2", lambda b: b.convblock1[0][0], False,  8,  8),
    (10, "lastconv",     lambda b: b.lastconv,          True,  16,  4),
]

# Phase 12: same map but lastconv oc_sl=8 (5 real OC padded to next multiple of 4)
LAYER_DUMP_MAP_P12 = [
    # (li, name,           accessor,                    is_tconv, ic_sl, oc_sl)
    (0,  "conv0_0",      lambda b: b.conv0[0][0],      False,  8, 16),
    (1,  "conv0_1",      lambda b: b.conv0[1][0],      False, 16, 16),
    (2,  "convblock_0",  lambda b: b.convblock[0][0],  False, 16, 16),
    (3,  "convblock_1",  lambda b: b.convblock[1][0],  False, 16, 16),
    (4,  "convblock_2",  lambda b: b.convblock[2][0],  False, 16, 16),
    (5,  "conv_sq",      lambda b: b.conv_sq[0],       False, 16, 16),
    (6,  "conv1",        lambda b: b.conv1[0][0],      False,  8,  8),
    (7,  "convblock1_0", lambda b: b.convblock1[0][0], False,  8,  8),
    (8,  "convblock1_1", lambda b: b.convblock1[0][0], False,  8,  8),
    (9,  "convblock1_2", lambda b: b.convblock1[0][0], False,  8,  8),
    (10, "lastconv",     lambda b: b.lastconv,          True,  16,  8),  # oc_sl=8: covers 5ch OC
]


def dump_block_weights_p12(model, block_idx):
    """
    Dump all 11 layer weights for block{block_idx} to:
      ./data/phase12/block{block_idx}/p12_b{block_idx}_L{li}_weight.txt

    Identical to dump_block_weights except lastconv uses oc_sl=8 (covers 5 OC channels).
    Used by gen_mvfb_phase12_stim.py.
    """
    phase = 12
    out_dir = os.path.join(f"./data/phase{phase}", f"block{block_idx}")
    os.makedirs(out_dir, exist_ok=True)
    block = getattr(model.dmvfn, f"block{block_idx}")

    print(f"\n{'='*60}")
    print(f" Weights P12: block{block_idx}  →  {out_dir}")
    print(f"{'='*60}")

    for (li, _name, accessor, is_tconv, ic_sl, oc_sl) in LAYER_DUMP_MAP_P12:
        module = accessor(block)
        w_np = module.weight.detach().cpu().numpy().astype(np.float32)
        fpath = os.path.join(out_dir, f"p{phase}_b{block_idx}_L{li}_weight.txt")
        if is_tconv:
            w_int = np.vectorize(_q)(_pad_weight_tconv2d(w_np, target_ic=ic_sl, target_oc=oc_sl))
            write_weight_tconv2d_txt(fpath, w_int)
        else:
            w_int = np.vectorize(_q)(_pad_weight_conv2d(w_np, target_oc=oc_sl, target_ic=ic_sl))
            write_weight_conv2d_txt(fpath, w_int)


def dump_block_weights(model, block_idx, phase):
    """
    Dump all 11 layer weights for block{block_idx} to:
      ./data/phase{phase}/block{block_idx}/p{phase}_b{block_idx}_L{li}_weight.txt

    Used by gen_mvfb_phase{9,10,11}_stim.py to load real DMVFN weights.
    """
    out_dir = os.path.join(f"./data/phase{phase}", f"block{block_idx}")
    os.makedirs(out_dir, exist_ok=True)
    block = getattr(model.dmvfn, f"block{block_idx}")

    print(f"\n{'='*60}")
    print(f" Weights: block{block_idx} (phase{phase})  →  {out_dir}")
    print(f"{'='*60}")

    for (li, _name, accessor, is_tconv, ic_sl, oc_sl) in LAYER_DUMP_MAP:
        module = accessor(block)
        w_np = module.weight.detach().cpu().numpy().astype(np.float32)
        fpath = os.path.join(out_dir, f"p{phase}_b{block_idx}_L{li}_weight.txt")
        if is_tconv:
            w_int = np.vectorize(_q)(_pad_weight_tconv2d(w_np, target_ic=ic_sl, target_oc=oc_sl))
            write_weight_tconv2d_txt(fpath, w_int)
        else:
            w_int = np.vectorize(_q)(_pad_weight_conv2d(w_np, target_oc=oc_sl, target_ic=ic_sl))
            write_weight_conv2d_txt(fpath, w_int)


# ==========================================
# Block input hooks: 17ch feature map capture
# ==========================================
# Each MVFB block receives a 17-channel tensor at its first conv layer.
# The hook captures this after F.interpolate(scale=1/scale) inside MVFB.forward():
#
#   ch 0-2  : img0 RGB
#   ch 3-5  : img1 RGB
#   ch 6-8  : warped_img0 RGB
#   ch 9-11 : warped_img1 RGB
#   ch 12   : mask
#   ch 13-16: accumulated optical flow (4ch)
#
# Spatial resolution per phase:
#   blocks 0-2 (scale=4): 64×112
#   blocks 3-5 (scale=2): 128×224
#   blocks 6-8 (scale=1): 256×448
#
# Output: ./data/block{i}_input_17ch.txt
#   one line per pixel, 17 Q8.8 unsigned hex values separated by spaces

_block_input_dump_count = {}
_block_flow_deltas = {}   # block_idx → tensor (4, H, W) float  (flow_d)
_block_mask_deltas = {}   # block_idx → tensor (1, H, W) float  (mask_d)


def get_block_input_hook(block_idx):
    """Return a forward hook that dumps block{block_idx} 17ch input on first call."""
    _block_input_dump_count.setdefault(block_idx, 0)

    def hook(_module, input, _output):
        if _block_input_dump_count[block_idx] == 0:
            fm = input[0].detach().cpu()  # (1, 17, H, W)
            _, C, H, W = fm.shape
            print(f"\n{'='*60}")
            print(f" Block{block_idx} input: C={C} H={H} W={W}")
            out_path = f"./data/block{block_idx}_input_17ch.txt"
            with open(out_path, "w") as f:
                data = np.round(fm[0].numpy() * 256.0).astype(np.int32)
                for y in range(H):
                    for x in range(W):
                        vals = [int(data[c, y, x]) & 0xFFFF for c in range(C)]
                        f.write(" ".join(f"{v:04x}" for v in vals) + "\n")
            print(f"  [Dump] block{block_idx}_input_17ch.txt | {H*W} pixels | {C}ch")
            print(f"{'='*60}")
            _block_input_dump_count[block_idx] += 1

    return hook


def get_block_output_hook(block_idx):
    """
    Return a forward hook on MVFB block{block_idx} that stores (flow_d, mask_d).
    MVFB.forward() returns (flow_d, mask_d) both at full resolution (H, W).
    Accumulated flow = sum of all block flow_d tensors (since _ForceOne → ref=1).
    """
    def hook(_module, _input, output):
        flow_d, mask_d = output          # each: (1, 4/1, H, W)
        _block_flow_deltas[block_idx] = flow_d.detach().cpu()[0]   # (4, H, W)
        _block_mask_deltas[block_idx] = mask_d.detach().cpu()[0]   # (1, H, W)
    return hook


# ==========================================
# Phase 15: Block0 lastconv input capture
# ==========================================
_p15_lastconv_fm = None   # (1, IC, H, W) float — block0.lastconv input


def get_p15_lastconv_hook():
    """Hook on block0.lastconv: capture the lastconv input feature map (fires once)."""
    def hook(_module, input, _output):
        global _p15_lastconv_fm
        if _p15_lastconv_fm is None:
            _p15_lastconv_fm = input[0].detach().cpu()   # (1, IC, H, W)
            _, IC, H, W = _p15_lastconv_fm.shape
            print(f"  [Phase15] block0.lastconv input captured: IC={IC} H={H} W={W}")
    return hook


def dump_phase15_data(img0_bgr, img1_bgr, model):
    """
    Dump stimulus for Phase 15: full-res Block0 lastconv.

    Writes to ./data/phase15/:
      p15_b0_L10_input.txt          — lastconv input FM (Q8.8 signed decimal,
                                       IC*H lines x W values, channel-major)
      block0/p15_b0_L10_weight.txt  — lastconv weights OC0-7 (Q8.8, oc_sl=8)
      p15_img0_bank{N}.txt  N=0..3  — img0 SRAM banks (even/odd x,y interleave)
      p15_img1_bank{N}.txt  N=0..3  — img1 SRAM banks
    """
    out_dir = "./data/phase15"
    os.makedirs(out_dir, exist_ok=True)

    if _p15_lastconv_fm is None:
        print("[Phase 15] ERROR: block0.lastconv hook did not fire.")
        return

    fm = _p15_lastconv_fm          # (1, IC, H, W) float
    _, IC, H, W = fm.shape
    print(f"\n{'='*60}")
    print(f" Phase 15 data dump: Block0 lastconv input IC={IC} H={H} W={W}")
    print(f"{'='*60}")

    # ── Lastconv input FM → Q8.8 signed decimal ──────────────────────────────
    # Format: IC*H lines, each line has W space-separated signed decimal integers.
    # Matches p9_b0_L10_input.txt layout (channel-major, row-major within channel).
    fm_q = np.round(fm[0].numpy() * 256.0).astype(np.int32)   # (IC, H, W)
    fm_q = np.clip(fm_q, -32768, 32767)
    fpath = os.path.join(out_dir, "p15_b0_L10_input.txt")
    with open(fpath, 'w') as f:
        for c in range(IC):
            for y in range(H):
                f.write(" ".join(str(int(fm_q[c, y, x])) for x in range(W)) + "\n")
    print(f"  [Dump] p15_b0_L10_input.txt | IC={IC} H={H} W={W} | {IC*H} lines x {W} cols")

    # ── Lastconv weights: ic_sl=actual (rounded up to MACPE=4), oc_sl=8 ──
    wt_dir = os.path.join(out_dir, "block0")
    os.makedirs(wt_dir, exist_ok=True)
    w_np = model.dmvfn.block0.lastconv.weight.detach().cpu().numpy().astype(np.float32)
    actual_ic = w_np.shape[0]   # ConvTranspose2d: (in_ch, out_ch, kH, kW)
    ic_sl = ((actual_ic + 3) // 4) * 4  # round up to MACPE=4 multiple
    w_int = np.vectorize(_q)(_pad_weight_tconv2d(w_np, target_ic=ic_sl, target_oc=8))
    write_weight_tconv2d_txt(os.path.join(wt_dir, "p15_b0_L10_weight.txt"), w_int)
    print(f"  [Dump] p15_b0_L10_weight.txt (ic_sl={ic_sl})")

    # ── Source images → 4-bank SRAM format ───────────────────────────────────
    # bank_id = (y%2)*2 + (x%2), matches warp address generator.
    # img0 → p15_img0_bank{0-3}, img1 → p15_img1_bank{0-3}.
    img0_rgb = img0_bgr[:, :, ::-1].copy()
    img1_rgb = img1_bgr[:, :, ::-1].copy()
    dump_img_banks(img0_rgb, "p15_img0", out_dir)
    dump_img_banks(img1_rgb, "p15_img1", out_dir)

    print(f"\n[Phase 15] All dumps written to {out_dir}")


# ==========================================
# Phase 16: Block1 + Block2 lastconv capture
# ==========================================
_p16_lastconv_fm = {1: None, 2: None}   # block_idx → (1, IC, H, W) float


def get_p16_lastconv_hook(block_idx):
    """Hook on block{block_idx}.lastconv: capture IC input FM (fires once per block)."""
    def hook(_module, input, _output):
        if _p16_lastconv_fm[block_idx] is None:
            _p16_lastconv_fm[block_idx] = input[0].detach().cpu()
            _, IC, H, W = _p16_lastconv_fm[block_idx].shape
            print(f"  [Phase16] block{block_idx}.lastconv input captured: IC={IC} H={H} W={W}")
    return hook


def dump_phase16_data(model):
    """
    Dump stimulus for Phase 16: Block1 + Block2 lastconv FM + weights.
    Source images are reused from Phase 15 (same run).

    Writes to ./data/phase16/:
      p16_b{1,2}_L10_input.txt          — lastconv input FM (Q8.8, IC*H lines x W values)
      block{1,2}/p16_b{1,2}_L10_weight.txt — lastconv weights (OC0-7, oc_sl=8)
    """
    out_dir = "./data/phase16"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Phase 16 data dump: Block1 + Block2 lastconv")
    print(f"{'='*60}")

    for bidx in [1, 2]:
        fm_tensor = _p16_lastconv_fm[bidx]
        if fm_tensor is None:
            print(f"  [Phase16] ERROR: block{bidx}.lastconv hook did not fire.")
            continue

        fm = fm_tensor          # (1, IC, H, W)
        _, IC, H, W = fm.shape

        # FM → Q8.8 signed decimal (IC*H lines x W values, channel-major)
        fm_q = np.round(fm[0].numpy() * 256.0).astype(np.int32)
        fm_q = np.clip(fm_q, -32768, 32767)
        fpath = os.path.join(out_dir, f"p16_b{bidx}_L10_input.txt")
        with open(fpath, 'w') as f:
            for c in range(IC):
                for y in range(H):
                    f.write(" ".join(str(int(fm_q[c, y, x])) for x in range(W)) + "\n")
        print(f"  [Dump] p16_b{bidx}_L10_input.txt | IC={IC} H={H} W={W} | {IC*H} lines x {W} cols")

        # Weights: ic_sl = actual IC (rounded up to MACPE=4), oc_sl=8
        wt_dir = os.path.join(out_dir, f"block{bidx}")
        os.makedirs(wt_dir, exist_ok=True)
        block = getattr(model.dmvfn, f"block{bidx}")
        w_np = block.lastconv.weight.detach().cpu().numpy().astype(np.float32)
        actual_ic = w_np.shape[0]   # ConvTranspose2d: (in_ch, out_ch, kH, kW)
        ic_sl = ((actual_ic + 3) // 4) * 4  # round up to MACPE=4 multiple
        w_int = np.vectorize(_q)(_pad_weight_tconv2d(w_np, target_ic=ic_sl, target_oc=8))
        write_weight_tconv2d_txt(os.path.join(wt_dir, f"p16_b{bidx}_L10_weight.txt"), w_int)
        print(f"  [Dump] p16_b{bidx}_L10_weight.txt (ic_sl={ic_sl})")

    print(f"\n[Phase 16] All dumps written to {out_dir}")
    print("  (Source images: reuse data/phase15/p15_img{{0,1}}_bank{{0-3}}.txt)")


# ==========================================
# Phase 17: Block3 + Block4 + Block5 lastconv capture
# ==========================================
_p17_lastconv_fm = {3: None, 4: None, 5: None}   # block_idx → (1, IC, H, W) float


def get_p17_lastconv_hook(block_idx):
    """Hook on block{block_idx}.lastconv: capture IC input FM (fires once per block)."""
    def hook(_module, input, _output):
        if _p17_lastconv_fm[block_idx] is None:
            _p17_lastconv_fm[block_idx] = input[0].detach().cpu()
            _, IC, H, W = _p17_lastconv_fm[block_idx].shape
            print(f"  [Phase17] block{block_idx}.lastconv input captured: IC={IC} H={H} W={W}")
    return hook


def dump_phase17_data(model):
    """
    Dump stimulus for Phase 17: Block3 + Block4 + Block5 lastconv FM + weights.
    Blocks 3-5 are at scale=2 (1/2 resolution); FM spatial dims differ from blocks 0-2.
    Source images are reused from Phase 15 (same run).

    Writes to ./data/phase17/:
      p17_b{3,4,5}_L10_input.txt          — lastconv input FM (Q8.8, IC*H lines x W values)
      block{3,4,5}/p17_b{3,4,5}_L10_weight.txt — lastconv weights (OC0-7, ic_sl=16 oc_sl=8)
    """
    out_dir = "./data/phase17"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Phase 17 data dump: Block3 + Block4 + Block5 lastconv")
    print(f"{'='*60}")

    for bidx in [3, 4, 5]:
        fm_tensor = _p17_lastconv_fm[bidx]
        if fm_tensor is None:
            print(f"  [Phase17] ERROR: block{bidx}.lastconv hook did not fire.")
            continue

        fm = fm_tensor          # (1, IC, H, W)
        _, IC, H, W = fm.shape

        # FM → Q8.8 signed decimal (IC*H lines x W values, channel-major)
        fm_q = np.round(fm[0].numpy() * 256.0).astype(np.int32)
        fm_q = np.clip(fm_q, -32768, 32767)
        fpath = os.path.join(out_dir, f"p17_b{bidx}_L10_input.txt")
        with open(fpath, 'w') as f:
            for c in range(IC):
                for y in range(H):
                    f.write(" ".join(str(int(fm_q[c, y, x])) for x in range(W)) + "\n")
        print(f"  [Dump] p17_b{bidx}_L10_input.txt | IC={IC} H={H} W={W} | {IC*H} lines x {W} cols")

        # Weights: ic_sl = actual IC (28 for blocks 3-5, aligned to MACPE=4), oc_sl=8
        wt_dir = os.path.join(out_dir, f"block{bidx}")
        os.makedirs(wt_dir, exist_ok=True)
        block = getattr(model.dmvfn, f"block{bidx}")
        w_np = block.lastconv.weight.detach().cpu().numpy().astype(np.float32)
        actual_ic = w_np.shape[0]   # ConvTranspose2d: (in_ch, out_ch, kH, kW)
        ic_sl = ((actual_ic + 3) // 4) * 4   # round up to MACPE=4 multiple
        w_int = np.vectorize(_q)(_pad_weight_tconv2d(w_np, target_ic=ic_sl, target_oc=8))
        write_weight_tconv2d_txt(os.path.join(wt_dir, f"p17_b{bidx}_L10_weight.txt"), w_int)
        print(f"  [Dump] p17_b{bidx}_L10_weight.txt (ic_sl={ic_sl})")

    print(f"\n[Phase 17] All dumps written to {out_dir}")
    print("  (Source images: reuse data/phase15/p15_img{{0,1}}_bank{{0-3}}.txt)")


# ==========================================
# Phase 18: Block6 + Block7 + Block8 lastconv capture
# ==========================================
_p18_lastconv_fm = {6: None, 7: None, 8: None}   # block_idx → (1, IC, H, W) float


def get_p18_lastconv_hook(block_idx):
    """Hook on block{block_idx}.lastconv: capture IC input FM (fires once per block)."""
    def hook(_module, input, _output):
        if _p18_lastconv_fm[block_idx] is None:
            _p18_lastconv_fm[block_idx] = input[0].detach().cpu()
            _, IC, H, W = _p18_lastconv_fm[block_idx].shape
            print(f"  [Phase18] block{block_idx}.lastconv input captured: IC={IC} H={H} W={W}")
    return hook


def dump_phase18_data(model):
    """
    Dump stimulus for Phase 18: Block6 + Block7 + Block8 lastconv FM + weights.
    Blocks 6-8 are at scale=1 (full resolution); FM spatial dims may differ from blocks 3-5.
    Source images are reused from Phase 15 (same run).

    Writes to ./data/phase18/:
      p18_b{6,7,8}_L10_input.txt          — lastconv input FM (Q8.8, IC*H lines x W values)
      block{6,7,8}/p18_b{6,7,8}_L10_weight.txt — lastconv weights (ic_sl=actual, oc_sl=8)
    """
    out_dir = "./data/phase18"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Phase 18 data dump: Block6 + Block7 + Block8 lastconv")
    print(f"{'='*60}")

    for bidx in [6, 7, 8]:
        fm_tensor = _p18_lastconv_fm[bidx]
        if fm_tensor is None:
            print(f"  [Phase18] ERROR: block{bidx}.lastconv hook did not fire.")
            continue

        fm = fm_tensor          # (1, IC, H, W)
        _, IC, H, W = fm.shape

        # FM → Q8.8 signed decimal (IC*H lines x W values, channel-major)
        fm_q = np.round(fm[0].numpy() * 256.0).astype(np.int32)
        fm_q = np.clip(fm_q, -32768, 32767)
        fpath = os.path.join(out_dir, f"p18_b{bidx}_L10_input.txt")
        with open(fpath, 'w') as f:
            for c in range(IC):
                for y in range(H):
                    f.write(" ".join(str(int(fm_q[c, y, x])) for x in range(W)) + "\n")
        print(f"  [Dump] p18_b{bidx}_L10_input.txt | IC={IC} H={H} W={W} | {IC*H} lines x {W} cols")

        # Weights: ic_sl = actual IC rounded up to MACPE=4, oc_sl=8
        wt_dir = os.path.join(out_dir, f"block{bidx}")
        os.makedirs(wt_dir, exist_ok=True)
        block = getattr(model.dmvfn, f"block{bidx}")
        w_np = block.lastconv.weight.detach().cpu().numpy().astype(np.float32)
        actual_ic = w_np.shape[0]   # ConvTranspose2d: (in_ch, out_ch, kH, kW)
        ic_sl = ((actual_ic + 3) // 4) * 4
        w_int = np.vectorize(_q)(_pad_weight_tconv2d(w_np, target_ic=ic_sl, target_oc=8))
        write_weight_tconv2d_txt(os.path.join(wt_dir, f"p18_b{bidx}_L10_weight.txt"), w_int)
        print(f"  [Dump] p18_b{bidx}_L10_weight.txt (ic_sl={ic_sl})")

    print(f"\n[Phase 18] All dumps written to {out_dir}")
    print("  (Source images: reuse data/phase15/p15_img{{0,1}}_bank{{0-3}}.txt)")


# ==========================================
# Phase 12: source image → 4-bank format dump
# ==========================================
def dump_img_banks(img_hwc_uint8, bank_prefix, out_dir):
    """
    Dump a (H, W, 3) uint8 RGB image to 4-bank SRAM format used by the warp HW.
    Bank assignment matches the warp address generator:
      bank0: (even_x, even_y) — TL neighbour
      bank1: (odd_x,  even_y) — TR
      bank2: (even_x, odd_y)  — BL
      bank3: (odd_x,  odd_y)  — BR
    Each line: "RR GG BB" (2-digit hex), space-separated.
    Files: {out_dir}/{bank_prefix}_bank{N}.txt  N=0..3
    """
    H, W, _ = img_hwc_uint8.shape
    banks = [[] for _ in range(4)]
    for y in range(H):
        for x in range(W):
            b_idx = (y % 2) * 2 + (x % 2)
            r = int(img_hwc_uint8[y, x, 0])
            g = int(img_hwc_uint8[y, x, 1])
            b = int(img_hwc_uint8[y, x, 2])
            banks[b_idx].append(f"{r:02x} {g:02x} {b:02x}")
    for n in range(4):
        fpath = os.path.join(out_dir, f"{bank_prefix}_bank{n}.txt")
        with open(fpath, 'w') as f:
            f.write("\n".join(banks[n]) + "\n")
        print(f"  [Dump] {os.path.basename(fpath)}: {len(banks[n])} entries")


def _warp_bilinear_cpu(img_t, flow_t):
    """
    Bilinear backward warp on CPU tensors.
    img_t:  (1, C, H, W) float [0,1]
    flow_t: (1, 2, H, W) float — pixel displacement (dx, dy)
    Returns (1, C, H, W) float, zeros outside boundary.
    Uses the same normalization as arch.py warp().
    """
    _, C, H, W = img_t.shape
    # Normalize flow to [-1,1] displacement (as arch.py does)
    flow_norm = torch.stack([
        flow_t[0, 0] / ((W - 1.0) / 2.0),
        flow_t[0, 1] / ((H - 1.0) / 2.0),
    ], dim=0).unsqueeze(0)   # (1, 2, H, W)
    # Build base grid [-1,1]
    xs = torch.linspace(-1.0, 1.0, W).view(1, 1, W).expand(1, H, W)
    ys = torch.linspace(-1.0, 1.0, H).view(1, H, 1).expand(1, H, W)
    base = torch.cat([xs, ys], dim=0).unsqueeze(0)   # (1, 2, H, W)
    grid = (base + flow_norm).permute(0, 2, 3, 1)    # (1, H, W, 2)
    return F.grid_sample(img_t, grid, mode='bilinear',
                         padding_mode='border', align_corners=True)


def dump_phase12_data(img0_bgr, img1_bgr):
    """
    Called after inference (all 9 block output hooks have fired).
    Computes accumulated flow and sigmoid(mask) then writes:

      data/phase12/p12_img0_bank{N}.txt  (N=0..3)  — img0 for bank0-3
      data/phase12/p12_img1_bank{N}.txt  (N=0..3)  — img1 for bank4-7
      data/phase12/p12_flow_mask.txt     — per pixel: fx0 fy0 fx1 fy1 sm  (Q8.8 hex)
      data/phase12/p12_golden_blend_{R,G,B}.txt     — golden blend (1 hex byte/line)
      data/phase12/p12_warp_config.txt   — "total_pixels tile_w tile_h"
    """
    out_dir = "./data/phase12"
    os.makedirs(out_dir, exist_ok=True)

    if len(_block_flow_deltas) != 9:
        print(f"[WARNING] Only {len(_block_flow_deltas)}/9 block output hooks fired.")

    # ── Accumulate flow and mask (sum of all block deltas, ref=1 for all) ──
    sample = _block_flow_deltas[sorted(_block_flow_deltas)[0]]
    _, H, W = sample.shape
    acc_flow = torch.zeros(4, H, W)
    acc_mask = torch.zeros(1, H, W)
    for i in range(9):
        if i in _block_flow_deltas:
            acc_flow += _block_flow_deltas[i]
        if i in _block_mask_deltas:
            acc_mask += _block_mask_deltas[i]
    sig_mask = torch.sigmoid(acc_mask)   # (1, H, W) in (0, 1)

    print(f"\n{'='*60}")
    print(f" Phase 12 data dump: H={H} W={W}")
    print(f"  acc_flow : min={acc_flow.min():.3f}  max={acc_flow.max():.3f}")
    print(f"  sig_mask : min={sig_mask.min():.3f}  max={sig_mask.max():.3f}")
    print(f"{'='*60}")

    # ── Source images (BGR→RGB, uint8) ─────────────────────────────────────
    img0_rgb = img0_bgr[:, :, ::-1].copy()
    img1_rgb = img1_bgr[:, :, ::-1].copy()
    dump_img_banks(img0_rgb, "p12_img0", out_dir)
    dump_img_banks(img1_rgb, "p12_img1", out_dir)

    # ── Flow + sigmoid(mask) → Q8.8 hex ────────────────────────────────────
    # Flow: signed Q8.8 — mask off to 16 bits to preserve sign via 2's complement
    # Mask: unsigned Q8.8 — clamped [0, 256]
    flow_q = np.round(acc_flow.numpy() * 256.0).astype(np.int32)   # (4, H, W)
    mask_q = np.round(sig_mask.numpy() * 256.0).clip(0, 256).astype(np.int32)  # (1, H, W)

    fpath = os.path.join(out_dir, "p12_flow_mask.txt")
    with open(fpath, 'w') as fp:
        for y in range(H):
            for x in range(W):
                fx0 = int(flow_q[0, y, x]) & 0xFFFF
                fy0 = int(flow_q[1, y, x]) & 0xFFFF
                fx1 = int(flow_q[2, y, x]) & 0xFFFF
                fy1 = int(flow_q[3, y, x]) & 0xFFFF
                sm  = int(mask_q[0, y, x]) & 0xFFFF
                fp.write(f"{fx0:04x} {fy0:04x} {fx1:04x} {fy1:04x} {sm:04x}\n")
    print(f"  [Dump] p12_flow_mask.txt | {H*W} pixels")

    # ── Golden blend: warp img0 with flow[:2], img1 with flow[2:4] ─────────
    # Use quantized Q8.8 flow/mask (divided back to float) so golden matches HW exactly
    img0_t = torch.from_numpy(img0_rgb.transpose(2, 0, 1).astype(np.float32) / 255.).unsqueeze(0)
    img1_t = torch.from_numpy(img1_rgb.transpose(2, 0, 1).astype(np.float32) / 255.).unsqueeze(0)
    flow_q_float = torch.from_numpy(flow_q.astype(np.float32) / 256.0)
    flow0  = flow_q_float[:2].unsqueeze(0)   # (1, 2, H, W) — warp img0
    flow1  = flow_q_float[2:].unsqueeze(0)   # (1, 2, H, W) — warp img1
    mask_q_float = torch.from_numpy(mask_q.astype(np.float32) / 256.0)
    warped0 = _warp_bilinear_cpu(img0_t, flow0)
    warped1 = _warp_bilinear_cpu(img1_t, flow1)
    m = mask_q_float.unsqueeze(0)            # (1, 1, H, W)
    blended = (warped0 * m + warped1 * (1.0 - m)).clamp(0, 1)
    blend_np = np.floor(blended[0].numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)  # (3, H, W) RGB

    for ch_name, ch_idx in [("R", 0), ("G", 1), ("B", 2)]:
        fpath = os.path.join(out_dir, f"p12_golden_blend_{ch_name}.txt")
        with open(fpath, 'w') as fp:
            for v in blend_np[ch_idx].flatten():
                fp.write(f"{int(v):02x}\n")
        print(f"  [Dump] p12_golden_blend_{ch_name}.txt | {H*W} pixels")

    # ── Warp config ─────────────────────────────────────────────────────────
    fpath = os.path.join(out_dir, "p12_warp_config.txt")
    with open(fpath, 'w') as fp:
        fp.write(f"{H*W} {W} {H}\n")
    print(f"  [Dump] p12_warp_config.txt | total={H*W} W={W} H={H}")
    print(f"\n[Phase 12] All dumps written to {out_dir}")


# ==========================================
# Setup
# ==========================================
device = torch.device("cuda")
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--image_0_path', required=True,  type=str, help='path to frame 0')
parser.add_argument('--image_1_path', required=True,  type=str, help='path to frame 1')
parser.add_argument('--load_path',    required=True,  type=str, help='model checkpoint')
parser.add_argument('--output_dir',   default="pred.png", type=str, help='output image path')
args = parser.parse_args()


# ==========================================
# Inference + hook registration
# ==========================================
def evaluate(model, args):
    with torch.no_grad():
        img_0 = cv2.imread(args.image_0_path)
        img_1 = cv2.imread(args.image_1_path)
        if img_0 is None or img_1 is None:
            raise Exception("Images not found.")

        # Keep original uint8 BGR for Phase 12 source-image dump
        img0_bgr = img_0.copy()
        img1_bgr = img_1.copy()

        img_0 = img_0.transpose(2, 0, 1).astype('float32')
        img_1 = img_1.transpose(2, 0, 1).astype('float32')

        img = torch.cat([torch.tensor(img_0), torch.tensor(img_1)], dim=0)
        img = img.unsqueeze(0).unsqueeze(0).to(device, non_blocking=True) / 255.

        # Register 17ch input hooks for all 9 blocks
        # Hooks fire on first forward pass and write block{i}_input_17ch.txt
        for i in range(9):
            getattr(model.dmvfn, f"block{i}").conv0[0][0].register_forward_hook(
                get_block_input_hook(i))

        # Register output hooks for Phase 12: capture (flow_d, mask_d) per block
        for i in range(9):
            getattr(model.dmvfn, f"block{i}").register_forward_hook(
                get_block_output_hook(i))

        # Register Phase 15: capture block0.lastconv input FM
        model.dmvfn.block0.lastconv.register_forward_hook(get_p15_lastconv_hook())

        # Register Phase 16: capture block1 + block2 lastconv input FM
        model.dmvfn.block1.lastconv.register_forward_hook(get_p16_lastconv_hook(1))
        model.dmvfn.block2.lastconv.register_forward_hook(get_p16_lastconv_hook(2))

        # Register Phase 17: capture block3 + block4 + block5 lastconv input FM
        model.dmvfn.block3.lastconv.register_forward_hook(get_p17_lastconv_hook(3))
        model.dmvfn.block4.lastconv.register_forward_hook(get_p17_lastconv_hook(4))
        model.dmvfn.block5.lastconv.register_forward_hook(get_p17_lastconv_hook(5))

        # Register Phase 18: capture block6 + block7 + block8 lastconv input FM
        model.dmvfn.block6.lastconv.register_forward_hook(get_p18_lastconv_hook(6))
        model.dmvfn.block7.lastconv.register_forward_hook(get_p18_lastconv_hook(7))
        model.dmvfn.block8.lastconv.register_forward_hook(get_p18_lastconv_hook(8))

        pred = model.eval(img, 'single_test')
        pred = np.array(pred.cpu().squeeze() * 255).transpose(1, 2, 0)
        cv2.imwrite(args.output_dir, pred)

        # Phase 12: compute accumulated flow/mask and write all stimulus files
        dump_phase12_data(img0_bgr, img1_bgr)

        # Phase 15: full-res Block0 lastconv FM + weights + source images
        dump_phase15_data(img0_bgr, img1_bgr, model)

        # Phase 16: Block1 + Block2 lastconv FM + weights
        dump_phase16_data(model)

        # Phase 17: Block3 + Block4 + Block5 lastconv FM + weights
        dump_phase17_data(model)

        # Phase 18: Block6 + Block7 + Block8 lastconv FM + weights
        dump_phase18_data(model)


# ==========================================
# Main: dump weights then run inference
# ==========================================
if __name__ == "__main__":
    model = Model(load_path=args.load_path, training=False)

    # Phase 9:  blocks 0-2  (scale=4, num_feature=160)
    for b in range(0, 3):
        dump_block_weights(model, b, phase=9)

    # Phase 10: blocks 3-5  (scale=2, num_feature=80)
    for b in range(3, 6):
        dump_block_weights(model, b, phase=10)

    # Phase 11: blocks 6-8  (scale=1, num_feature=44)
    for b in range(6, 9):
        dump_block_weights(model, b, phase=11)

    # Phase 12: blocks 6-8 weights with lastconv oc_sl=8 (covers 5-ch output)
    for b in range(6, 9):
        dump_block_weights_p12(model, b)

    # Run inference:
    #   - triggers block input hooks  → block{i}_input_17ch.txt
    #   - triggers block output hooks → _block_flow_deltas / _block_mask_deltas
    #   - calls dump_phase12_data()   → data/phase12/ stimulus files
    evaluate(model, args)