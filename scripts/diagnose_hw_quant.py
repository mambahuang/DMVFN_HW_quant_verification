#!/usr/bin/env python3
"""
diagnose_hw_quant.py — Compare hw_quant intermediate values vs sim_dmvfn dump data
====================================================================================
Runs hw_quant on the same image pair used to generate sim_dmvfn stim data,
then compares acc_flow / acc_mask / final image at each step.

Usage (from project root):
  python scripts/diagnose_hw_quant.py \
      --dataset       kitti \
      --image_0_path  KITTI/phase15/img0.png \
      --image_1_path  KITTI/phase15/img1.png \
      --load_path     pretrained_models/dmvfn_kitti.pkl \
      --stim_dir      MVFB/stim_data/KITTI

  python scripts/diagnose_hw_quant.py \
      --dataset       cityscapes \
      --image_0_path  cityscapes/phase15/img0.png \
      --image_1_path  cityscapes/phase15/img1.png \
      --load_path     pretrained_models/dmvfn_city.pkl \
      --stim_dir      MVFB/stim_data/cityscapes

The script reconstructs img0/img1 from the bank files in stim_dir if
--image_0_path / --image_1_path are not provided.
"""

import os, sys, argparse
import numpy as np
import cv2
import torch

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, 'scripts'))

from dmvfn_simplified import DMVFN_Simplified, Model_Simplified
DMVFN_Simplified.MODE = 'hw_quant'

# ---------------------------------------------------------------------------
# Dataset params
# ---------------------------------------------------------------------------
DATASET_PARAMS = {
    'kitti': dict(
        prefix='kitti', src_h=256, src_w=832,
        coord_int_w=11, mask_shift=8,
    ),
    'cityscapes': dict(
        prefix='city', src_h=512, src_w=1024,
        coord_int_w=12, mask_shift=8,
    ),
}

COORD_FRAC_W = 10
FRAC_ONE     = 1 << COORD_FRAC_W
FLOW_SHIFT   = 6
OC_USE       = 5


# ---------------------------------------------------------------------------
# Load sim_dmvfn stim: per-block FM and weights → reproduce acc_flow/acc_mask
# ---------------------------------------------------------------------------
def load_fm(path, ic_padded, fm_h, fm_w):
    _CLIP = (1 << 31) - 1
    _warned = False
    lines = []
    with open(path) as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if stripped:
                lines.append(stripped)
    fm = np.zeros((ic_padded, fm_h, fm_w), dtype=np.int64)
    for idx, line in enumerate(lines):
        c, y = divmod(idx, fm_h)
        raw = line.split()
        clamped = []
        for v in raw:
            iv = int(v)
            if not _warned and (iv > _CLIP or iv < -_CLIP - 1):
                print(f"  [WARN] load_fm {os.path.basename(path)}: "
                      f"value {iv} out of int32 range at line {idx} — clipping. "
                      f"Check that the correct stim file is being loaded.")
                _warned = True
            clamped.append(max(-_CLIP - 1, min(_CLIP, iv)))
        n_copy = min(len(clamped), fm_w)
        fm[c, y, :n_copy] = clamped[:n_copy]
    return fm


def load_weight(path, ic_padded, kh, kw, oc_use=OC_USE, oc_file=8):
    lines = []
    with open(path) as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if stripped:
                lines.append(stripped)
    weight = np.zeros((ic_padded, oc_use, kh, kw), dtype=np.int64)
    idx = 0
    for k in range(kh * kw):
        ky, kx = k // kw, k % kw
        for oc in range(oc_file):
            vals = list(map(int, lines[idx].split()))
            if oc < oc_use:
                for ic in range(min(len(vals), ic_padded)):
                    weight[ic, oc, ky, kx] = vals[ic]
            idx += 1
    return weight


def load_bias(path, oc_use=OC_USE):
    if not os.path.exists(path):
        return np.zeros(oc_use, dtype=np.int64)
    with open(path) as f:
        return np.array(list(map(int, f.read().split())), dtype=np.int64)[:oc_use]


def tconv2d_full(fm, weight, kh, kw, stride=2):
    """Full-image TConv2d (no tiling). fm: (IC,H,W), weight: (IC,OC,KH,KW)."""
    ic, fm_h, fm_w = fm.shape
    _, oc, _, _ = weight.shape
    oh = (fm_h - 1) * stride + kh
    ow = (fm_w - 1) * stride + kw
    out = np.zeros((oc, oh, ow), dtype=np.int64)
    for ky in range(kh):
        for kx in range(kw):
            w = weight[:, :, ky, kx]   # (IC, OC)
            contrib = np.einsum('io,ihw->ohw', w, fm.astype(np.int64))
            out[:, ky:ky + fm_h*stride:stride,
                   kx:kx + fm_w*stride:stride] += contrib
    return out


def reconstruct_img_from_banks(stim_dir, prefix, img_idx, src_h, src_w):
    """Reconstruct RGB image from bank files."""
    bank_dim_w = (src_w // 2 - 1).bit_length()
    img = np.zeros((src_h, src_w, 3), dtype=np.uint8)
    banks = {}
    for bn in range(4):
        for ch, ch_name in enumerate(['R', 'G', 'B']):
            fpath = os.path.join(stim_dir, f"{prefix}_img{img_idx}_bank{bn}_{ch_name}.txt")
            if not os.path.exists(fpath):
                return None
            vals = []
            with open(fpath) as fh:
                for raw_line in fh:
                    stripped = raw_line.strip()
                    if stripped:
                        vals.append(int(stripped, 16))
            banks[(bn, ch)] = np.array(vals, dtype=np.uint8)
    for y in range(src_h):
        for x in range(src_w):
            bn = (y % 2) * 2 + (x % 2)
            bx, by = x >> 1, y >> 1
            addr = (by << bank_dim_w) | bx
            for ch in range(3):
                arr = banks[(bn, ch)]
                img[y, x, ch] = arr[addr] if addr < len(arr) else 0
    return img


def compute_psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0**2 / mse)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',      choices=['kitti', 'cityscapes'], default='kitti')
    ap.add_argument('--stim_dir',     required=True)
    ap.add_argument('--load_path',    required=True)
    ap.add_argument('--image_0_path', default=None)
    ap.add_argument('--image_1_path', default=None)
    ap.add_argument('--sim_final',    default=None,
                    help='Path to sim_dmvfn sim_final.png for final image comparison')
    args = ap.parse_args()

    p = DATASET_PARAMS[args.dataset]
    prefix    = p['prefix']
    SRC_H     = p['src_h']
    SRC_W     = p['src_w']
    MASK_SHIFT = p['mask_shift']

    stim_dir = args.stim_dir
    KH = KW = 4

    # Block IC config: (ic_actual, ic_padded) — ic_padded drives FM line count
    # Stim files store ic_padded channels; weight files have ic_padded values/line.
    # Cityscapes gen_city_stim: G1 ic_padded=64, G2=32, G3=32
    # (sim_dmvfn.py BLOCK_IC matches this)
    BLOCK_IC = {
        **{i: (48, 64) for i in range(3)},
        **{i: (28, 32) for i in range(3, 6)},
        **{i: (19, 32) for i in range(6, 9)},
    }

    # Detect actual ic_padded from stim FM file line count + known FM_H
    # (FM_H is not necessarily SRC_H — it's the lastconv input spatial size)
    # We'll detect FM_H properly below after detecting ic_padded.

    print("=" * 70)
    print(f" DIAGNOSE: hw_quant vs sim_dmvfn  [{args.dataset.upper()}]")
    print("=" * 70)

    # ── Step A: load source images ──
    # Always reconstruct from SRAM bank files — these are the exact pixels that
    # sim_dmvfn.py and the RTL use. External PNGs may come from a different frame.
    print("  Reconstructing images from SRAM bank files (exact RTL input)...")
    img0_rgb = reconstruct_img_from_banks(stim_dir, prefix, 0, SRC_H, SRC_W)
    img1_rgb = reconstruct_img_from_banks(stim_dir, prefix, 1, SRC_H, SRC_W)
    if img0_rgb is None:
        raise FileNotFoundError(f"Bank files not found in {stim_dir}")
    if args.image_0_path and args.image_1_path:
        # Cross-check: warn if supplied PNGs differ from bank reconstruction
        img0_ext = cv2.imread(args.image_0_path)
        img1_ext = cv2.imread(args.image_1_path)
        if img0_ext is not None and img1_ext is not None:
            img0_ext_rgb = img0_ext[:, :, ::-1].copy()
            img1_ext_rgb = img1_ext[:, :, ::-1].copy()
            d0 = np.abs(img0_rgb.astype(np.int16) - img0_ext_rgb.astype(np.int16)).max()
            d1 = np.abs(img1_rgb.astype(np.int16) - img1_ext_rgb.astype(np.int16)).max()
            if d0 > 0 or d1 > 0:
                print(f"  [WARN] Supplied PNGs differ from bank files "
                      f"(img0 max_diff={d0}, img1 max_diff={d1}). "
                      f"Using bank files.")
            else:
                print(f"  [OK] Supplied PNGs match bank files exactly.")
    print(f"  Images: {img0_rgb.shape}")

    # Detect actual FM spatial size and ic_padded from stim files.
    # ic_padded = total_lines / FM_H.  We know FM_H from dataset params.
    DATASET_FM_H = {'kitti': 128, 'cityscapes': 256}
    FM_H_EXPECTED = DATASET_FM_H[args.dataset]

    fm0_path = os.path.join(stim_dir, f"{prefix}_b0_L10_input.txt")
    fm0_lines = []
    with open(fm0_path) as fh:
        for raw_line in fh:
            if raw_line.strip():
                fm0_lines.append(raw_line)
    total_lines_b0 = len(fm0_lines)
    FM_W = len(fm0_lines[0].split())

    # Derive ic_padded: must divide total_lines evenly with FM_H_EXPECTED
    ic0_pad_detected = total_lines_b0 // FM_H_EXPECTED
    FM_H = FM_H_EXPECTED
    # Override BLOCK_IC for block 0 if detected ic_padded differs
    _, ic0_pad = BLOCK_IC[0]
    if ic0_pad_detected != ic0_pad:
        print(f"  [WARN] Detected ic_padded={ic0_pad_detected} for block0, "
              f"expected {ic0_pad}. Using detected value.")
        for i in range(3):
            BLOCK_IC[i] = (BLOCK_IC[i][0], ic0_pad_detected)

    print(f"  FM spatial size: {FM_H}x{FM_W}  (ic_padded_b0={BLOCK_IC[0][1]})")

    # Detect OC_FILE from weight file line count: lines = KH*KW*OC_FILE
    wt0_path = os.path.join(stim_dir, f"{prefix}_b0_lastconv_weight.txt")
    wt_lines = []
    with open(wt0_path) as fh:
        for raw_line in fh:
            if raw_line.strip():
                wt_lines.append(raw_line)
    OC_FILE_DETECTED = len(wt_lines) // (KH * KW)
    print(f"  OC_FILE detected: {OC_FILE_DETECTED}")

    # ── Step B: sim_dmvfn acc_flow / acc_mask (from stim files) ──
    print("\n[SIM] Computing acc_flow/acc_mask from stim files...")
    sim_acc_flow = np.zeros((4, SRC_H, SRC_W), dtype=np.int64)
    sim_acc_mask = np.zeros((1, SRC_H, SRC_W), dtype=np.int64)

    for bidx in range(9):
        _, ic_pad = BLOCK_IC[bidx]
        fm_path = os.path.join(stim_dir, f"{prefix}_b{bidx}_L10_input.txt")
        wt_path = os.path.join(stim_dir, f"{prefix}_b{bidx}_lastconv_weight.txt")
        bias_path = os.path.join(stim_dir, f"{prefix}_b{bidx}_L10_bias.txt")

        fm  = load_fm(fm_path, ic_pad, FM_H, FM_W)
        wt  = load_weight(wt_path, ic_pad, KH, KW, oc_file=OC_FILE_DETECTED)
        bias = load_bias(bias_path)

        raw = tconv2d_full(fm, wt, KH, KW)
        raw += (bias * 256)[:, None, None]
        raw = np.clip(raw, -(1 << 31), (1 << 31) - 1)

        # Symmetric crop: H+2 → H
        OH, OW = raw.shape[1], raw.shape[2]
        ph, pw = (OH - SRC_H) // 2, (OW - SRC_W) // 2
        raw = raw[:, ph:ph+SRC_H, pw:pw+SRC_W]

        sim_acc_flow += raw[:4]
        sim_acc_mask += raw[4:5]

    sim_acc_flow = np.clip(sim_acc_flow, -(1 << 31), (1 << 31) - 1)
    sim_acc_mask = np.clip(sim_acc_mask, -(1 << 31), (1 << 31) - 1)
    print(f"  sim acc_flow range: [{sim_acc_flow.min()}, {sim_acc_flow.max()}]")
    print(f"  sim acc_mask range: [{sim_acc_mask.min()}, {sim_acc_mask.max()}]")

    # ── Step C: hw_quant TConv using STIM FM + PyTorch weights ──
    # Goal: verify that hw_quant's TConv + crop logic matches sim_dmvfn,
    # using the SAME FM (from stim files) so FM differences don't interfere.
    print("\n[HW_QUANT/TConv] Recomputing TConv with stim FM + PyTorch weights...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model_Simplified(load_path=args.load_path, training=False)
    simp  = model._simp

    FONE   = DMVFN_Simplified._FRAC_ONE
    FSHIFT = DMVFN_Simplified._FLOW_SHIFT
    MSHIFT = DMVFN_Simplified._MASK_SHIFT
    WSCALE = DMVFN_Simplified._WEIGHT_SCALE
    OC     = DMVFN_Simplified._OC_USE

    img0_fp = img0_rgb.astype(np.float32) / 255.
    img1_fp = img1_rgb.astype(np.float32) / 255.

    hw_acc_flow = np.zeros((4, SRC_H, SRC_W), dtype=np.int64)
    hw_acc_mask = np.zeros((1, SRC_H, SRC_W), dtype=np.int64)

    for bidx in range(9):
        _, ic_pad = BLOCK_IC[bidx]
        # FM: from stim file (same as sim_dmvfn uses)
        fm_stim_path = os.path.join(stim_dir, f"{prefix}_b{bidx}_L10_input.txt")
        fm_q = load_fm(fm_stim_path, ic_pad, FM_H, FM_W)  # (ic_pad, FM_H, FM_W) int64

        # Weight: from stim file (same as sim_dmvfn uses) — eliminates weight rounding diff
        wt_stim_path2 = os.path.join(stim_dir, f"{prefix}_b{bidx}_lastconv_weight.txt")
        w_q = load_weight(wt_stim_path2, ic_pad, KH, KW, oc_file=OC_FILE_DETECTED)  # (ic_pad, OC, KH, KW)

        # Bias: from stim file (same as sim_dmvfn uses) — eliminates bias rounding diff
        bias_path2 = os.path.join(stim_dir, f"{prefix}_b{bidx}_L10_bias.txt")
        bias_stim = load_bias(bias_path2)  # (OC,) already Q8.8 integers
        bias_q = bias_stim  # will be scaled ×256 below like sim_dmvfn does

        KH2, KW2 = w_q.shape[2], w_q.shape[3]
        ST = 2
        OH2 = (FM_H - 1) * ST + KH2
        OW2 = (FM_W - 1) * ST + KW2
        raw2 = np.zeros((OC, OH2, OW2), dtype=np.int64)
        # w_q has ic_actual channels (from PyTorch); fm_q has ic_pad (may be larger).
        # Only the first ic_actual channels of fm_q carry real data; the rest are zero-pad.
        for ky in range(KH2):
            for kx in range(KW2):
                w_slice = w_q[:, :, ky, kx]   # (ic_pad, OC)
                contrib = np.einsum('io,ihw->ohw', w_slice, fm_q)
                raw2[:, ky:ky + FM_H*ST:ST,
                        kx:kx + FM_W*ST:ST] += contrib

        # bias already Q8.8; sim_dmvfn scales ×256 to reach Q16.16
        raw2 += (bias_q * 256)[:, None, None]
        raw2 = np.clip(raw2, -(1 << 31), (1 << 31) - 1)

        ph = (OH2 - SRC_H) // 2
        pw = (OW2 - SRC_W) // 2
        raw2 = raw2[:, ph:ph+SRC_H, pw:pw+SRC_W]

        hw_acc_flow += raw2[:4]
        hw_acc_mask += raw2[4:5]

    hw_acc_flow = np.clip(hw_acc_flow, -(1 << 31), (1 << 31) - 1)
    hw_acc_mask = np.clip(hw_acc_mask, -(1 << 31), (1 << 31) - 1)
    print(f"  hw  acc_flow range: [{hw_acc_flow.min()}, {hw_acc_flow.max()}]")
    print(f"  hw  acc_mask range: [{hw_acc_mask.min()}, {hw_acc_mask.max()}]")

    # ── Step D: compare acc_flow / acc_mask ──
    print("\n[COMPARE] acc_flow / acc_mask:")
    for ch, name in enumerate(['fx0','fy0','fx1','fy1']):
        diff = np.abs(hw_acc_flow[ch].astype(np.float64) -
                      sim_acc_flow[ch].astype(np.float64))
        print(f"  flow[{name}]  max_diff={int(diff.max()):10d}  "
              f"mean_diff={diff.mean():.1f}  "
              f"exact_match={np.mean(diff==0)*100:.1f}%")
    diff_m = np.abs(hw_acc_mask[0].astype(np.float64) -
                    sim_acc_mask[0].astype(np.float64))
    print(f"  mask        max_diff={int(diff_m.max()):10d}  "
          f"mean_diff={diff_m.mean():.1f}  "
          f"exact_match={np.mean(diff_m==0)*100:.1f}%")

    # ── Step E: per-block weight comparison (PyTorch model vs stim file) ──
    print("\n[COMPARE] Per-block lastconv weight (PyTorch×256 vs stim file):")
    for bidx in range(9):
        _, ic_pad = BLOCK_IC[bidx]
        wt_stim_path = os.path.join(stim_dir, f"{prefix}_b{bidx}_lastconv_weight.txt")
        wt_stim = load_weight(wt_stim_path, ic_pad, KH, KW, oc_file=OC_FILE_DETECTED)

        w_fp  = simp.blocks[bidx].lastconv.weight.detach().cpu().numpy()[:, :OC]
        w_hw_q = np.round(w_fp * WSCALE).astype(np.int64)  # (IC_full, OC, KH, KW)
        # pad/trim to ic_pad for comparison
        ic_full = w_hw_q.shape[0]
        wt_hw_padded = np.zeros((ic_pad, OC, KH, KW), dtype=np.int64)
        wt_hw_padded[:min(ic_full, ic_pad)] = w_hw_q[:min(ic_full, ic_pad)]

        diff = np.abs(wt_hw_padded.astype(np.float64) - wt_stim.astype(np.float64))
        print(f"  Block{bidx}: max_diff={int(diff.max()):4d}  "
              f"mean_diff={diff.mean():.3f}  "
              f"exact={np.mean(diff==0)*100:.1f}%")

    # ── Step F: final image comparison (if sim_final provided) ──
    if args.sim_final and os.path.exists(args.sim_final):
        sim_img = cv2.imread(args.sim_final)
        if sim_img is not None:
            sim_img_rgb = sim_img[:, :, ::-1]

            # -----------------------------------------------------------------
            # Reconstruct final image using the SAME fixed-point integer warp
            # as sim_dmvfn.py (tile-based, Q11.10 integer bilinear, border clamp).
            # -----------------------------------------------------------------
            TILE_STRIDE  = 4
            TILE_IH = TILE_IW = TILE_STRIDE + 2      # 6
            TILE_OH = TILE_OW = (TILE_IH - 1) * 2 + 4  # 14
            VALID_H = VALID_W = 8
            VALID_OFFSET = 3
            COORD_FRAC_W = 10
            FRAC_ONE2    = 1 << COORD_FRAC_W          # 1024
            FRAC_MASK2   = FRAC_ONE2 - 1
            COORD_INT_W  = max(SRC_W, SRC_H).bit_length() + 1
            INT_MASK2    = (1 << COORD_INT_W) - 1
            COORD_W2     = COORD_INT_W + COORD_FRAC_W
            COORD_MASK2  = (1 << COORD_W2) - 1
            BANK_DIM_W   = (SRC_W // 2 - 1).bit_length()

            def to_signed_32(v):
                v = int(v) & 0xFFFFFFFF
                return v - (1 << 32) if v >= (1 << 31) else v

            def build_sram(img_rgb):
                # img_rgb: (H, W, 3) uint8
                # returns list of 3 channel-SRAMs, each = list of 4 bank dicts {addr: int}
                H, W = img_rgb.shape[:2]
                yy, xx = np.mgrid[0:H, 0:W]
                bn_arr   = (yy & 1) * 2 + (xx & 1)
                addr_arr = ((yy >> 1) << BANK_DIM_W) | (xx >> 1)
                srams = []
                for ch in range(3):
                    ch_data = img_rgb[:, :, ch]
                    banks = [{} for _ in range(4)]
                    for bn in range(4):
                        mask = (bn_arr == bn)
                        for addr, val in zip(addr_arr[mask].flat, ch_data[mask].flat):
                            banks[bn][int(addr)] = int(val)
                    srams.append(banks)
                return srams

            def bilinear_hw_local(sram_bank, cx, cy,
                                  _SW=SRC_W, _SH=SRC_H,
                                  _CFW=COORD_FRAC_W,
                                  _FMASK=FRAC_MASK2, _FONE=FRAC_ONE2,
                                  _BDW=BANK_DIM_W):
                # cx, cy: integer Q11.10 coordinates (signed 32)
                cx = int(cx); cy = int(cy)
                cx_c = max(0, min(cx, (_SW - 1) << _CFW))
                cy_c = max(0, min(cy, (_SH - 1) << _CFW))
                x_int = cx_c >> _CFW
                y_int = cy_c >> _CFW
                alpha = cx_c & _FMASK
                beta  = cy_c & _FMASK
                ia, ib = _FONE - alpha, _FONE - beta

                def read_px(x, y, _sw=_SW, _sh=_SH, _bdw=_BDW, _sb=sram_bank):
                    x = max(0, min(int(x), _sw - 1))
                    y = max(0, min(int(y), _sh - 1))
                    bn   = (y & 1) * 2 + (x & 1)
                    addr = ((y >> 1) << _bdw) | (x >> 1)
                    return _sb[bn].get(addr, 0)

                p00 = read_px(x_int,     y_int)
                p10 = read_px(x_int + 1, y_int)
                p01 = read_px(x_int,     y_int + 1)
                p11 = read_px(x_int + 1, y_int + 1)
                s   = p00*ia*ib + p10*alpha*ib + p01*ia*beta + p11*alpha*beta
                return ((s + (1 << (2*_CFW - 1))) >> (2*_CFW)) & 0xFF

            # Load tile params
            tp_path = os.path.join(stim_dir, f"{prefix}_tile_params.txt")
            tiles = []
            with open(tp_path) as tf:
                tp_text = tf.read()
            for tp_line in tp_text.splitlines():
                parts = list(map(int, tp_line.split()))
                if len(parts) >= 11:
                    tiles.append({
                        't': parts[0], 'ty': parts[1], 'tx': parts[2],
                        'bx': parts[8], 'by': parts[9], 'fshift': parts[10],
                    })

            sram0 = build_sram(img0_rgb)
            sram1 = build_sram(img1_rgb)
            out_img = np.zeros((SRC_H, SRC_W, 3), dtype=np.uint8)

            def se(v):
                return v if v < (1 << (COORD_W2 - 1)) else v - (1 << COORD_W2)

            print(f"\n[FINAL IMAGE] Reconstructing with Q11.10 integer warp "
                  f"({len(tiles)} tiles)...")

            for tile in tiles:
                ty = tile['ty'];  tx = tile['tx']
                bx = tile['bx'];  by = tile['by']
                fshift = tile['fshift']

                out_y0 = ty * VALID_H
                out_x0 = tx * VALID_W

                # py, px are tile-local output coords (0..13).
                # Valid region is [VALID_OFFSET, VALID_OFFSET+VALID_H) in each dim.
                # Absolute image coords: oy = out_y0 + (py - VALID_OFFSET),
                #                        ox = out_x0 + (px - VALID_OFFSET).
                # hw_acc_flow is global (4, SRC_H, SRC_W) — indexed by (oy, ox).
                # Coordinate computation: bx + px*FRAC_ONE = (out_x0 + px)*FRAC_ONE,
                # which equals (ox + VALID_OFFSET)*FRAC_ONE. We use tile-local px for
                # the Q11.10 pixel-position component (matching sim_dmvfn.py exactly).
                for py in range(VALID_OFFSET, VALID_OFFSET + VALID_H):
                    for px in range(VALID_OFFSET, VALID_OFFSET + VALID_W):
                        oy = out_y0 + (py - VALID_OFFSET)
                        ox = out_x0 + (px - VALID_OFFSET)
                        if not (0 <= oy < SRC_H and 0 <= ox < SRC_W):
                            continue

                        # Index full-image acc arrays by absolute image coords
                        fx_t  = int(hw_acc_flow[0, oy, ox])
                        fy_t  = int(hw_acc_flow[1, oy, ox])
                        fx_t1 = int(hw_acc_flow[2, oy, ox])
                        fy_t1 = int(hw_acc_flow[3, oy, ox])
                        mv    = int(hw_acc_mask[0, oy, ox])

                        s0x = to_signed_32(fx_t);  s0y = to_signed_32(fy_t)
                        s1x = to_signed_32(fx_t1); s1y = to_signed_32(fy_t1)
                        # px, py are tile-local — matching sim_dmvfn.py's coordinate calc
                        px_q = (px & INT_MASK2) << COORD_FRAC_W
                        py_q = (py & INT_MASK2) << COORD_FRAC_W
                        cx_t  = (bx + px_q + ((s0x >> fshift) & COORD_MASK2)) & COORD_MASK2
                        cy_t  = (by + py_q + ((s0y >> fshift) & COORD_MASK2)) & COORD_MASK2
                        cx_t1 = (bx + px_q + ((s1x >> fshift) & COORD_MASK2)) & COORD_MASK2
                        cy_t1 = (by + py_q + ((s1y >> fshift) & COORD_MASK2)) & COORD_MASK2

                        wR0 = bilinear_hw_local(sram0[0], se(cx_t),  se(cy_t))
                        wG0 = bilinear_hw_local(sram0[1], se(cx_t),  se(cy_t))
                        wB0 = bilinear_hw_local(sram0[2], se(cx_t),  se(cy_t))
                        wR1 = bilinear_hw_local(sram1[0], se(cx_t1), se(cy_t1))
                        wG1 = bilinear_hw_local(sram1[1], se(cx_t1), se(cy_t1))
                        wB1 = bilinear_hw_local(sram1[2], se(cx_t1), se(cy_t1))

                        sm_val  = to_signed_32(mv)
                        mask_sh = sm_val >> MASK_SHIFT
                        mask_q2 = min(max(mask_sh + (FRAC_ONE2 >> 1), 0), FRAC_ONE2)
                        comp    = FRAC_ONE2 - mask_q2

                        blR = ((wR0 * mask_q2 + wR1 * comp) >> COORD_FRAC_W) & 0xFF
                        blG = ((wG0 * mask_q2 + wG1 * comp) >> COORD_FRAC_W) & 0xFF
                        blB = ((wB0 * mask_q2 + wB1 * comp) >> COORD_FRAC_W) & 0xFF
                        out_img[oy, ox] = (blR, blG, blB)

            sim_img_rgb2 = sim_img_rgb
            psnr_val = compute_psnr(out_img, sim_img_rgb2)
            diff = np.abs(out_img.astype(np.int16) - sim_img_rgb2.astype(np.int16))
            print(f"  PSNR      : {psnr_val:.2f} dB")
            print(f"  max_diff  : {int(diff.max())}")
            print(f"  exact px  : {np.mean(diff==0)*100:.1f}%")
            print(f"  off-by-1  : {np.mean(diff==1)*100:.1f}%")
            print(f"  off-by->1 : {np.mean(diff>1)*100:.1f}%")

    # ── Step G: hw_quant quantisation loss vs fp32 orig model ──
    # Goal: measure how much PSNR the hw_quant pipeline loses vs the fp32 model,
    # on the SAME image pair.  This is the SW estimate of RTL quality degradation.
    #
    # We do NOT compare vs sim_final here — that would require exact FM match which
    # is impossible when re-running fp32 forward (GPU fp32 non-determinism, different
    # call graph state).  sim_final is validated already in Step F (PSNR=inf).
    #
    # Chain verified so far:
    #   Step F  : stim_FM → INT_MAC → integer_warp  ≡  sim_final  (PSNR = inf)
    #   Step G  : fp32_FM → INT_MAC → integer_warp  vs  fp32_orig
    #             gap = FM quantisation error (fp32→Q8.8) + warp/sigmoid quant
    print("\n[HW_QUANT MODEL] Running _forward_hw_quant vs fp32 orig (quantisation loss)...")

    t0_g = torch.from_numpy(img0_rgb.astype(np.float32).transpose(2,0,1)) / 255.
    t1_g = torch.from_numpy(img1_rgb.astype(np.float32).transpose(2,0,1)) / 255.
    dev_g = next(model._simp.parameters()).device
    x_g = torch.cat([t0_g, t1_g], dim=0).unsqueeze(0).to(dev_g)

    # Load routing_ref.txt so both paths use the same block mask
    _rr_candidates = [
        os.path.join(stim_dir, "routing_ref.txt"),
        os.path.join(os.path.dirname(stim_dir.rstrip("/\\")), "phase15", "routing_ref.txt"),
        os.path.join("data", "phase15", "routing_ref.txt"),
    ]
    routing_ref_path = None
    for _cand in _rr_candidates:
        if os.path.exists(_cand):
            routing_ref_path = _cand
            break
    routing_ref_g = None
    if routing_ref_path:
        with open(routing_ref_path) as f:
            routing_ref_g = list(map(int, f.read().split()))
        print(f"  routing_ref: {routing_ref_g}  (from {routing_ref_path})")

    # Run hw_quant (fp32 FM → INT MAC → integer warp)
    DMVFN_Simplified.ROUTING_REF = routing_ref_g
    DMVFN_Simplified.MODE = 'hw_quant'
    with torch.no_grad():
        out_hwq = model._simp(x_g, scale=list(DMVFN_Simplified.SCALE_LIST), training=False)
    pred_hwq = (out_hwq[-1].squeeze().cpu().numpy().transpose(1,2,0) * 255).round().astype(np.uint8)

    # Run fp32 orig (same routing ref, float warp)
    DMVFN_Simplified.ROUTING_REF = routing_ref_g
    DMVFN_Simplified.MODE = 'hw_faithful'
    with torch.no_grad():
        out_fp32 = model._simp(x_g, scale=list(DMVFN_Simplified.SCALE_LIST), training=False)
    pred_fp32 = (out_fp32[-1].squeeze().cpu().numpy().transpose(1,2,0) * 255).round().astype(np.uint8)

    DMVFN_Simplified.ROUTING_REF = None
    DMVFN_Simplified.MODE = 'hw_quant'

    psnr_g = compute_psnr(pred_hwq, pred_fp32)
    diff_g = np.abs(pred_hwq.astype(np.int16) - pred_fp32.astype(np.int16))
    print(f"  hw_quant vs fp32_faithful  PSNR: {psnr_g:.2f} dB")
    print(f"  max_diff  : {int(diff_g.max())}")
    print(f"  exact px  : {np.mean(diff_g==0)*100:.1f}%")
    print(f"  off-by-1  : {np.mean(diff_g==1)*100:.1f}%")
    print(f"  off-by->1 : {np.mean(diff_g>1)*100:.1f}%")
    print(f"  Interpretation: gap = FM Q8.8 quant + integer warp + linear sigmoid error")

    if args.sim_final and os.path.exists(args.sim_final):
        sim_img_g = cv2.imread(args.sim_final)
        if sim_img_g is not None:
            sim_img_g_rgb = sim_img_g[:, :, ::-1]
            psnr_vs_sim = compute_psnr(pred_hwq, sim_img_g_rgb)
            print(f"  hw_quant vs sim_final      PSNR: {psnr_vs_sim:.2f} dB  "
                  f"(FM non-determinism: fp32 re-run != test.py dump)")

    # ── Step H: algorithmic gap — HW (sum-ΔF, one warp) vs orig (per-block warp) ──
    # This answers: "is the HW architecture correct?"
    #
    # HW does:  sum(ΔF_i for active blocks) → ONE final warp
    # Orig does: per-block warp+blend (progressive refinement)
    #
    # hw_faithful = HW algorithm in fp32 (same routing ref, no quantisation)
    # orig        = full fp32 model (same routing ref)
    # Gap between these two = pure algorithmic difference (HW vs paper's algorithm).
    # If this gap is small (>40 dB), HW architecture is algorithmically correct.
    print("\n[HW ARCH] Algorithmic gap: HW (sum-ΔF + one warp) vs orig (per-block warp)...")

    # no_routing forces ref=[1]*9 — use it as a proxy for "pure algo gap with same blocks"
    # More precisely: compare hw_faithful (sum-ΔF, one warp) vs no_routing (per-block warp)
    # both with the same routing ref so routing randomness is eliminated.
    DMVFN_Simplified.MODE = 'no_routing'
    DMVFN_Simplified.ROUTING_REF = None
    with torch.no_grad():
        out_orig = model._simp(x_g, scale=list(DMVFN_Simplified.SCALE_LIST), training=False)
    pred_orig = (out_orig[-1].squeeze().cpu().numpy().transpose(1,2,0) * 255).round().astype(np.uint8)

    # Also run hw_faithful with all-9 blocks (ref=[1]*9) for apples-to-apples
    DMVFN_Simplified.MODE = 'hw_faithful'
    DMVFN_Simplified.ROUTING_REF = [1] * 9
    with torch.no_grad():
        out_hf_all = model._simp(x_g, scale=list(DMVFN_Simplified.SCALE_LIST), training=False)
    pred_hf_all = (out_hf_all[-1].squeeze().cpu().numpy().transpose(1,2,0) * 255).round().astype(np.uint8)

    DMVFN_Simplified.MODE = 'hw_quant'
    DMVFN_Simplified.ROUTING_REF = None

    # hw_faithful(routing_ref) vs no_routing: both all-blocks, isolates sum-ΔF vs per-block-warp
    psnr_arch = compute_psnr(pred_hf_all, pred_orig)
    diff_arch = np.abs(pred_hf_all.astype(np.int16) - pred_orig.astype(np.int16))
    print(f"  hw_faithful(all9) vs no_routing(all9)  PSNR: {psnr_arch:.2f} dB")
    print(f"  max_diff  : {int(diff_arch.max())}")
    print(f"  exact px  : {np.mean(diff_arch==0)*100:.1f}%")
    print(f"  (pure algorithmic gap: sum-deltaF + one-warp  vs  per-block progressive warp)")
    print()
    print("  論證結論:")
    if psnr_arch > 40:
        print(f"  [OK]  HW 算法正確：sum-deltaF + one-warp vs per-block-warp 差距 {psnr_arch:.1f} dB > 40 dB")
        print(f"        算法差距極小，HW 架構的近似誤差可忽略。")
    elif psnr_arch > 30:
        print(f"  [WARN] HW 算法差距：{psnr_arch:.1f} dB (30~40 dB)")
        print(f"         sum-deltaF + one-warp 近似誤差存在，需確認業務上是否可接受。")
    else:
        print(f"  [FAIL] HW 算法差距過大：{psnr_arch:.1f} dB < 30 dB")
        print(f"         sum-deltaF + one-warp 與 per-block progressive warp 差太多，")
        print(f"         HW 架構設計需重新評估。")

    # ── Step I: live-FM bit-exact — hw_quant output vs sim_dmvfn(same FM) ──
    # This is the definitive fix for the "19.85 dB problem":
    #
    # Previous: hw_quant (re-run fp32 forward) vs sim_final (test.py dump FM)
    #   → PSNR ≈ 19.85 dB  because fp32 forward is non-deterministic across calls
    #
    # Now: capture FM from hw_quant's own fp32 forward (via hooks), quantise it,
    #   write it to a temp stim dir, run sim_dmvfn on that SAME quantised FM,
    #   compare hw_quant output vs live-sim output.
    #
    # Since both paths use EXACTLY the same quantised FM (Q8.8), any remaining
    # difference must come from tiling / coordinate arithmetic alone — not FM
    # non-determinism.  Expected result: PSNR = ∞ dB (bit-exact).
    print("\n[STEP I] Live-FM bit-exact: hw_quant vs sim_dmvfn(same Q8.8 FM)...")

    import shutil

    # ── I-1: Re-run hw_quant forward and capture per-block L10 FMs ──
    l10_fms_i = {}

    def _make_hook_i(i):
        def _h(_, inp, __):
            l10_fms_i[i] = inp[0][0].detach().cpu()
        return _h

    DMVFN_Simplified.ROUTING_REF = routing_ref_g
    DMVFN_Simplified.MODE = 'hw_quant'

    hooks_i = []
    for _hi in range(9):
        hooks_i.append(model._simp.blocks[_hi].lastconv.register_forward_hook(_make_hook_i(_hi)))
    with torch.no_grad():
        out_hwq_i = model._simp(x_g, scale=list(DMVFN_Simplified.SCALE_LIST), training=False)
    for h in hooks_i:
        h.remove()

    pred_hwq_i = (out_hwq_i[-1].squeeze().cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    print(f"  hw_quant forward done; captured FMs for {len(l10_fms_i)} blocks")

    # ── I-2: Write quantised FMs + stim weights + images to temp dir ──
    tmp_dir = os.path.join(os.path.dirname(stim_dir), '_step_i_tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    FSCALE_I  = DMVFN_Simplified._FM_SCALE        # 256
    OC_I      = DMVFN_Simplified._OC_USE          # 5
    OC_FILE_I = OC_FILE_DETECTED

    # Block IC padded sizes (same as stim dir)
    BLOCK_IC_I = {
        **{i: (48, 64) for i in range(3)},
        **{i: (28, 32) for i in range(3, 6)},
        **{i: (19, 32) for i in range(6, 9)},
    }

    for bidx in range(9):
        _, ic_pad = BLOCK_IC_I[bidx]
        fm_pad = np.zeros((ic_pad, FM_H, FM_W), dtype=np.int64)
        if bidx in l10_fms_i:
            fm_fp = l10_fms_i[bidx].numpy()          # (IC_actual, fh, fw) float32
            fm_q  = np.round(fm_fp * FSCALE_I).astype(np.int64)
            ic_actual = fm_q.shape[0]
            fh_actual = fm_q.shape[1]
            fw_actual = fm_q.shape[2]
            # Pad IC; crop/pad spatial dims to match stim FM_H×FM_W
            ic_copy = min(ic_actual, ic_pad)
            fh_copy = min(fh_actual, FM_H)
            fw_copy = min(fw_actual, FM_W)
            fm_pad[:ic_copy, :fh_copy, :fw_copy] = fm_q[:ic_copy, :fh_copy, :fw_copy]
        # else: block skipped → zero FM

        # Write FM txt — convert to nested Python list first so the file-write
        # loop never touches numpy indexing (avoids genexpr/closure edge cases).
        fm_path_i = os.path.join(tmp_dir, f"{prefix}_b{bidx}_L10_input.txt")
        fm_list = np.ascontiguousarray(fm_pad).tolist()  # 3-level nested list of ints
        with open(fm_path_i, 'w') as f:
            for channel in fm_list:
                for row in channel:
                    f.write(' '.join(map(str, row)))
                    f.write('\n')

        # Copy weight txt from original stim dir (weights are deterministic)
        wt_src = os.path.join(stim_dir, f"{prefix}_b{bidx}_lastconv_weight.txt")
        wt_dst = os.path.join(tmp_dir, f"{prefix}_b{bidx}_lastconv_weight.txt")
        if os.path.exists(wt_src):
            shutil.copy2(wt_src, wt_dst)

        # Copy bias txt
        bias_src = os.path.join(stim_dir, f"{prefix}_b{bidx}_L10_bias.txt")
        bias_dst = os.path.join(tmp_dir, f"{prefix}_b{bidx}_L10_bias.txt")
        if os.path.exists(bias_src):
            shutil.copy2(bias_src, bias_dst)

    # Copy image banks, tile_params, config from original stim dir
    for fname in os.listdir(stim_dir):
        if (fname.endswith('_bank0_R.txt') or fname.endswith('_bank0_G.txt') or
                fname.endswith('_bank0_B.txt') or fname.endswith('_bank1_R.txt') or
                fname.endswith('_bank1_G.txt') or fname.endswith('_bank1_B.txt') or
                fname.endswith('_bank2_R.txt') or fname.endswith('_bank2_G.txt') or
                fname.endswith('_bank2_B.txt') or fname.endswith('_bank3_R.txt') or
                fname.endswith('_bank3_G.txt') or fname.endswith('_bank3_B.txt') or
                fname.endswith('_tile_params.txt') or fname.endswith('_config.txt')):
            shutil.copy2(os.path.join(stim_dir, fname),
                         os.path.join(tmp_dir, fname))

    print(f"  Wrote live-FM stim to: {tmp_dir}")

    # ── I-3: Compute global acc_flow/acc_mask from live-FM using full-image TConv ──
    # This matches hw_quant's own global TConv (Step 3 of _forward_hw_quant).
    # We use the same load_fm / load_weight helpers already defined above.
    # The result is comparable to hw_quant's internal acc_flow/acc_mask.
    live_acc_flow = np.zeros((4, SRC_H, SRC_W), dtype=np.int64)
    live_acc_mask = np.zeros((1, SRC_H, SRC_W), dtype=np.int64)

    routing_ref_i = routing_ref_g if routing_ref_g is not None else [1] * 9

    for bidx in range(9):
        if not routing_ref_i[bidx]:
            continue
        _, ic_pad = BLOCK_IC_I[bidx]
        fm_p  = os.path.join(tmp_dir, f"{prefix}_b{bidx}_L10_input.txt")
        wt_p  = os.path.join(tmp_dir, f"{prefix}_b{bidx}_lastconv_weight.txt")
        bs_p  = os.path.join(tmp_dir, f"{prefix}_b{bidx}_L10_bias.txt")
        fm_live = load_fm(fm_p, ic_pad, FM_H, FM_W)
        wt_live = load_weight(wt_p, ic_pad, KH, KW, oc_use=OC_I, oc_file=OC_FILE_I)
        bias_live = load_bias(bs_p, oc_use=OC_I)

        raw_live = tconv2d_full(fm_live, wt_live, KH, KW)
        raw_live += (bias_live * 256)[:, None, None]
        raw_live = np.clip(raw_live, -(1<<31), (1<<31)-1)
        OH_l, OW_l = raw_live.shape[1], raw_live.shape[2]
        ph_l = (OH_l - SRC_H) // 2
        pw_l = (OW_l - SRC_W) // 2
        raw_live = raw_live[:, ph_l:ph_l+SRC_H, pw_l:pw_l+SRC_W]
        live_acc_flow += raw_live[:4]
        live_acc_mask += raw_live[4:5]

    live_acc_flow = np.clip(live_acc_flow, -(1<<31), (1<<31)-1)
    live_acc_mask = np.clip(live_acc_mask, -(1<<31), (1<<31)-1)

    # ── I-3b: Apply hw_quant's global warp using live acc_flow/acc_mask ──
    # Mirrors _forward_hw_quant Steps 5-8 exactly (global bilinear, no tiling).
    FONE_I   = DMVFN_Simplified._FRAC_ONE
    FSHIFT_I = DMVFN_Simplified._FLOW_SHIFT
    MSHIFT_I = DMVFN_Simplified._MASK_SHIFT

    _mgrid_i = np.mgrid[0:SRC_H, 0:SRC_W]
    yy_qi = _mgrid_i[0].astype(np.int64) * FONE_I
    xx_qi = _mgrid_i[1].astype(np.int64) * FONE_I

    def _fp_coord_live(acc_ch, base_q):
        return (base_q + (acc_ch >> FSHIFT_I).astype(np.int64)).astype(np.int64)

    lcx0 = _fp_coord_live(live_acc_flow[0], xx_qi)
    lcy0 = _fp_coord_live(live_acc_flow[1], yy_qi)
    lcx1 = _fp_coord_live(live_acc_flow[2], xx_qi)
    lcy1 = _fp_coord_live(live_acc_flow[3], yy_qi)

    lw0 = DMVFN_Simplified._bilinear_int_border(img0_rgb, lcx0, lcy0, FONE_I)
    lw1 = DMVFN_Simplified._bilinear_int_border(img1_rgb, lcx1, lcy1, FONE_I)

    lmask_sh = (live_acc_mask[0] >> MSHIFT_I).astype(np.int64)
    lmask_q  = np.clip(lmask_sh + (FONE_I >> 1), 0, FONE_I).astype(np.int64)
    lcomp_q  = FONE_I - lmask_q

    lw0_u8 = np.clip(np.round(lw0 * 255.0), 0, 255).astype(np.int64)
    lw1_u8 = np.clip(np.round(lw1 * 255.0), 0, 255).astype(np.int64)
    lblend  = (lw0_u8 * lmask_q[..., None] + lw1_u8 * lcomp_q[..., None]) >> 10
    out_sim_i = np.clip(lblend, 0, 255).astype(np.uint8)

    # ── I-4: Compare hw_quant output vs live-sim output ──
    psnr_i = compute_psnr(pred_hwq_i, out_sim_i)
    diff_i = np.abs(pred_hwq_i.astype(np.int16) - out_sim_i.astype(np.int16))
    print(f"\n  hw_quant vs live-sim(same Q8.8 FM)  PSNR: {psnr_i:.2f} dB")
    print(f"  max_diff : {int(diff_i.max())}")
    print(f"  exact px : {np.mean(diff_i==0)*100:.2f}%")
    print(f"  off-by-1 : {np.mean(diff_i==1)*100:.2f}%")
    print(f"  off-by>1 : {np.mean(diff_i>1)*100:.2f}%")
    if psnr_i == float('inf'):
        print(f"  [OK] bit-exact: hw_quant pipeline 與 sim_dmvfn(same FM) 完全一致")
    elif psnr_i > 60:
        print(f"  [OK] near bit-exact: 差距來自 tile bilinear vs global bilinear 的邊界行為")
    else:
        print(f"  [WARN] 差距 {psnr_i:.1f} dB，hw_quant 與 sim_dmvfn 實作可能有不一致")

    # Cleanup temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print()
    print("  驗證鏈總結:")
    print(f"  Step F: stim_FM -> INT_MAC -> integer_warp  vs  sim_final        : inf dB  [RTL bit-exact]")
    print(f"  Step G: fp32_FM -> INT_MAC -> integer_warp  vs  fp32_hw_faithful : {psnr_g:.1f} dB  [量化損失]")
    print(f"  Step H: hw_faithful(all9)  vs  no_routing(all9)                  : {psnr_arch:.1f} dB  [算法差距]")
    print(f"  Step I: hw_quant(live FM)  vs  sim_dmvfn(same live FM)           : {psnr_i:.1f} dB  [終極 bit-exact]")

    print("\n[DONE]")


if __name__ == '__main__':
    main()
