#!/usr/bin/env python3
"""
gen_city_stim.py —  Full-Resolution 9-Block E2E Verification
==============================================================================
Full 256x512 FM for all 9 blocks → 8192 tiles (6x6 overlapping input) → 512x1024 output image.

Corrected parameters (matching PyTorch arch.py):
  G1 (blocks 0-2): IC=48 (nf=160→160//4+8),  FM 256x512
  G2 (blocks 3-5): IC=32 (28 real + 4 pad),   FM 256x512
  G3 (blocks 6-8): IC=24 (19 real + 5 pad),   FM 256x512

HW accumulation pattern per tile (MACPE=8):
  OC pass 0 (flow, OC 0-3):
    Block0 skip=0 (init) → all remaining IC passes skip=1
    Blocks 1-8 ALL skip=1
    Total: 6+6+6 + 4+4+4 + 3+3+3 = 39 IC passes
  OC pass 1 (mask, OC 4, ALL 9 blocks accumulated):
    Block0 skip=0 (init) → all remaining skip=1
    Same 39 IC passes

BASE_X=0 BASE_Y=0 (full image from origin)

Usage:
  cd src/MVFB
  python ../python/gen_city_stim.py
"""

import numpy as np
import os, sys, time

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR     = os.path.dirname(SCRIPT_DIR)
OUT_DIR     = os.path.join(SRC_DIR, "MVFB", "stim_data", "cityscapes")
DATA_DIR    = os.path.join(SRC_DIR, "cityscapes")
DATA15_DIR  = os.path.join(DATA_DIR, "phase15")

# ---------------------------------------------------------------------------
# Layer / tiling parameters
# ---------------------------------------------------------------------------
NUM_BLOCKS  = 9
OC_FILE     = 8
OC          = 5
KH = KW     = 4
STRIDE      = 2
KK          = KH * KW    # 16
PE_NUM      = 8
MACPE       = 32
OUT_DEPTH   = 512   # must hold 2*TILE_PX=392 (flow+mask); 512 for clean power-of-2

# Group IC params (corrected)
G1_IC_ACT   = 48   # blocks 0-2: actual IC (nf=160 -> 160//4+8)
G1_IC       = 48   # blocks 0-2: nf=160 -> 160//4+8
G2_IC       = 32   # blocks 3-5: nf=80  ->  80//4+8 (28 real + 4 pad to MACPE=32 multiple)
G3_IC_ACT   = 19   # blocks 6-8 actual
G3_IC       = 32   # blocks 6-8 padded to MACPE=32 multiple (ceil(19/32)*32=32)
G1_IC_PASSES = -(-G1_IC // MACPE)   # ceil(48/32)=2
G2_IC_PASSES = -(-G2_IC // MACPE)   # ceil(32/32)=1
G3_IC_PASSES = -(-G3_IC // MACPE)   # ceil(32/32)=1
# Pad G1_IC to MACPE multiple
G1_IC       = G1_IC_PASSES * MACPE   # 64
MAX_IC      = G1_IC   # 64

# FM dimensions (corrected: all groups 512x1024)
FM_H        = 256
FM_W        = 512

# Tiling — full resolution
# TILE_STRIDE: distance between tile origins in input FM (= valid output pixels per tile)
# TILE_IH/IW:  actual input region per tile (TILE_STRIDE+1 for 1-row overlap on leading edge)
#   Overlap by 1 input row/col on the "before" side so that the tconv valid rows 2..TILE_OH-3
#   receive contributions from both the previous tile's last row and the current tile's first row,
#   eliminating the cross-tile contamination checkerboard artifact.
TILE_STRIDE = 4              # input FM rows/cols between tile origins
TILE_IH     = TILE_STRIDE + 2   # 6
TILE_IW     = TILE_STRIDE + 2   # 6
TILES_Y     = FM_H // TILE_STRIDE    # 64
TILES_X     = FM_W // TILE_STRIDE    # 128
NUM_TILES   = TILES_Y * TILES_X      # 8192
TILE_OH     = (TILE_IH - 1) * STRIDE + KH   # 14  (was incorrectly 12)
TILE_OW     = (TILE_IW - 1) * STRIDE + KW   # 14  (was incorrectly 12)
TILE_PX     = TILE_OH * TILE_OW              # 196 (was incorrectly 144)
assert 2 * TILE_PX <= OUT_DEPTH, f"2*TILE_PX={2*TILE_PX} > OUT_DEPTH={OUT_DEPTH}"

TOTAL_BLEND_PX = NUM_TILES * TILE_PX         # 1179648
CROP_H      = FM_H     # 256 (full)
CROP_W      = FM_W     # 512 (full)

# ---------------------------------------------------------------------------
# Q11.10 warp parameters (Updated for Cityscapes)
# ---------------------------------------------------------------------------
SRC_H, SRC_W = 512, 1024
COORD_FRAC_W = 10
COORD_INT_W  = 12
FRAC_ONE     = 1 << COORD_FRAC_W
FRAC_MASK    = FRAC_ONE - 1
INT_MASK     = (1 << COORD_INT_W) - 1
COORD_W      = COORD_INT_W + COORD_FRAC_W
COORD_MASK   = (1 << COORD_W) - 1

# Full image from origin
BASE_X = 0
BASE_Y = 0

# ---------------------------------------------------------------------------
# Per-block configuration
# ---------------------------------------------------------------------------
BLOCK_CFG = [
    (0, G1_IC_ACT, G1_IC, FM_H, "phase15/city_b0_L10_input.txt", "phase15/block0/city_b0_L10_weight.txt", "phase15/block0/city_b0_L10_bias.txt"),
    (1, G1_IC_ACT, G1_IC, FM_H, "phase16/city_b1_L10_input.txt", "phase16/block1/city_b1_L10_weight.txt", "phase16/block1/city_b1_L10_bias.txt"),
    (2, G1_IC_ACT, G1_IC, FM_H, "phase16/city_b2_L10_input.txt", "phase16/block2/city_b2_L10_weight.txt", "phase16/block2/city_b2_L10_bias.txt"),
    (3, 28, G2_IC, FM_H, "phase17/city_b3_L10_input.txt", "phase17/block3/city_b3_L10_weight.txt", "phase17/block3/city_b3_L10_bias.txt"),
    (4, 28, G2_IC, FM_H, "phase17/city_b4_L10_input.txt", "phase17/block4/city_b4_L10_weight.txt", "phase17/block4/city_b4_L10_bias.txt"),
    (5, 28, G2_IC, FM_H, "phase17/city_b5_L10_input.txt", "phase17/block5/city_b5_L10_weight.txt", "phase17/block5/city_b5_L10_bias.txt"),
    (6, G3_IC_ACT, G3_IC, FM_H, "phase18/city_b6_L10_input.txt", "phase18/block6/city_b6_L10_weight.txt", "phase18/block6/city_b6_L10_bias.txt"),
    (7, G3_IC_ACT, G3_IC, FM_H, "phase18/city_b7_L10_input.txt", "phase18/block7/city_b7_L10_weight.txt", "phase18/block7/city_b7_L10_bias.txt"),
    (8, G3_IC_ACT, G3_IC, FM_H, "phase18/city_b8_L10_input.txt", "phase18/block8/city_b8_L10_weight.txt", "phase18/block8/city_b8_L10_bias.txt"),
]


# ===========================================================================
# TConv2d golden model
# ===========================================================================
def tconv2d_golden(inp, weight, stride):
    n_ic, ih, iw = inp.shape
    _, n_oc, kh, kw = weight.shape
    oh = (ih - 1) * stride + kh
    ow = (iw - 1) * stride + kw
    out = np.zeros((n_oc, oh, ow), dtype=np.int64)
    for ky in range(kh):
        for kx in range(kw):
            w = weight[:, :, ky, kx]
            contrib = np.einsum('io,ihw->ohw',
                                w.astype(np.int64), inp.astype(np.int64))
            out[:, ky:ky + ih * stride:stride,
                   kx:kx + iw * stride:stride] += contrib
    return out


# ===========================================================================
# Q9.10 warp helpers
# ===========================================================================
def to_signed_32(val):
    val = int(val) & 0xFFFFFFFF
    return val - 0x100000000 if val & 0x80000000 else val


def compute_warp_coord(flow_raw, shift, base, pixel_pos):
    s       = to_signed_32(flow_raw)
    shifted = (s >> shift) & COORD_MASK
    px_q    = (pixel_pos & INT_MASK) << COORD_FRAC_W
    return (base + px_q + shifted) & COORD_MASK


BANK_DIM_W = (SRC_W // 2 - 1).bit_length()  # = 9  (ceil(log2(SRC_W/2)) for 1024-wide)
BANK_ADDR_W_PY = 17   # RTL addr wire width; addr is truncated to this many bits


def img_to_sram_banks(img):
    """Convert (H, W) single-channel uint8/uint16 image to 4-bank SRAM dict arrays.
    Matches RTL Address_Generator: bank = {is_Y_odd, is_X_odd}, addr = {by, bx}.
    """
    H, W = img.shape
    sram = [dict() for _ in range(4)]
    for y in range(H):
        for x in range(W):
            bid  = (y % 2) * 2 + (x % 2)
            addr = ((y // 2) << BANK_DIM_W) | (x // 2)
            sram[bid][addr] = int(img[y, x])
    return sram


def sram_read(sram_banks, bid, addr):
    """Read from bank — returns 0 for unpopulated addresses (matches RTL zero-init)."""
    return sram_banks[bid].get(addr, 0)


def bilinear_interpolate_hw(sram_banks, coord_x, coord_y):
    """Bilinear interpolation matching RTL Address_Generator + Crossbar + MAC.

    Border-mode clamping: coordinates are clamped to [0, (SRC-1)<<FRAC]
    before address/weight generation, matching PyTorch grid_sample(
    padding_mode='border', align_corners=True) and the RTL clamp logic.
    """
    # Sign-extend full coordinate
    cx_s = coord_x if coord_x < (1 << (COORD_W - 1)) else coord_x - (1 << COORD_W)
    cy_s = coord_y if coord_y < (1 << (COORD_W - 1)) else coord_y - (1 << COORD_W)

    # Border-mode clamp
    max_cx = (SRC_W - 1) << COORD_FRAC_W
    max_cy = (SRC_H - 1) << COORD_FRAC_W
    cx_c = max(0, min(cx_s, max_cx))
    cy_c = max(0, min(cy_s, max_cy))

    x_int = cx_c >> COORD_FRAC_W
    y_int = cy_c >> COORD_FRAC_W
    alpha = cx_c & FRAC_MASK
    beta  = cy_c & FRAC_MASK

    x0, x1 = x_int, x_int + 1
    y0, y1 = y_int, y_int + 1

    def read_pixel(x, y):
        x = max(0, min(x, SRC_W - 1))
        y = max(0, min(y, SRC_H - 1))
        bid  = (y & 1) * 2 + (x & 1)
        bx   = (x >> 1) & ((1 << BANK_DIM_W) - 1)
        by   = (y >> 1) & ((1 << BANK_DIM_W) - 1)
        addr = ((by << BANK_DIM_W) | bx)
        return sram_read(sram_banks, bid, addr)

    p_TL = read_pixel(x0, y0)
    p_TR = read_pixel(x1, y0)
    p_BL = read_pixel(x0, y1)
    p_BR = read_pixel(x1, y1)

    ia, ib = FRAC_ONE - alpha, FRAC_ONE - beta
    s = (p_TL * ia * ib +
         p_TR * alpha * ib +
         p_BL * ia * beta +
         p_BR * alpha * beta)
    return ((s + (1 << (2 * COORD_FRAC_W - 1))) >> (2 * COORD_FRAC_W)) & 0xFFFF


# Q8.8 FM * Q8.8 Weight = Q16.16 raw output.
# To get Q9.10 displacement: raw / (65536 / FRAC_ONE) = raw / 64 = raw >> 6.
# This is a GLOBAL constant, not per-tile.
FLOW_SHIFT = 16 - COORD_FRAC_W  # 6


def find_valid_shift(out5, t_oh, t_ow, base_x, base_y):
    """Find smallest shift keeping all warped coords within image bounds."""
    for shift in range(32):
        ok = True
        for py in range(t_oh):
            for px in range(t_ow):
                for oc_xy in [(0, 1), (2, 3)]:
                    cx = compute_warp_coord(int(out5[oc_xy[0], py, px]),
                                            shift, base_x, px)
                    cy = compute_warp_coord(int(out5[oc_xy[1], py, px]),
                                            shift, base_y, py)
                    if ((cx >> COORD_FRAC_W) & INT_MASK) >= SRC_W - 1:
                        ok = False; break
                    if ((cy >> COORD_FRAC_W) & INT_MASK) >= SRC_H - 1:
                        ok = False; break
                if not ok: break
            if not ok: break
        if ok:
            return shift
    return 16


def find_mask_shift(acc_mask_all):
    """Find MASK_SHIFT such that clamp((val >> shift) + FRAC_ONE//2, 0, FRAC_ONE)
    provides a useful linear approximation of sigmoid over the data range.

    We want the linear region to cover ~[-2, 2] in logit space for a decent
    approximation. raw ≈ float_logit * 65536, so we need:
    float_logit * 65536 / 2^MSHIFT ≈ ±FRAC_ONE/2 at logit=±2
    => 2 * 65536 / 2^MSHIFT = FRAC_ONE/2 = 64
    => 2^MSHIFT = 2048 => MSHIFT = 11

    But we auto-tune based on actual data range for better fit.
    """
    max_abs = max(abs(int(acc_mask_all.max())), abs(int(acc_mask_all.min())))
    if max_abs == 0:
        return 1
    # Target: max_abs >> shift ≈ FRAC_ONE/2 (half range, since we add offset)
    target = FRAC_ONE // 2
    for shift in range(20):
        if (max_abs >> shift) <= target * 4:
            return max(shift, 1)
    return 16


# ===========================================================================
# SRAM bank helpers
# ===========================================================================
def load_img_from_banks(prefix, data_dir, H=SRC_H, W=SRC_W):
    banks = [[], [], [], []]
    for n in range(4):
        fpath = os.path.join(data_dir, f"{prefix}_bank{n}.txt")
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    r, g, b = [int(x, 16) for x in line.split()]
                    banks[n].append((r, g, b))
    img = np.zeros((H, W, 3), dtype=np.uint8)
    bank_iters = [iter(banks[n]) for n in range(4)]
    for y in range(H):
        for x in range(W):
            bid = (y % 2) * 2 + (x % 2)
            r, g, b = next(bank_iters[bid])
            img[y, x] = (r, g, b)
    return img


# ===========================================================================
# Constraint table generation
# ===========================================================================
def gen_constraints_tconv2d(kh, kw, t_ih, t_iw, stride, y_offset=0):
    table = []
    for ky in range(kh):
        for kx in range(kw):
            table.append((
                0,       t_iw - 1,                    1,
                0,       t_ih - 1,                    1,
                kx,      kx + (t_iw - 1) * stride,   stride,
                ky + y_offset,
                ky + (t_ih - 1) * stride + y_offset,
                stride,
            ))
    return table


# ===========================================================================
# File I/O helpers
# ===========================================================================
def load_fm_full(filepath, n_ic_actual, n_ic_padded, fm_h, fm_w):
    """Load full FM (no crop)."""
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == n_ic_actual * fm_h, \
        (f"{os.path.basename(filepath)}: expected {n_ic_actual * fm_h} lines, "
         f"got {len(lines)}")
    fm = np.zeros((n_ic_padded, fm_h, fm_w), dtype=np.int64)
    for idx, line in enumerate(lines):
        c, y = divmod(idx, fm_h)
        vals = line.split()
        for x in range(min(len(vals), fm_w)):
            fm[c, y, x] = int(vals[x])
    return fm


def load_weight_tconv2d(filepath, n_ic, n_oc_use, kh, kw, n_oc_file=None):
    if n_oc_file is None:
        n_oc_file = n_oc_use
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == kh * kw * n_oc_file, \
        (f"{os.path.basename(filepath)}: expected {kh * kw * n_oc_file} lines, "
         f"got {len(lines)}")
    weight = np.zeros((n_ic, n_oc_use, kh, kw), dtype=np.int64)
    idx = 0
    for k in range(kh * kw):
        ky, kx = k // kw, k % kw
        for oc in range(n_oc_file):
            vals = list(map(int, lines[idx].split()))
            if oc < n_oc_use:
                for ic in range(min(len(vals), n_ic)):
                    weight[ic, oc, ky, kx] = vals[ic]
            idx += 1
    return weight


def write_fm_stim(filepath, fm):
    IC_, H, W = fm.shape
    with open(filepath, 'w') as f:
        for c in range(IC_):
            for y in range(H):
                f.write(" ".join(str(int(fm[c, y, x])) for x in range(W)) + "\n")


def write_weight_stim(filepath, weight):
    n_ic, n_oc, kh, kw = weight.shape
    with open(filepath, 'w') as f:
        for k in range(kh * kw):
            ky, kx = k // kw, k % kw
            for oc in range(n_oc):
                f.write(" ".join(str(int(weight[ic, oc, ky, kx]))
                                 for ic in range(n_ic)) + "\n")


# ===========================================================================
# Main
# ===========================================================================
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print(" Cityscapes: Full-Resolution 9-Block E2E Verification")
    print(f"   G1 (b0-2): IC={G1_IC} passes={G1_IC_PASSES}")
    print(f"   G2 (b3-5): IC={G2_IC} passes={G2_IC_PASSES}")
    print(f"   G3 (b6-8): IC={G3_IC} passes={G3_IC_PASSES}")
    print(f"   OC={OC}  K={KH}x{KW}  stride={STRIDE}")
    print(f"   FM: {FM_H}x{FM_W} (full)")
    print(f"   Tiles: {TILES_Y}x{TILES_X} = {NUM_TILES}")
    print(f"   Blend px: {TOTAL_BLEND_PX}")
    print(f"   BASE_X=0x{BASE_X:04x}  BASE_Y=0x{BASE_Y:04x}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load FMs and weights for all 9 blocks
    # ------------------------------------------------------------------
    fms, weights, biases = [], [], []

    for (bidx, ic_actual, ic_padded, fm_h_blk,
         fm_rel, wt_rel, bias_rel) in BLOCK_CFG:
        fm_path = os.path.join(DATA_DIR, fm_rel)
        wt_path = os.path.join(DATA_DIR, wt_rel)
        bias_path = os.path.join(DATA_DIR, bias_rel)


        if not os.path.exists(fm_path):
            print(f"ERROR: {fm_path} not found."); sys.exit(1)
        if not os.path.exists(wt_path):
            print(f"ERROR: {wt_path} not found."); sys.exit(1)

        fm = load_fm_full(fm_path, ic_actual, ic_padded, fm_h_blk, FM_W)
        fms.append(fm)
        print(f"  Block{bidx} FM: {fm.shape}  "
              f"range=[{int(fm[:ic_actual].min())}, {int(fm[:ic_actual].max())}]")

        wt = load_weight_tconv2d(wt_path, ic_padded, OC, KH, KW,
                                 n_oc_file=OC_FILE)
        weights.append(wt)
        print(f"  Block{bidx} wt: {wt.shape}  "
              f"range=[{int(wt.min())}, {int(wt.max())}]")
        
        if os.path.exists(bias_path):
            with open(bias_path, 'r') as f:
                b_arr = np.array([int(x) for x in f.read().split()], dtype=np.int64)
        else:
            b_arr = np.zeros(OC, dtype=np.int64)
        biases.append(b_arr)

        # Write stim FM and weight files
        write_fm_stim(os.path.join(OUT_DIR, f"city_b{bidx}_L10_input.txt"), fm)
        write_weight_stim(os.path.join(OUT_DIR, f"city_b{bidx}_lastconv_weight.txt"), wt)
        
        # 🌟 複製 Bias 給 sim_dmvfn 使用！
        with open(os.path.join(OUT_DIR, f"city_b{bidx}_L10_bias.txt"), 'w') as f:
            f.write(" ".join(str(v) for v in b_arr) + "\n")
    print()

    # ------------------------------------------------------------------
    # 2. Load source images (reused from Phase 15)
    # ------------------------------------------------------------------
    img0 = load_img_from_banks("p15_img0", DATA15_DIR)
    img1 = load_img_from_banks("p15_img1", DATA15_DIR)
    print(f"  Images loaded from Phase 15: {img0.shape}")

    # Build per-channel SRAM bank arrays matching RTL {by, bx} addressing
    # This ensures OOB coordinates read 0 (from zero-init gaps), matching RTL
    sram0 = [img_to_sram_banks(img0[:, :, ch]) for ch in range(3)]  # R, G, B
    sram1 = [img_to_sram_banks(img1[:, :, ch]) for ch in range(3)]
    print(f"  SRAM banks built (4 banks per channel, stride-256 addressing)")
    print()

    # ------------------------------------------------------------------
    # 3. Write constraint tables (same for all tiles)
    # ------------------------------------------------------------------
    ct_oc0 = gen_constraints_tconv2d(KH, KW, TILE_IH, TILE_IW,
                                     STRIDE, y_offset=0)
    ct_oc1 = gen_constraints_tconv2d(KH, KW, TILE_IH, TILE_IW,
                                     STRIDE, y_offset=TILE_OH)
    for suffix, ct in [("oc0", ct_oc0), ("oc1", ct_oc1)]:
        ct_path = os.path.join(OUT_DIR, f"city_ct_{suffix}.txt")
        with open(ct_path, 'w') as f:
            for entry in ct:
                f.write(" ".join(str(int(v)) for v in entry) + "\n")
    print(f"  Constraint tables written (shared for all {NUM_TILES} tiles)")

    # ------------------------------------------------------------------
    # 4. Per-tile processing: compute golden TConv2d + warp + blend
    # ------------------------------------------------------------------
    print(f"\n  Processing {NUM_TILES} tiles...")
    golden_R, golden_G, golden_B = [], [], []
    tile_params = []

    def extract_tile_fm(fm, ty, tx):
        """Extract overlapping (TILE_IH x TILE_IW) FM patch for tile (ty, tx).
        Origin of tile in input FM is at (ty*TILE_STRIDE - 1, tx*TILE_STRIDE - 1).
        Out-of-bounds rows/cols are zero-padded (matches HW behavior).
        """
        ic_padded = fm.shape[0]
        row_start = ty * TILE_STRIDE - 1
        col_start = tx * TILE_STRIDE - 1
        row_end   = row_start + TILE_IH   # exclusive
        col_end   = col_start + TILE_IW
        tile = np.zeros((ic_padded, TILE_IH, TILE_IW), dtype=np.int64)
        r0 = max(row_start, 0);  r1 = min(row_end, FM_H)
        c0 = max(col_start, 0);  c1 = min(col_end, FM_W)
        ri0 = r0 - row_start;    ri1 = ri0 + (r1 - r0)
        ci0 = c0 - col_start;    ci1 = ci0 + (c1 - c0)
        tile[:, ri0:ri1, ci0:ci1] = fm[:, r0:r1, c0:c1]
        return tile

    # Auto-find MASK_SHIFT: run a sample tile to estimate mask magnitude
    # (use center tile for representative values)
    sample_ty, sample_tx = TILES_Y // 2, TILES_X // 2
    sample_mask_acc = np.zeros((TILE_OH, TILE_OW), dtype=np.int64)
    for i, (bidx, ic_actual, ic_padded, *_) in enumerate(BLOCK_CFG):
        tile_inp = extract_tile_fm(fms[i], sample_ty, sample_tx)
        raw = tconv2d_golden(tile_inp, weights[i], STRIDE)
        raw = np.clip(raw, -(1 << 31), (1 << 31) - 1).astype(np.int64)
        sample_mask_acc += raw[4]
    auto_mshift = find_mask_shift(sample_mask_acc)
    # Use MASK_SHIFT=7: with FRAC_ONE=1024 (Q9.10), shift=7 maps accumulator range
    # to [-512, 512] before center bias, giving [0, 1024] after bias+clamp.
    MASK_SHIFT = 8
    print(f"  MASK_SHIFT={MASK_SHIFT} (auto={auto_mshift}, sample tile mask range: "
          f"[{int(sample_mask_acc.min())}, {int(sample_mask_acc.max())}])")

    for ty in range(TILES_Y):
        for tx in range(TILES_X):
            t    = ty * TILES_X + tx
            t_oh = TILE_OH
            t_ow = TILE_OW

            # Run tconv for all 9 blocks and accumulate
            raw_outs = []
            for i, (bidx, ic_actual, ic_padded, *_) in enumerate(BLOCK_CFG):
                tile_inp = extract_tile_fm(fms[i], ty, tx)
                raw = tconv2d_golden(tile_inp, weights[i], STRIDE)
                
                # 🌟 對齊 Q16.16 加入 Bias
                bias_q16 = biases[i] * 256
                raw += bias_q16[:, None, None]
                
                raw = np.clip(raw, -(1 << 31), (1 << 31) - 1).astype(np.int64)
                raw_outs.append(raw)

            # Accumulated flow = sum of all 9 blocks' OC 0-3
            acc_flow = sum(r[:4] for r in raw_outs)
            acc_flow = np.clip(acc_flow, -(1 << 31), (1 << 31) - 1).astype(np.int64)

            # Accumulated mask = sum of all 9 blocks' OC 4
            acc_mask = sum(r[4:5] for r in raw_outs)
            acc_mask = np.clip(acc_mask, -(1 << 31), (1 << 31) - 1).astype(np.int64)

            acc_out = np.concatenate([acc_flow, acc_mask], axis=0)
            # (smoothing removed)

            # With 1-input-row overlap, local output pixel (py=0) corresponds to
            # global output row ty*TILE_STRIDE*STRIDE - STRIDE = ty*8-2.
            bx = (BASE_X + (tx * 8) * FRAC_ONE) & COORD_MASK
            by = (BASE_Y + (ty * 8) * FRAC_ONE) & COORD_MASK
            fshift = FLOW_SHIFT   # constant = 6

            # Compute warp + blend golden values
            n_oob = 0
            for py in range(t_oh):
                for px in range(t_ow):
                    fx_t   = int(acc_out[0, py, px])
                    fy_t   = int(acc_out[1, py, px])
                    fx_t1  = int(acc_out[2, py, px])
                    fy_t1  = int(acc_out[3, py, px])
                    mask_v = int(acc_out[4, py, px])

                    cx_t  = compute_warp_coord(fx_t,  fshift, bx, px)
                    cy_t  = compute_warp_coord(fy_t,  fshift, by, py)
                    cx_t1 = compute_warp_coord(fx_t1, fshift, bx, px)
                    cy_t1 = compute_warp_coord(fy_t1, fshift, by, py)

                    ix_t  = (cx_t  >> COORD_FRAC_W) & INT_MASK
                    iy_t  = (cy_t  >> COORD_FRAC_W) & INT_MASK
                    ix_t1 = (cx_t1 >> COORD_FRAC_W) & INT_MASK
                    iy_t1 = (cy_t1 >> COORD_FRAC_W) & INT_MASK
                    if (ix_t  >= SRC_W-1 or iy_t  >= SRC_H-1 or
                            ix_t1 >= SRC_W-1 or iy_t1 >= SRC_H-1):
                        n_oob += 1

                    wR0 = bilinear_interpolate_hw(sram0[0], cx_t,  cy_t)
                    wG0 = bilinear_interpolate_hw(sram0[1], cx_t,  cy_t)
                    wB0 = bilinear_interpolate_hw(sram0[2], cx_t,  cy_t)
                    wR1 = bilinear_interpolate_hw(sram1[0], cx_t1, cy_t1)
                    wG1 = bilinear_interpolate_hw(sram1[1], cx_t1, cy_t1)
                    wB1 = bilinear_interpolate_hw(sram1[2], cx_t1, cy_t1)

                    # Mask: match RTL behavior — shift, add center bias, clamp.
                    # RTL: mask_biased = mask_sh + (1 << (COORD_FRAC_W-1))
                    #      mask_q = clamp(mask_biased, 0, FRAC_ONE)
                    # Center bias is applied (confirmed with MACPE=8 VCS debug).
                    sm_val = to_signed_32(mask_v)
                    mask_sh = sm_val >> MASK_SHIFT
                    mask_biased = mask_sh + (FRAC_ONE >> 1)   # +64 center offset
                    mask_q = min(max(mask_biased, 0), FRAC_ONE)
                    comp   = FRAC_ONE - mask_q

                    blR = ((wR0*mask_q + wR1*comp) >> COORD_FRAC_W) & 0xFFFF
                    blG = ((wG0*mask_q + wG1*comp) >> COORD_FRAC_W) & 0xFFFF
                    blB = ((wB0*mask_q + wB1*comp) >> COORD_FRAC_W) & 0xFFFF

                    # Debug dump for tile 0, first 5 pixels
                    if t == 0 and py * t_ow + px < 5:
                        pix_idx = py * t_ow + px
                        print(f"  [PY-DBG px{pix_idx}] flow=({fx_t},{fy_t},{fx_t1},{fy_t1}) mask_raw={mask_v}")
                        print(f"    coord_t=({cx_t:04x},{cy_t:04x}) coord_t1=({cx_t1:04x},{cy_t1:04x})")
                        print(f"    warp_t_R={wR0} warp_t1_R={wR1} mask_q={mask_q} comp={comp}")
                        print(f"    blend_R={blR} blend_G={blG} blend_B={blB}")

                    golden_R.append(blR)
                    golden_G.append(((wG0*mask_q + wG1*comp) >> COORD_FRAC_W) & 0xFFFF)
                    golden_B.append(((wB0*mask_q + wB1*comp) >> COORD_FRAC_W) & 0xFFFF)

            tile_params.append(
                (t, ty, tx, TILE_IH, TILE_IW, t_oh, t_ow, TILE_PX, bx, by, fshift))
            # (TILE_IH=6: overlapping input; t_oh=14, TILE_PX=196 per tile)
            if t % 100 == 0 or t == NUM_TILES - 1:
                dt = time.time() - t0
                print(f"    tile {t}/{NUM_TILES}  "
                      f"(ty={ty},tx={tx}) bx=0x{bx:04x} by=0x{by:04x} "
                      f"fshift={fshift}  [{dt:.1f}s]")

    # ------------------------------------------------------------------
    # 5. Write tile_params.txt
    # ------------------------------------------------------------------
    with open(os.path.join(OUT_DIR, "city_tile_params.txt"), 'w') as f:
        for row in tile_params:
            f.write(" ".join(str(v) for v in row) + "\n")

    # ------------------------------------------------------------------
    # 6. Write golden blend
    # ------------------------------------------------------------------
    assert len(golden_R) == TOTAL_BLEND_PX, \
        f"Expected {TOTAL_BLEND_PX} golden pixels, got {len(golden_R)}"
    for ch_name, gvals in [('R', golden_R), ('G', golden_G), ('B', golden_B)]:
        fpath = os.path.join(OUT_DIR, f"city_golden_blend_{ch_name}.txt")
        with open(fpath, 'w') as f:
            for v in gvals:
                f.write(f"{v & 0xFF:02x}\n")
    print(f"\n  Golden blend: {TOTAL_BLEND_PX} pixels  "
          f"R[{min(golden_R)},{max(golden_R)}] "
          f"G[{min(golden_G)},{max(golden_G)}] "
          f"B[{min(golden_B)},{max(golden_B)}]")

    # ------------------------------------------------------------------
    # 7. Write config
    # ------------------------------------------------------------------
    with open(os.path.join(OUT_DIR, "city_config.txt"), 'w') as f:
        f.write(f"{G1_IC} {G2_IC} {G3_IC} {OC} {KH} {KW} {STRIDE} "
                f"{CROP_H} {CROP_W} {NUM_TILES} {TOTAL_BLEND_PX} "
                f"{MASK_SHIFT} {SRC_W} {SRC_H} {NUM_BLOCKS} {OUT_DEPTH}\n")

    # ------------------------------------------------------------------
    # 8. Write SRAM bank files for RTL testbench (city_img{0,1}_bank{0-3}_{R,G,B}.txt)
    #    Source: cityscapes/phase15/p15_img{0,1}_bank{0-3}.txt
    #    Format of source: each line = "RR GG BB" (2-digit hex, space-separated)
    #    Output format: one file per channel, one 2-digit hex value per line.
    #    Each bank file: SRC_H*SRC_W/4 = 131072 lines.
    # ------------------------------------------------------------------
    print(f"\n  Splitting SRAM bank files from p15_img* -> city_img* (R/G/B) ...")
    for img_idx in range(2):
        for bn in range(4):
            src_path = os.path.join(DATA15_DIR,
                                    f"p15_img{img_idx}_bank{bn}.txt")
            if not os.path.exists(src_path):
                print(f"ERROR: {src_path} not found"); sys.exit(1)

            r_lines, g_lines, b_lines = [], [], []
            with open(src_path) as f:
                for line in f:
                    parts = line.strip().split()
                    r_lines.append(parts[0] + "\n")
                    g_lines.append(parts[1] + "\n")
                    b_lines.append(parts[2] + "\n")

            for ch_name, lines in [('R', r_lines), ('G', g_lines), ('B', b_lines)]:
                fpath = os.path.join(OUT_DIR,
                                     f"city_img{img_idx}_bank{bn}_{ch_name}.txt")
                with open(fpath, 'w') as f:
                    f.writelines(lines)

        print(f"    img{img_idx}: {len(r_lines)} entries × 4 banks split into R/G/B")

    dt_total = time.time() - t0
    print(f"\n{'='*70}")
    print(f" Cityscapes stimulus written -> {OUT_DIR}/")
    print(f"   city_b{{0..8}}_L10_input.txt      (per-block IC*{FM_H} lines x {FM_W})")
    print(f"   city_b{{0..8}}_lastconv_weight.txt (KK*OC rows x IC values)")
    print(f"   city_ct_oc0.txt / city_ct_oc1.txt  (shared for all tiles, TILE_IH=6)")
    print(f"   city_tile_params.txt  ({NUM_TILES} tiles, ih={TILE_IH} iw={TILE_IW} oh={TILE_OH} px={TILE_PX})")
    print(f"   city_golden_blend_{{R,G,B}}.txt  ({TOTAL_BLEND_PX} pixels)")
    print(f"   city_img{{0,1}}_bank{{0-3}}_{{R,G,B}}.txt  (131072 entries each, Q11.10 addr)")
    print(f"   MASK_SHIFT={MASK_SHIFT}  OUT_DEPTH={OUT_DEPTH}")
    print(f"   Total time: {dt_total:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
