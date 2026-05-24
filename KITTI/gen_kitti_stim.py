#!/usr/bin/env python3
import numpy as np
import os, sys, time

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR     = os.path.dirname(SCRIPT_DIR)
OUT_DIR     = os.path.join(SRC_DIR, "MVFB", "stim_data", "KITTI")

DATA_DIR    = os.path.join(SRC_DIR, "KITTI")
DATA15_DIR  = os.path.join(DATA_DIR, "phase15")

NUM_BLOCKS, OC_FILE, OC, KH, KW, STRIDE = 9, 8, 5, 4, 4, 2
MACPE, OUT_DEPTH = 32, 512

G1_IC_ACT, G1_IC = 48, 64
G2_IC = 32
G3_IC_ACT, G3_IC = 19, 32

FM_H, FM_W = 128, 416
CROP_H, CROP_W = FM_H, FM_W  # KITTI 的 CROP 尺寸

# 6x6 
TILE_STRIDE = 4
TILE_IH     = TILE_STRIDE + 2   # 6
TILE_IW     = TILE_STRIDE + 2   # 6
TILES_Y, TILES_X = FM_H // TILE_STRIDE, FM_W // TILE_STRIDE
NUM_TILES = TILES_Y * TILES_X
TILE_OH = (TILE_IH - 1) * STRIDE + KH   # 14
TILE_OW = (TILE_IW - 1) * STRIDE + KW   # 14
TILE_PX = TILE_OH * TILE_OW

TOTAL_BLEND_PX = NUM_TILES * TILE_PX
SRC_H, SRC_W = 256, 832

COORD_FRAC_W, COORD_INT_W = 10, 11
FRAC_ONE = 1 << COORD_FRAC_W
FRAC_MASK, INT_MASK = FRAC_ONE - 1, (1 << COORD_INT_W) - 1
COORD_W = COORD_INT_W + COORD_FRAC_W
COORD_MASK = (1 << COORD_W) - 1

BASE_X, BASE_Y = 0, 0
BANK_DIM_W = (SRC_W // 2 - 1).bit_length()
BANK_ADDR_W_PY = 16
BANK_ADDR_DEPTH = 1 << BANK_ADDR_W_PY 

# 🌟 對齊 test.py 的檔名前綴 "p_bX"
BLOCK_CFG = [
    (0, G1_IC_ACT, G1_IC, FM_H, "phase15/p_b0_L10_input.txt", "phase15/block0/p_b0_L10_weight.txt", "phase15/block0/p_b0_L10_bias.txt"),
    (1, G1_IC_ACT, G1_IC, FM_H, "phase16/p_b1_L10_input.txt", "phase16/block1/p_b1_L10_weight.txt", "phase16/block1/p_b1_L10_bias.txt"),
    (2, G1_IC_ACT, G1_IC, FM_H, "phase16/p_b2_L10_input.txt", "phase16/block2/p_b2_L10_weight.txt", "phase16/block2/p_b2_L10_bias.txt"),
    (3, 28, G2_IC, FM_H, "phase17/p_b3_L10_input.txt", "phase17/block3/p_b3_L10_weight.txt", "phase17/block3/p_b3_L10_bias.txt"),
    (4, 28, G2_IC, FM_H, "phase17/p_b4_L10_input.txt", "phase17/block4/p_b4_L10_weight.txt", "phase17/block4/p_b4_L10_bias.txt"),
    (5, 28, G2_IC, FM_H, "phase17/p_b5_L10_input.txt", "phase17/block5/p_b5_L10_weight.txt", "phase17/block5/p_b5_L10_bias.txt"),
    (6, G3_IC_ACT, G3_IC, FM_H, "phase18/p_b6_L10_input.txt", "phase18/block6/p_b6_L10_weight.txt", "phase18/block6/p_b6_L10_bias.txt"),
    (7, G3_IC_ACT, G3_IC, FM_H, "phase18/p_b7_L10_input.txt", "phase18/block7/p_b7_L10_weight.txt", "phase18/block7/p_b7_L10_bias.txt"),
    (8, G3_IC_ACT, G3_IC, FM_H, "phase18/p_b8_L10_input.txt", "phase18/block8/p_b8_L10_weight.txt", "phase18/block8/p_b8_L10_bias.txt"),
]

def tconv2d_golden(inp, weight, stride):
    n_ic, ih, iw = inp.shape
    _, n_oc, kh, kw = weight.shape
    oh = (ih - 1) * stride + kh
    ow = (iw - 1) * stride + kw
    out = np.zeros((n_oc, oh, ow), dtype=np.int64)
    for ky in range(kh):
        for kx in range(kw):
            w = weight[:, :, ky, kx]
            contrib = np.einsum('io,ihw->ohw', w.astype(np.int64), inp.astype(np.int64))
            out[:, ky:ky + ih * stride:stride, kx:kx + iw * stride:stride] += contrib
    return out

def to_signed_32(val):
    val = int(val) & 0xFFFFFFFF
    return val - 0x100000000 if val & 0x80000000 else val

def compute_warp_coord(flow_raw, shift, base, pixel_pos):
    s = to_signed_32(flow_raw)
    shifted = (s >> shift) & COORD_MASK
    px_q = (pixel_pos & INT_MASK) << COORD_FRAC_W
    return (base + px_q + shifted) & COORD_MASK

def img_to_sram_banks(img):
    H, W = img.shape
    sram = [dict() for _ in range(4)]
    for y in range(H):
        for x in range(W):
            bid  = (y % 2) * 2 + (x % 2)
            addr = ((y // 2) << BANK_DIM_W) | (x // 2)
            sram[bid][addr] = int(img[y, x])
    return sram

def bilinear_interpolate_hw(sram_banks, coord_x, coord_y):
    cx_s = coord_x if coord_x < (1 << (COORD_W - 1)) else coord_x - (1 << COORD_W)
    cy_s = coord_y if coord_y < (1 << (COORD_W - 1)) else coord_y - (1 << COORD_W)
    max_cx, max_cy = (SRC_W - 1) << COORD_FRAC_W, (SRC_H - 1) << COORD_FRAC_W
    cx_c, cy_c = max(0, min(cx_s, max_cx)), max(0, min(cy_s, max_cy))

    x_int, y_int = cx_c >> COORD_FRAC_W, cy_c >> COORD_FRAC_W
    alpha, beta  = cx_c & FRAC_MASK, cy_c & FRAC_MASK
    x0, x1, y0, y1 = x_int, x_int + 1, y_int, y_int + 1

    def read_pixel(x, y):
        x, y = max(0, min(x, SRC_W - 1)), max(0, min(y, SRC_H - 1))
        bid  = (y & 1) * 2 + (x & 1)
        addr = ((y // 2) << BANK_DIM_W) | (x // 2)
        return sram_banks[bid].get(addr, 0)

    p_TL, p_TR = read_pixel(x0, y0), read_pixel(x1, y0)
    p_BL, p_BR = read_pixel(x0, y1), read_pixel(x1, y1)

    ia, ib = FRAC_ONE - alpha, FRAC_ONE - beta
    s = (p_TL * ia * ib + p_TR * alpha * ib + p_BL * ia * beta + p_BR * alpha * beta)
    return ((s + (1 << (2 * COORD_FRAC_W - 1))) >> (2 * COORD_FRAC_W)) & 0xFFFF

FLOW_SHIFT = 16 - COORD_FRAC_W

def load_img_from_banks(prefix, data_dir, H=SRC_H, W=SRC_W):
    banks = [[], [], [], []]
    for n in range(4):
        fpath = os.path.join(data_dir, f"{prefix}_bank{n}.txt")
        with open(fpath) as f:
            for line in f:
                if line.strip():
                    r, g, b = [int(x, 16) for x in line.split()]
                    banks[n].append((r, g, b))
    img = np.zeros((H, W, 3), dtype=np.uint8)
    bank_iters = [iter(banks[n]) for n in range(4)]
    for y in range(H):
        for x in range(W):
            bid = (y % 2) * 2 + (x % 2)
            img[y, x] = next(bank_iters[bid])
    return img

def load_fm_full(filepath, n_ic_actual, n_ic_padded, fm_h, fm_w):
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]
    fm = np.zeros((n_ic_padded, fm_h, fm_w), dtype=np.int64)
    for idx, line in enumerate(lines):
        c, y = divmod(idx, fm_h)
        vals = line.split()
        for x in range(min(len(vals), fm_w)):
            fm[c, y, x] = int(vals[x])
    return fm

def load_weight_tconv2d(filepath, n_ic, n_oc_use, kh, kw, n_oc_file=None):
    n_oc_file = n_oc_file or n_oc_use
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]
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
                f.write(" ".join(str(int(weight[ic, oc, ky, kx])) for ic in range(n_ic)) + "\n")

def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70 + "\n KITTI: 6x6 Full-Resolution E2E Verification\n" + "=" * 70)

    fms, weights, biases = [], [], []
    for (bidx, ic_actual, ic_padded, fm_h_blk, fm_rel, wt_rel, bias_rel) in BLOCK_CFG:
        fm_path, wt_path, bias_path = os.path.join(DATA_DIR, fm_rel), os.path.join(DATA_DIR, wt_rel), os.path.join(DATA_DIR, bias_rel)

        fm = load_fm_full(fm_path, ic_actual, ic_padded, fm_h_blk, FM_W)
        wt = load_weight_tconv2d(wt_path, ic_padded, OC, KH, KW, n_oc_file=OC_FILE)
        
        # 🌟 讀取 Bias
        if os.path.exists(bias_path):
            with open(bias_path, 'r') as f: b_arr = np.array([int(x) for x in f.read().split()], dtype=np.int64)
        else:
            b_arr = np.zeros(OC, dtype=np.int64)
            
        fms.append(fm); weights.append(wt); biases.append(b_arr)

        write_fm_stim(os.path.join(OUT_DIR, f"kitti_b{bidx}_L10_input.txt"), fm)
        write_weight_stim(os.path.join(OUT_DIR, f"kitti_b{bidx}_lastconv_weight.txt"), wt)
        with open(os.path.join(OUT_DIR, f"kitti_b{bidx}_L10_bias.txt"), 'w') as f:
            f.write(" ".join(str(v) for v in b_arr) + "\n")

    img0, img1 = load_img_from_banks("p15_img0", DATA15_DIR), load_img_from_banks("p15_img1", DATA15_DIR)
    sram0 = [img_to_sram_banks(img0[:, :, ch]) for ch in range(3)]
    sram1 = [img_to_sram_banks(img1[:, :, ch]) for ch in range(3)]

    golden_R, golden_G, golden_B, tile_params = [], [], [], []

    def extract_tile_fm(fm, ty, tx):
        # 🌟 6x6 邊界包容擷取 (從 -1 開始)
        row_start, col_start = ty * TILE_STRIDE - 1, tx * TILE_STRIDE - 1
        row_end, col_end = row_start + TILE_IH, col_start + TILE_IW
        tile = np.zeros((fm.shape[0], TILE_IH, TILE_IW), dtype=np.int64)
        r0, r1 = max(row_start, 0), min(row_end, FM_H)
        c0, c1 = max(col_start, 0), min(col_end, FM_W)
        tile[:, r0-row_start:r0-row_start+(r1-r0), c0-col_start:c0-col_start+(c1-c0)] = fm[:, r0:r1, c0:c1]
        return tile

    MASK_SHIFT = 8
    for ty in range(TILES_Y):
        for tx in range(TILES_X):
            t = ty * TILES_X + tx
            raw_outs = []
            for i, _ in enumerate(BLOCK_CFG):
                tile_inp = extract_tile_fm(fms[i], ty, tx)
                raw = tconv2d_golden(tile_inp, weights[i], STRIDE)
                
                raw += (biases[i] * 256)[:, None, None]
                raw_outs.append(np.clip(raw, -(1 << 31), (1 << 31) - 1).astype(np.int64))

            acc_flow = np.clip(sum(r[:4] for r in raw_outs), -(1 << 31), (1 << 31) - 1).astype(np.int64)
            acc_mask = np.clip(sum(r[4:5] for r in raw_outs), -(1 << 31), (1 << 31) - 1).astype(np.int64)
            acc_out = np.concatenate([acc_flow, acc_mask], axis=0)

            # 6x6
            bx = (BASE_X + (tx * 8) * FRAC_ONE) & COORD_MASK
            by = (BASE_Y + (ty * 8) * FRAC_ONE) & COORD_MASK
            fshift = FLOW_SHIFT

            for py in range(TILE_OH):
                for px in range(TILE_OW):
                    fx_t, fy_t, fx_t1, fy_t1, mask_v = int(acc_out[0, py, px]), int(acc_out[1, py, px]), int(acc_out[2, py, px]), int(acc_out[3, py, px]), int(acc_out[4, py, px])
                    cx_t, cy_t = compute_warp_coord(fx_t, fshift, bx, px), compute_warp_coord(fy_t, fshift, by, py)
                    cx_t1, cy_t1 = compute_warp_coord(fx_t1, fshift, bx, px), compute_warp_coord(fy_t1, fshift, by, py)

                    wR0, wG0, wB0 = bilinear_interpolate_hw(sram0[0], cx_t, cy_t), bilinear_interpolate_hw(sram0[1], cx_t, cy_t), bilinear_interpolate_hw(sram0[2], cx_t, cy_t)
                    wR1, wG1, wB1 = bilinear_interpolate_hw(sram1[0], cx_t1, cy_t1), bilinear_interpolate_hw(sram1[1], cx_t1, cy_t1), bilinear_interpolate_hw(sram1[2], cx_t1, cy_t1)

                    mask_sh = to_signed_32(mask_v) >> MASK_SHIFT
                    mask_q = min(max(mask_sh + (FRAC_ONE >> 1), 0), FRAC_ONE)
                    comp = FRAC_ONE - mask_q

                    blR, blG, blB = ((wR0*mask_q + wR1*comp) >> COORD_FRAC_W) & 0xFFFF, ((wG0*mask_q + wG1*comp) >> COORD_FRAC_W) & 0xFFFF, ((wB0*mask_q + wB1*comp) >> COORD_FRAC_W) & 0xFFFF
                    golden_R.append(blR); golden_G.append(blG); golden_B.append(blB)

            tile_params.append((t, ty, tx, TILE_IH, TILE_IW, TILE_OH, TILE_OW, TILE_PX, bx, by, fshift))

    with open(os.path.join(OUT_DIR, "kitti_tile_params.txt"), 'w') as f:
        for row in tile_params: f.write(" ".join(str(v) for v in row) + "\n")

    for ch_name, gvals in [('R', golden_R), ('G', golden_G), ('B', golden_B)]:
        with open(os.path.join(OUT_DIR, f"kitti_golden_blend_{ch_name}.txt"), 'w') as f:
            for v in gvals: f.write(f"{v & 0xFF:02x}\n")

    with open(os.path.join(OUT_DIR, "kitti_config.txt"), 'w') as f:
        f.write(f"{G1_IC} {G2_IC} {G3_IC} {OC} {KH} {KW} {STRIDE} {CROP_H} {CROP_W} {NUM_TILES} {TOTAL_BLEND_PX} {MASK_SHIFT} {SRC_W} {SRC_H} {NUM_BLOCKS} {OUT_DEPTH}\n")

    for img_idx in range(2):
        for bn in range(4):
            src_path = os.path.join(DATA15_DIR, f"p15_img{img_idx}_bank{bn}.txt")
            r_dict, g_dict, b_dict = {}, {}, {}
            with open(src_path) as f:
                for seq_idx, line in enumerate(f):
                    if len(parts := line.strip().split()) >= 3:
                        addr = ((seq_idx // (SRC_W // 2)) << BANK_DIM_W) | (seq_idx % (SRC_W // 2))
                        r_dict[addr], g_dict[addr], b_dict[addr] = parts[0], parts[1], parts[2]

            for ch_name, d in [('R', r_dict), ('G', g_dict), ('B', b_dict)]:
                with open(os.path.join(OUT_DIR, f"kitti_img{img_idx}_bank{bn}_{ch_name}.txt"), 'w') as f:
                    for addr in range(BANK_ADDR_DEPTH): f.write(d.get(addr, "0000") + "\n")
    print("Done.")

if __name__ == "__main__":
    main()