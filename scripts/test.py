import os
import cv2
import sys
import math
import time
import torch
import random
import argparse
import lpips
import logging
import importlib
import numpy as np
from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim
from torch.utils.data import DataLoader, Dataset

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from utils.util import *
from model.model import Model
import model.arch as dmvfn_arch  

def _q(val): return int(round(float(val) * 256))

def _pad_weight_tconv2d(w_np, target_ic, target_oc):
    ic, oc, kh, kw = w_np.shape
    out = np.zeros((target_ic, target_oc, kh, kw), dtype=w_np.dtype)
    out[:min(ic, target_ic), :min(oc, target_oc)] = w_np[:target_ic, :target_oc]
    return out

def write_weight_tconv2d_txt(filepath, w_int):
    n_ic, n_oc, kh, kw = w_int.shape
    with open(filepath, 'w') as f:
        for k in range(kh * kw):
            ky, kx = k // kw, k % kw
            for oc in range(n_oc):
                f.write(" ".join(str(int(w_int[ic, oc, ky, kx])) for ic in range(n_ic)) + "\n")

def dump_img_banks(img_hwc_uint8, bank_prefix, out_dir):
    H, W, _ = img_hwc_uint8.shape
    banks = [[] for _ in range(4)]
    for y in range(H):
        for x in range(W):
            b_idx = (y % 2) * 2 + (x % 2)
            banks[b_idx].append(f"{int(img_hwc_uint8[y, x, 2]):02x} {int(img_hwc_uint8[y, x, 1]):02x} {int(img_hwc_uint8[y, x, 0]):02x}")
            
    for n in range(4):
        with open(os.path.join(out_dir, f"{bank_prefix}_bank{n}.txt"), 'w') as f:
            f.write("\n".join(banks[n]) + "\n")

_fp_lastconv_fm = {}

def _dump_fm_and_weights(out_dir, prefix, bidx, fm_tensor, model, macpe=32):
    wt_dir = os.path.join(out_dir, f"block{bidx}")
    os.makedirs(wt_dir, exist_ok=True)
    block = getattr(model.dmvfn, f"block{bidx}")
    
    # 🌟 核心修復：如果被路由機制跳過，強制產生「全 0」的檔案來覆蓋舊資料！
    if fm_tensor is None:
        print(f"  👉 [Block {bidx}] 🛡️ 被 Routing 機制跳過！輸出全零 (HW 將累加 0)")
        w_np = block.lastconv.weight.detach().cpu().numpy().astype(np.float32)
        actual_ic = w_np.shape[0]
        H = 256 if prefix == 'city' else 128
        W = 512 if prefix == 'city' else 416
        
        fm_q = np.zeros((actual_ic, H, W), dtype=np.int32)
        w_np = np.zeros_like(w_np)
        bias_np = np.zeros_like(block.lastconv.bias.detach().cpu().numpy().astype(np.float32)) if getattr(block.lastconv, 'bias', None) is not None else None
    else:
        _, IC, H, W = fm_tensor.shape
        fm_max = fm_tensor.max().item()
        fm_min = fm_tensor.min().item()
        print(f"  👉 [Block {bidx}] 🟢 執行中 | FM 範圍: Min={fm_min:.2f}, Max={fm_max:.2f}")
        
        fm_q = np.clip(np.round(fm_tensor[0].numpy() * 256.0).astype(np.int32), -32768, 32767)
        w_np = block.lastconv.weight.detach().cpu().numpy().astype(np.float32)
        bias_np = block.lastconv.bias.detach().cpu().numpy().astype(np.float32) if getattr(block.lastconv, 'bias', None) is not None else None

    # 1. 寫入 FM (確保覆蓋)
    actual_ic = fm_q.shape[0]
    with open(os.path.join(out_dir, f"{prefix}_b{bidx}_L10_input.txt"), 'w') as f:
        for c in range(actual_ic):
            for y in range(H):
                f.write(" ".join(str(int(fm_q[c, y, x])) for x in range(W)) + "\n")

    # 2. 寫入 Weight
    ic_sl = ((w_np.shape[0] + macpe - 1) // macpe) * macpe
    w_int = np.vectorize(_q)(_pad_weight_tconv2d(w_np, target_ic=ic_sl, target_oc=8))
    write_weight_tconv2d_txt(os.path.join(wt_dir, f"{prefix}_b{bidx}_L10_weight.txt"), w_int)

    # 3. 寫入 Bias
    if bias_np is not None:
        bias_q = np.round(bias_np * 256.0).astype(np.int32)
        with open(os.path.join(wt_dir, f"{prefix}_b{bidx}_L10_bias.txt"), 'w') as f:
            f.write(" ".join(str(int(b)) for b in bias_q) + "\n")

def dump_all_phases(img0_uint8, img1_uint8, model, dataset_name):
    prefix = "city" if "City" in dataset_name else "p"
    
    out15 = "./data/phase15"
    os.makedirs(out15, exist_ok=True)
    _dump_fm_and_weights(out15, prefix, 0, _fp_lastconv_fm.get(0), model)
    dump_img_banks(img0_uint8, "p15_img0", out15)
    dump_img_banks(img1_uint8, "p15_img1", out15)

    out16 = "./data/phase16"
    os.makedirs(out16, exist_ok=True)
    for bidx in [1, 2]: _dump_fm_and_weights(out16, prefix, bidx, _fp_lastconv_fm.get(bidx), model)

    out17 = "./data/phase17"
    os.makedirs(out17, exist_ok=True)
    for bidx in [3, 4, 5]: _dump_fm_and_weights(out17, prefix, bidx, _fp_lastconv_fm.get(bidx), model)

    out18 = "./data/phase18"
    os.makedirs(out18, exist_ok=True)
    for bidx in [6, 7, 8]: _dump_fm_and_weights(out18, prefix, bidx, _fp_lastconv_fm.get(bidx), model)

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

def base_build_dataset(name):
    return getattr(importlib.import_module('dataset.dataset', package=None), name)()

parser = argparse.ArgumentParser()
parser.add_argument('--save_img', action='store_true', help='save or not')
parser.add_argument('--val_datasets', type=str, nargs='+', default=['CityValDataset'])
parser.add_argument('--load_path', required=True, type=str, help='model path')
args = parser.parse_args()

exp = os.path.abspath('.').split('/')[-1]
loss_fn_alex = lpips.LPIPS(net='alex').to(device)
log_path = './logs/test_log/{}'.format(exp)
os.makedirs(log_path, exist_ok=True)
setup_logger('base', log_path, 'test', level=logging.INFO, screen=True, to_file=True)
logger = logging.getLogger('base')

def test(model, args):
    print('Start testing on GPU...')
    for dataset_name in args.val_datasets:
        val_dataset = base_build_dataset(dataset_name)
        val_data = DataLoader(val_dataset, batch_size=1, pin_memory=False, num_workers=1)
        evaluate(model, val_data, dataset_name, args.save_img) 

def evaluate(model, val_data, name, save_img):
    save_img_path = './save_img/test_log_{}/{}'.format(name, exp)
    
    if name in ["CityValDataset", "KittiValDataset", "DavisValDataset"]:
        with torch.no_grad():
            lpips_score, psnr_score, msssim_score, ssim_score = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
            num = val_data.__len__()
            print(f"\n🧪 開始評估 Dataset: {name} (共 {num} 張)")

            warmup_img, _ = val_data.dataset[0] 
            warmup_img = warmup_img.unsqueeze(0).to(device) / 255.
            for _ in range(5): _ = model.eval(warmup_img, name)

            total_eval_time = 0.0

            for i, data in tqdm(enumerate(val_data), desc="Processing", total=num):
                data_gpu, data_name = data
                data_gpu = data_gpu.to(device, non_blocking=True) / 255.
                
                if i == 0:
                    print(f"\n🚀 擷取 Phase 15~18 (同步動態路由, 來自第一張圖)...")
                    # 🌟 使用 frame 2,3 以對齊 model.eval() 的 imgs[:,2], imgs[:,3]
                    img0_t = data_gpu[0, 2:3].clone()
                    img1_t = data_gpu[0, 3:4].clone()
                    # 🌟 用 round 而非 truncate，避免平均 0.5 LSB 量化損失
                    img0_uint8 = np.clip(np.round(img0_t[0].cpu().numpy().transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
                    img1_uint8 = np.clip(np.round(img1_t[0].cpu().numpy().transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)

                    # 🌟 Preview routing probabilities (before Bernoulli sampling)
                    with torch.no_grad():
                        x_r = torch.cat((img0_t, img1_t), 1)
                        rv = model.dmvfn.routing(x_r[:, :6]).reshape(1, -1)
                        rv = torch.sigmoid(model.dmvfn.l1(rv))
                        rv = rv / (rv.sum(1, True) + 1e-6) * 4.5
                        rv = torch.clamp(rv, 0, 1)
                        print(f"🎯 Routing probs (pre-Bernoulli): {[f'{v:.3f}' for v in rv[0].tolist()]}")

                    # 🌟 Fix seed so Bernoulli sampling is reproducible across runs
                    torch.manual_seed(42)
                    torch.cuda.manual_seed_all(42)

                    _fp_lastconv_fm.clear()
                    hooks = []
                    def _make_hook(bidx):
                        def hook(_m, inp, _o): _fp_lastconv_fm[bidx] = inp[0].detach().cpu()
                        return hook

                    for bidx in range(9):
                        hooks.append(getattr(model.dmvfn, f"block{bidx}").lastconv.register_forward_hook(_make_hook(bidx)))

                    _merged = model.dmvfn(torch.cat((img0_t, img1_t), 1), scale=[4,4,4,2,2,2,1,1,1], training=False)

                    for h in hooks: h.remove()

                    # 🌟 Dump which blocks were actually executed by routing
                    ref_actual = [1 if bidx in _fp_lastconv_fm else 0 for bidx in range(9)]
                    print(f"🎯 Routing ref (actual): {ref_actual}  ({sum(ref_actual)}/9 blocks active)")
                    os.makedirs("./data/phase15", exist_ok=True)
                    with open("./data/phase15/routing_ref.txt", 'w') as f:
                        f.write(" ".join(str(r) for r in ref_actual) + "\n")

                    # Restore original seed
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)

                    dump_all_phases(img0_uint8, img1_uint8, model, name)
                    
                    _pred = (_merged[-1][0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                    os.makedirs("./data/preview", exist_ok=True)
                    cv2.imwrite(f"./data/preview/{name}_pred.png", _pred[:, :, ::-1].copy())
                    print("✅ 純淨測資已 Dump 完畢，接續測量效能...\n")

                start_time = time.perf_counter()
                preds = model.eval(data_gpu, name)
                total_eval_time += (time.perf_counter() - start_time)

                b,n,c,h,w = preds.shape
                gt, pred = data_gpu[0], preds[0]
                if save_img: os.makedirs(os.path.join(save_img_path, data_name[0]), exist_ok=True)

                for j in range(5):
                    psnr_score[j] += -10 * math.log10(torch.mean((gt[j+4] - pred[j])**2).cpu().data)
                    ssim_score[j] += ssim(gt[j+4:j+5], pred[j:j+1], data_range=1.0, size_average=False)
                    msssim_score[j] += ms_ssim(gt[j+4:j+5], pred[j:j+1], data_range=1.0, size_average=False)
                    lpips_score[j] += loss_fn_alex(((gt[j+4:j+5]-0.5)*2.0).clone(), ((pred[j:j+1]-0.5)*2.0).clone())

            avg_lat = total_eval_time / num
            print(f"\n🚀 效能: {avg_lat*1000:.2f} ms/frame\n")

if __name__ == "__main__":    
    model = Model(load_path=args.load_path, training=False)
    test(model, args)