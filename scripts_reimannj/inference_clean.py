#!/usr/bin/env python3
import time, os, sys, argparse, importlib, warnings, yaml
from datetime import datetime
from os.path import join, basename, dirname
import numpy as np
import rasterio as rs
import torch
from torch import set_float32_matmul_precision
from skimage.transform import rescale
from scipy.ndimage import gaussian_filter


sys.path.append('/scratch2/reimannj/pangaea-bench')

from inference_helper import *
from parser import str2bool

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")

ENC_YAMLS = {
    
    'scalemae': 'scalemae.yaml',
    'prithvi': 'prithvi.yaml',
    'remoteclip': 'remoteclip.yaml',
    'satlasnet_mi': 'satlasnet_mi.yaml',
    'satlasnet_si': 'satlasnet_si.yaml',
    
    'croma_optical': 'croma_optical.yaml',
    'resnet50_pretrained': 'resnet50_pretrained.yaml',
    'resnet50_scratch': 'resnet50_scratch.yaml',
    'vit': 'vit.yaml',
    'vit_scratch': 'vit_scratch.yaml',

    
    'croma_joint': 'croma_joint.yaml',
    
    
    'galileo_base': 'galileo_base.yaml',
    'galileo_nano': 'galileo_nano.yaml',
    'galileo_tiny': 'galileo_tiny.yaml',
    'satmae_base': 'satmae_base.yaml',
    'ssl4eo_data2vec': 'ssl4eo_data2vec.yaml',
    'ssl4eo_dino': 'ssl4eo_dino.yaml',
    'ssl4eo_moco': 'ssl4eo_moco.yaml',
    'ssl4eo_mae_optical': 'ssl4eo_mae_optical.yaml'
}

def inf_parser():
    p = argparse.ArgumentParser(description='Generic PANGAEA AGBD Inference')
    p.add_argument('--dataset_path', type=str, default='local')
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--year', type=int, default=2020)
    p.add_argument('--models', type=str, nargs='+', default=['scalemae'])
    p.add_argument('--arch', type=str, default='regupernet')
    p.add_argument('--tile_name', type=str, default='S2B_MSIL2A_20200324T135109_N9999_R024_T21JWM_20240422T081051')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--masking', type=str2bool, default='false')

    
    p.add_argument('--patch_size', nargs=2, type=int, default=None, help='H W, default=encoder input_size')
    p.add_argument('--overlap_size', nargs=2, type=int, default=None, help='H W; if missing we use --overlap_frac')
    p.add_argument('--overlap_frac', type=float, default=0.75, help='fraction of patch size to overlap (used if --overlap_size not set)')

    
    p.add_argument('--dw', action='store_true')  
    p.add_argument('--visualize', action='store_true', default=False)
    p.add_argument('--no-visualize', dest='visualize', action='store_false')
    p.add_argument('--viz_output', type=str, default=None)
    p.add_argument('--tta', action='store_true', help='flip/rotate TTA at patch level')
    p.add_argument('--smooth', type=int, default=0, help='gaussian sigma in pixels on final map (0 disables)')
    p.add_argument('--saving_dir', type=str)
    p.add_argument('--scalemae_checkpoint', type=str)

    args = p.parse_args()
    if args.scalemae_checkpoint and not args.checkpoint:
        args.checkpoint = args.scalemae_checkpoint
        print("Warning: --scalemae_checkpoint is deprecated, use --checkpoint instead")
    return args

def _load_yaml(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def _encoder_yaml_path(model_name):
    return f'/scratch2/reimannj/pangaea-bench/configs/encoder/{ENC_YAMLS[model_name]}'

def get_required_bands_from_config(model_name):
    cfg = _load_yaml(_encoder_yaml_path(model_name))
    input_bands = cfg.get('input_bands', {'optical':['B4','B3','B2']})
    if isinstance(input_bands, str) and 'dataset' in input_bands:
        return ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
    if 'optical' in input_bands: return input_bands['optical']
    if 'sar' in input_bands: return input_bands['sar']
    return ['B4','B3','B2']

def _encoder_input_size(model_name):
    cfg = _load_yaml(_encoder_yaml_path(model_name))
    s = cfg.get('input_size', 224)
    return int(s) if isinstance(s, int) else 224  

def convert_pangaea_to_agbd_bands(pangaea_bands):
    out = []
    for b in pangaea_bands:
        if b in ['B8A','B11','B12']: out.append(b)
        elif b.startswith('B') and len(b)==2: out.append(f'B{b[1:].zfill(2)}')
        else: out.append(b)
    return out

def load_input_simple(paths, tile_name, model_name):
    pangaea_bands = get_required_bands_from_config(model_name)
    agbd_bands = convert_pangaea_to_agbd_bands(pangaea_bands)
    print(f"Model {model_name} bands (pangaea): {pangaea_bands}")
    print(f"Loading S2 bands (AGBD filenames): {agbd_bands}")

    if 'composite' in tile_name:
        return load_composite_data(tile_name, paths, agbd_bands)

    s2_prod = tile_name
    transform, upsampling_shape, s2_bands, crs, bounds, boa_offset, lat_cos, lat_sin, lon_cos, lon_sin, meta = process_S2_tile(s2_prod, paths['tiles'])
    selected = {}
    for band in agbd_bands:
        if band in s2_bands:
            selected[band] = (s2_bands[band] - boa_offset*1000)/10000
        else:
            raise ValueError(f"Missing band {band}. Available: {list(s2_bands.keys())}")
    band_data = np.stack([selected[b] for b in agbd_bands], axis=-1)
    data = torch.from_numpy(band_data).to(torch.float)
    mask = np.zeros(data.shape[:2], dtype=bool)
    return data, mask, meta

def load_composite_data(tile_name, paths, required_bands):
    import glob, zipfile, tempfile
    zips = glob.glob(join(paths['tiles'], f"{tile_name}.zip"))
    if not zips: raise FileNotFoundError(f"{tile_name}.zip not in {paths['tiles']}")
    zip_path = zips[0]
    print(f"Loading composite: {zip_path}")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path,'r') as zf: zf.extractall(tmp)
        subdirs = [d for d in os.listdir(tmp) if os.path.isdir(join(tmp,d))]
        if not subdirs: raise ValueError("No composite dir in zip")
        comp_dir = join(tmp, subdirs[0])
        bands_data, ref_meta = {}, None
        for b in required_bands:
            bf = join(comp_dir, f"{b}.tif")
            if not os.path.exists(bf): raise FileNotFoundError(f"Missing {b}.tif in composite")
            with rs.open(bf) as src:
                arr = src.read(1).astype(np.float32)
                bands_data[b] = arr
                if ref_meta is None: ref_meta = src.meta.copy()
        for b in bands_data: bands_data[b] = bands_data[b] / 10000.0
        band_data = np.stack([bands_data[b] for b in required_bands], axis=-1)
        data = torch.from_numpy(band_data).to(torch.float)
        mask = np.zeros(data.shape[:2], dtype=bool)
        return data, mask, ref_meta

def predict_patch(model, patch, device, model_name, tta=False):
    x = torch.from_numpy(patch).to(device).float()  
    agbd_cfg = '/scratch2/reimannj/pangaea-bench/configs/dataset/agbd.yaml'
    agbd = _load_yaml(agbd_cfg)
    means = agbd['data_mean']['optical']; stds = agbd['data_std']['optical']
    loaded_bands = get_required_bands_from_config(model_name)
    agbd_to_p = {'B01':'B1','B02':'B2','B03':'B3','B04':'B4','B05':'B5','B06':'B6',
                 'B07':'B7','B08':'B8','B8A':'B8A','B09':'B9','B11':'B11','B12':'B12'}
    agbd_order_p = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
    idxs = []
    for b in loaded_bands:
        pb = b if b in agbd_order_p else agbd_to_p.get(b,b)
        idxs.append(agbd_order_p.index(pb) if pb in agbd_order_p else 0)
    mean = torch.tensor([means[i] for i in idxs], device=device)
    std  = torch.tensor([stds[i]  for i in idxs], device=device)
    x = (x - mean) / std  

    def _run(inp):
        if model_name in ['prithvi','satlasnet_mi','satlasnet_si']:
            inp = inp.permute(2,0,1).unsqueeze(0).unsqueeze(2)
        else:
            inp = inp.permute(2,0,1).unsqueeze(0)
        input_dict = {'optical': inp} 
        with torch.no_grad():
            y = model.model(input_dict).detach().cpu().numpy()
        return y[0,0,:,:]

    if not tta:
        return _run(x)

    outs = []
    
    for k in range(4):
        xk = torch.rot90(x, k, dims=(0,1))
        yk = _run(xk)
        yk = np.rot90(yk, -k)
        outs.append(yk)
        xkf = torch.flip(xk, dims=(1,))
        ykf = _run(xkf)
        ykf = np.flip(np.rot90(ykf, -k), axis=1)
        outs.append(ykf)
    return np.mean(outs, axis=0)

def _hann2d(h, w, power=2, eps=1e-6):
    if h<=1 or w<=1: return np.ones((h,w), np.float32)
    wy = 0.5*(1-np.cos(2*np.pi*np.arange(h)/(h-1+eps)))
    wx = 0.5*(1-np.cos(2*np.pi*np.arange(w)/(w-1+eps)))
    ww = np.outer(wy, wx).astype(np.float32)
    ww = ww ** power
    ww /= (ww.max()+1e-6)
    return ww

def _grid_positions(total, win, step):
    pos = list(range(0, max(total - win, 0)+1, step))
    if not pos or pos[-1] != total - win:
        pos.append(max(total - win, 0))
    return pos

def _reflect_pad(img, pad_h, pad_w):
    return np.pad(img, ((pad_h,pad_h),(pad_w,pad_w),(0,0)), mode='reflect')

def _unpad(img, pad_h, pad_w):
    return img[pad_h:-pad_h, pad_w:-pad_w, :]

def predict_tile(img, models, patch_size, overlap_size, device, model_name, tta=False):
    H0, W0, _ = img.shape
    ph, pw = patch_size
    oh, ow = overlap_size
    sh, sw = ph - oh, pw - ow

    pad_h = ph//2; pad_w = pw//2
    img_p = _reflect_pad(img, pad_h, pad_w)
    H, W, _ = img_p.shape

    ys = _grid_positions(H - 2*pad_h, ph, sh)
    xs = _grid_positions(W - 2*pad_w, pw, sw)
    ys = [y for y in ys]; xs = [x for x in xs]

    blend = _hann2d(ph, pw, power=2)
    out = np.zeros((len(models), H, W), np.float32)
    wgt = np.zeros((H, W), np.float32)

    print('Sliding over tile...')
    t0 = time.time()
    for yi in ys:
        for xi in xs:
            y0, x0 = yi, xi
            patch = img_p[y0:y0+ph, x0:x0+pw, :]
            for m_idx, m in enumerate(models):
                pred = predict_patch(m, patch, device, model_name, tta=tta)  
                out[m_idx, y0:y0+ph, x0:x0+pw] += pred * blend
                wgt[y0:y0+ph, x0:x0+pw] += blend
    wgt = np.clip(wgt, 1e-6, None)
    out = out / wgt[None, :, :]
    out = out[:, pad_h:pad_h+H0, pad_w:pad_w+W0]
    print(f'Done in {time.time()-t0:.1f}s')
    return out  


class PangaeaInference:
    def __init__(self, model_name, checkpoint_path, device):
        self.model_name = model_name.lower()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.load_model()

    def get_encoder_class_and_params(self):
        cfg_path = _encoder_yaml_path(self.model_name)
        if not os.path.exists(cfg_path): raise FileNotFoundError(cfg_path)
        cfg = _load_yaml(cfg_path)
        target_class = cfg['_target_']; module_name, cls_name = target_class.rsplit('.',1)
        enc_mod = importlib.import_module(module_name)
        enc_cls = getattr(enc_mod, cls_name)
        params = cfg.copy(); params.pop('_target_', None)

        if isinstance(params.get('input_bands'), str) and 'dataset' in str(params['input_bands']):
            agbd_cfg = _load_yaml('/scratch2/reimannj/pangaea-bench/configs/dataset/agbd.yaml')
            params['input_bands'] = agbd_cfg['bands']

        if isinstance(params.get('input_size'), str) and 'dataset' in str(params['input_size']):
            params['input_size'] = 25

        if self.model_name=='prithvi' and 'num_frames' in params:
            if isinstance(params['num_frames'], str) and 'dataset' in params['num_frames']:
                params['num_frames'] = 1

        if 'encoder_weights' in params and params['encoder_weights'] not in [None,'null']:
            if self.model_name=='resnet50_pretrained':
                params['encoder_weights'] = 'IMAGENET1K_V1'
            elif self.model_name=='resnet50_scratch':
                params['encoder_weights'] = None
            else:
                params['encoder_weights'] = "dummy_path.pt"

        params.setdefault('download_url', "")
        
        
        if self.model_name in ['resnet50_pretrained', 'resnet50_scratch']:
            
            params.pop('download_url', None)

        return enc_cls, params

    def load_model(self):
        print(f"Loading {self.model_name} from {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' not in ckpt: raise ValueError("Checkpoint missing 'model'")
        state_dict = ckpt['model']

        
        if self.model_name in ['galileo_base', 'galileo_nano', 'galileo_tiny']:
            print("Applying fix for Galileo state_dict keys...")
            new_state_dict = {}
            
            for key, value in state_dict.items():
                new_key = key.replace(".galileo_encoder.", ".")
                new_state_dict[new_key] = value
            state_dict = new_state_dict 

        enc_cls, enc_params = self.get_encoder_class_and_params()
        encoder = enc_cls(**enc_params)
        from pangaea.decoders.upernet import RegUPerNet
        decoder = RegUPerNet(encoder=encoder, finetune=False, channels=512)
        decoder.load_state_dict(state_dict)
        self.model = decoder.to(self.device).eval()
        print("Model loaded.")

def extract_checkpoint_info(path):
    import re
    d = dirname(path); f = basename(path)
    m = re.search(r'(\d{8}_\d{6}_[a-f0-9]+)_([^_]+)_reg_upernet_agbd', d)
    if m: return f"{m.group(2)}_{m.group(1)}"
    return f.replace('.pth','').replace('checkpoint_','').replace('_best','')

def run_inference():
    args = inf_parser()
    set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset_path=='local':
        data_dir  = args.data_dir  if args.data_dir  else join(os.getcwd(),'S2_L2A')
        out_dir   = args.output_dir if args.output_dir else (args.saving_dir if args.saving_dir else join(os.getcwd(),'results'))
    else:
        data_dir = args.dataset_path
        out_dir  = args.output_dir if args.output_dir else (args.saving_dir if args.saving_dir else join(os.getcwd(),'results'))
    paths = {'tiles': data_dir, 'saving_dir': out_dir}
    os.makedirs(paths['saving_dir'], exist_ok=True)

    primary_model = args.models[0]
    patch_size = tuple(args.patch_size) if args.patch_size else (_encoder_input_size(primary_model), _encoder_input_size(primary_model))
    if args.overlap_size:
        overlap_size = tuple(args.overlap_size)
    else:
        overlap_size = (int(round(patch_size[0]*args.overlap_frac)), int(round(patch_size[1]*args.overlap_frac)))
    print(f"Patch size: {patch_size}, overlap: {overlap_size} (frac={args.overlap_frac:.2f})")

    print(f"Loading data for tile: {args.tile_name}")
    
    
    sar_models = ['ssl4eo_mae_sar', 'croma_joint', 'croma_sar', 'dofa']
    is_sar_model = primary_model in sar_models
    
    try:
        img, mask, meta = load_input_simple(paths, args.tile_name, primary_model)
    except ValueError as e:
        if is_sar_model:
            print(f"Skipping {primary_model} as it requires SAR bands which are not available in the data source. Error: {e}")
            return None
        else:
            raise e

    if img is None:
        print(f"Incompatible model/data for {primary_model}."); return None
    if isinstance(img, torch.Tensor): img = img.numpy()

    if primary_model=='croma_optical':
        patch_size = (120,120); overlap_size = (90,90)  
        print(f"CROMA-specific patch/overlap: {patch_size}/{overlap_size}")

    pred_mask = rescale(mask, 1.0)

    inf_models = []
    for mname in args.models:
        
        if mname in sar_models:
            print(f"Skipping {mname} as it requires SAR data.")
            continue
        inf_models.append(PangaeaInference(mname, args.checkpoint, device))
    
    if not inf_models: 
        print(f"No compatible models to run for this data source.")
        return None

    preds = predict_tile(img, inf_models, patch_size, overlap_size, device, args.models[0], tta=args.tta)
    preds = np.nanmean(preds, axis=0)

    if args.masking:
        preds[pred_mask] = np.nan

    if args.smooth and args.smooth > 0:
        
        preds = gaussian_filter(preds, sigma=args.smooth)

    preds = np.clip(preds, 0, 65535)
    preds[np.isinf(preds)] = 65535
    preds[np.isnan(preds)] = 65535
    preds = preds.astype(np.uint16)

    meta.update(driver='GTiff', dtype=np.uint16, count=1, compress='lzw', nodata=65535)
    model_str = '_'.join(args.models)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = join(paths['saving_dir'], f'{model_str}_{args.tile_name}_{args.year}_{timestamp}.tif')
    print(f'Saving to {out_path}')
    with rs.open(out_path, 'w', **meta) as f:
        f.write(preds, 1)
        f.set_band_description(1, 'AGBD')

    print("Inference completed.")
    ck = extract_checkpoint_info(args.checkpoint)

    if args.visualize:
        try:
            from visualize_results import visualize_agbd_prediction
            if args.viz_output: viz_out = args.viz_output
            else:
                parts = args.tile_name.split('_')
                tile_short = f"{parts[0]}_{parts[1]}_{parts[2]}" if len(parts)>=3 else args.tile_name[:25]
                viz_out = join(paths['saving_dir'], f"{args.models[0]}_{tile_short}_{ck}.png")
            ok = visualize_agbd_prediction(out_path, viz_out, args.tile_name, f"{args.models[0].upper()} ({ck})")
            print("Visualization completed!" if ok else "Visualization failed!")
        except Exception as e:
            print(f"Visualization error: {e}\nRun manually: python visualize_results.py --input {out_path} --tile-name {args.tile_name}")
    else:
        print(f"Visualization skipped. Run: python visualize_results.py --input {out_path} --tile-name {args.tile_name}")
    return out_path

if __name__ == '__main__':
    print("Starting Generic PANGAEA AGBD Inference...")
    result = run_inference()
    if result:
        print(f"\nAll done! Predictions saved to: {result}")
    else:
        print(f"\nInference skipped for the given model.")