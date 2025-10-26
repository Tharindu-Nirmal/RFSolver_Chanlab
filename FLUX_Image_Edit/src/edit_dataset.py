import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
import torch
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image

from flux.sampling import denoise, get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)
from transformers import pipeline
from PIL import Image
import numpy as np

import os
import csv
from pathlib import Path

# metrics
try:
    import lpips
    _LPIPS_AVAILABLE = True
except Exception:
    _LPIPS_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
    _SKIMAGE_AVAILABLE = True
except Exception:
    _SKIMAGE_AVAILABLE = False

# optional general identity (CLIP image-image similarity)
try:
    import open_clip
    _OPENCLIP_AVAILABLE = True
except Exception:
    _OPENCLIP_AVAILABLE = False

# optional face identity (ArcFace via insightface)
try:
    from insightface.app import FaceAnalysis
    _INSIGHT_AVAILABLE = True
except Exception:
    _INSIGHT_AVAILABLE = False

print(f"[metrics] LPIPS={_LPIPS_AVAILABLE}, SKIMAGE={_SKIMAGE_AVAILABLE}, OPENCLIP={_OPENCLIP_AVAILABLE}, INSIGHTFACE={_INSIGHT_AVAILABLE}")
_OPENCLIP_CACHE = {"model": None, "preprocess": None, "device": None}
NSFW_THRESHOLD = 0.85

def _get_openclip(device):
    if not _OPENCLIP_AVAILABLE:
        return None, None
    if _OPENCLIP_CACHE["model"] is None or _OPENCLIP_CACHE["device"] != str(device):
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        model = model.to(device).eval()
        _OPENCLIP_CACHE.update({"model": model, "preprocess": preprocess, "device": str(device)})
    return _OPENCLIP_CACHE["model"], _OPENCLIP_CACHE["preprocess"]

@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    # prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image

def _pil_to_torch_im_01(pil):
    # to CHW float32 in [0,1]
    x = np.asarray(pil).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return x

def _np_to_torch_im_01(np_img):
    x = np_img.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return x

def _to_m1p1(x_01):
    return x_01 * 2.0 - 1.0

def _list_images(root, exts=('jpg','jpeg','png'), recursive=False):
    exts = tuple('.' + e.lower().lstrip('.') for e in exts)
    paths = []
    if recursive:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(exts):
                    paths.append(os.path.join(dirpath, f))
    else:
        for f in sorted(os.listdir(root)):
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return sorted(paths)

def _read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [ln.rstrip('\n') for ln in f]

def _resize_np_image(np_img, size_wh):
    # size_wh = (W, H)
    return np.array(Image.fromarray(np_img).resize(size_wh, resample=Image.BICUBIC))


@torch.inference_mode()
def compute_metrics_bundle(pil_out, np_gt_ref, device, id_metric='clip'):
    """
    pil_out: predicted PIL image (RGB, HxW)      # x̂
    np_gt_ref: ground-truth image as np.uint8 [H,W,3]  # x
    """
    metrics = {
        'lpips': None,
        'ssim': None,
        'psnr': None,
        'id_metric': id_metric,
        'id_sim': None,
    }

    x_out_01 = _pil_to_torch_im_01(pil_out)
    x_gt_01  = _np_to_torch_im_01(np_gt_ref)

    x_out_m1p1 = _to_m1p1(x_out_01).to(device)
    x_gt_m1p1  = _to_m1p1(x_gt_01).to(device)
    
    # --- LPIPS ---
    if _LPIPS_AVAILABLE:
        try:
            lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
            metrics['lpips'] = float(lpips_fn(x_out_m1p1, x_gt_m1p1).item())
        except Exception as e:
            print(f"[metrics][lpips] {type(e).__name__}: {e}")
    
    # --- SSIM / PSNR ---
    if _SKIMAGE_AVAILABLE:
        try:
            out_np = np.asarray(pil_out).astype(np.float32) / 255.0
            gt_np  = np_gt_ref.astype(np.float32) / 255.0
            try:
                ssim_val = ssim(gt_np, out_np, channel_axis=2, data_range=1.0)
            except TypeError:
                ssim_val = ssim(gt_np, out_np, multichannel=True, data_range=1.0)
            metrics['ssim'] = float(ssim_val)
            metrics['psnr'] = float(psnr(gt_np, out_np, data_range=1.0))
        except Exception as e:
            print(f"[metrics][skimage] {type(e).__name__}: {e}")

    # --- Identity similarity (CLIP) ---
    if id_metric == 'clip' and _OPENCLIP_AVAILABLE:
        try:
            model, preprocess = _get_openclip(device)
            if model is not None:
                with torch.no_grad():
                    im1 = preprocess(pil_out).unsqueeze(0).to(device)
                    im2 = preprocess(Image.fromarray(np_gt_ref)).unsqueeze(0).to(device)
                    f1 = model.encode_image(im1).float()
                    f2 = model.encode_image(im2).float()
                    f1 = f1 / (f1.norm(dim=-1, keepdim=True) + 1e-6)
                    f2 = f2 / (f2.norm(dim=-1, keepdim=True) + 1e-6)
                    metrics['id_sim'] = float((f1 @ f2.T).squeeze().item())
        except Exception as e:
            print(f"[metrics][clip] {type(e).__name__}: {e}")


    # --- ArcFace identity (faces only) ---
    if id_metric == 'arcface' and _INSIGHT_AVAILABLE:
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            app = FaceAnalysis(name='buffalo_l', providers=providers)
            app.prepare(ctx_id=0, det_size=(640, 640))
            def _embed(pil):
                arr = np.asarray(pil)
                faces = app.get(arr)
                if not faces: return None
                return faces[0].normed_embedding
            e_out = _embed(pil_out)
            e_in  = _embed(Image.fromarray(np_gt_ref))
            metrics['id_sim'] = float(np.dot(e_out, e_in)) if (e_out is not None and e_in is not None) else None
        except Exception as e:
            print(f"[metrics][arcface] {type(e).__name__}: {e}")

    return metrics

def append_metrics_csv(csv_path, row_dict):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'filename','H','W','seconds',
            'lpips','ssim','psnr','id_metric','id_sim'
        ])
        if write_header:
            w.writeheader()
        w.writerow(row_dict)


@torch.inference_mode()
def main(
    args,
    seed: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    offload: bool = False,
    add_sampling_metadata: bool = True,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
    """
    torch.set_grad_enabled(False)
    name = args.name
    source_prompt = args.source_prompt
    target_prompt = args.target_prompt
    guidance = args.guidance
    output_dir = args.output_dir
    num_steps = args.num_steps
    offload = args.offload
    degradation_type = args.degradation

    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 25

    # init all components
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.encoder.to(torch_device)
    
    if args.input_dir is None:
        degraded_paths = [args.source_img_dir]
        gt_paths = [None]   # will skip metrics unless you pass gt via batch args
        src_prompts = [args.source_prompt]
        tgt_prompts = [args.target_prompt]
    else:
        # ---------- BATCH MODE ----------
        if not (args.gt_dir and args.source_prompts_file and args.target_prompts_file):
            raise ValueError("Batch mode needs --gt_dir, --source_prompts_file, and --target_prompts_file.")

        exts = tuple(e.strip().lower() for e in args.exts.split(',') if e.strip())
        degraded_paths = _list_images(args.input_dir, exts=exts, recursive=args.recursive)
        gt_paths       = _list_images(args.gt_dir,     exts=exts, recursive=args.recursive)

        if len(degraded_paths) == 0:
            raise ValueError(f"No images found in {args.input_dir} with exts={exts}")
        if len(gt_paths) == 0:
            raise ValueError(f"No images found in {args.gt_dir} with exts={exts}")

        # "in order" → strict zip by sorted order
        if len(gt_paths) != len(degraded_paths):
            print(f"[warn] #GT ({len(gt_paths)}) != #degraded ({len(degraded_paths)}). Will zip by order and truncate to min length.")
            n = min(len(gt_paths), len(degraded_paths))
            degraded_paths = degraded_paths[:n]
            gt_paths = gt_paths[:n]

        src_prompts = _read_lines(args.source_prompts_file)
        tgt_prompts = _read_lines(args.target_prompts_file)
        if len(src_prompts) < len(degraded_paths) or len(tgt_prompts) < len(degraded_paths):
            raise ValueError("Prompt files have fewer lines than images. Ensure one line per image, in order.")

    # Prepare NSFW and ensure output folder exists
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Main loop over images (single or batch)
    for idx_img, (src_path, src_prompt_i, tgt_prompt_i, gt_path) in enumerate(zip(degraded_paths, src_prompts, tgt_prompts, gt_paths)):
        print(f"\n[{idx_img+1}/{len(degraded_paths)}] {src_path}")
        init_image_np = np.array(Image.open(src_path).convert('RGB'))
        H0, W0, _ = init_image_np.shape
        new_h = H0 if H0 % 16 == 0 else H0 - (H0 % 16)
        new_w = W0 if W0 % 16 == 0 else W0 - (W0 % 16)
        init_image_np = init_image_np[:new_h, :new_w, :]

        # Ground truth (for metrics). If none (single-image w/o gt), metrics against GT will be skipped.
        np_gt_for_metrics = None
        if gt_path is not None:
            gt_np = np.array(Image.open(gt_path).convert('RGB'))
            # size-match for metrics
            if args.metrics_resize == 'gt_to_pred':
                # we'll resize later to pred size (we only know pred size after decode); store GT now
                np_gt_for_metrics = gt_np
            elif args.metrics_resize == 'pred_to_gt':
                # we will resize prediction later to GT size (requires care); simplest is gt_to_pred
                np_gt_for_metrics = gt_np
            else:
                # 'none': will try to crop to min common size after prediction
                np_gt_for_metrics = gt_np

        height, width, _ = init_image_np.shape

        # ----- build y (degraded tensor) -----
        scale_h = 4; scale_w = 4
        if height >= width:
            scale_h, scale_w = scale_w, scale_h

        y = torch.from_numpy(init_image_np).float().unsqueeze(0) / 127.5 - 1
        B,H,W,C = y.shape
        if degradation_type == "super resolution":
            assert H % scale_h == 0 and W % scale_w == 0
            y = rearrange(y, 'b (h s1) (w s2) c-> b c h w s1 s2', s1=scale_h, s2=scale_w).mean(dim=(-1, -2))
        else:
            y = rearrange(y, 'b h w c-> b c h w')
        y = y.to(torch_device)

        print('y shape:', y.shape)
        print(f"y Tensor range: min={y.min().item():.4f}, max={y.max().item():.4f}")
        
        print('init_image to flux shape:',init_image_np.shape) # [512,512,3]

        # ----- encode input for inversion -----
        init_latent = encode(init_image_np, torch_device, ae)  # [1, C, H/patch, W/patch]

        # ----- per-image options -----
        opts = SamplingOptions(
            source_prompt=src_prompt_i,
            target_prompt=tgt_prompt_i,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        # ----- forward (invert + denoise) -----
        if offload:
            t5, clip = t5.to(torch_device), clip.to(torch_device)
            

        info = dict(feature_path=args.feature_path, feature={}, inject_step=args.inject, ddnm_step=args.ddnm_inject)
        os.makedirs(args.feature_path, exist_ok=True)

        inp = prepare(t5, clip, init_latent, prompt=opts.source_prompt)
        inp_target = prepare(t5, clip, init_latent, prompt=opts.target_prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        z = torch.randn_like(inp["img"])
        z, info = denoise(model, **inp, timesteps=timesteps, y=y, z=z,
                          degradation_type=degradation_type, width=width, height=height,
                          guidance=1, inverse=True, info=info)

        inp_target["img"] = z
        timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))
        x, _ = denoise(model, **inp_target, timesteps=timesteps, y=y, z=z,
                       degradation_type=degradation_type, width=opts.width, height=opts.height,
                       guidance=guidance, inverse=False, info=info)

        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # ----- decode & save -----
        t0 = time.perf_counter()
        batch_x = unpack(x.float(), opts.height, opts.width)
        for x_dec in batch_x:
            x_dec = x_dec.unsqueeze(0)
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                x_dec = ae.decode(x_dec)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.perf_counter()

            x_dec = x_dec.clamp(-1, 1)
            x_dec = embed_watermark(x_dec.float())
            x_dec = rearrange(x_dec[0], "c h w -> h w c")
            img = Image.fromarray((127.5 * (x_dec + 1.0)).cpu().byte().numpy())

            # ----- metrics vs GT (NOT vs degraded) -----
            metrics = None
            if args.compute_metrics and (np_gt_for_metrics is not None):
                gt_for_metrics = np_gt_for_metrics
                print("[metrics] computing for", Path(src_path).name, "GT:", gt_path is not None)
                # size-match according to flag
                if args.metrics_resize == 'gt_to_pred':
                    gt_for_metrics = _resize_np_image(gt_for_metrics, img.size)  # (W,H)
                elif args.metrics_resize == 'none':
                    # crop to min common rectangle
                    h_m = min(img.height, gt_for_metrics.shape[0])
                    w_m = min(img.width,  gt_for_metrics.shape[1])
                    gt_for_metrics = gt_for_metrics[:h_m, :w_m, :]
                    img = img.crop((0,0,w_m,h_m))
                # (pred_to_gt is not recommended here; requires resizing pred tensor before PIL)
                try:
                    metrics = compute_metrics_bundle(
                        pil_out=img,
                        np_gt_ref=gt_for_metrics,
                        device=torch_device,
                        id_metric=args.id_metric
                    )
                    print(f"[metrics]{Path(src_path).name}: {metrics}")
                except Exception as e:
                    print(f"[metrics] error: {e}")

            # ----- nsfw filter & write -----
            nsfw_score = [d["score"] for d in nsfw_classifier(img) if d["label"] == "nsfw"][0]
            stem = Path(src_path).stem
            out_name = os.path.join(output_dir, f"{stem}_edit.jpg")

            if nsfw_score < NSFW_THRESHOLD:
                exif_data = Image.Exif()
                exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                exif_data[ExifTags.Base.Model] = name
                if add_sampling_metadata:
                    exif_data[ExifTags.Base.ImageDescription] = src_prompt_i
                img.save(out_name, exif=exif_data, quality=95, subsampling=0)
                print(f"Saved: {out_name}")
            else:
                print("NSFW flagged. Skipped saving.")

            # ----- append to dataset metrics CSV -----
            if args.compute_metrics and (np_gt_for_metrics is not None):
                dataset_dir = Path(output_dir)
                csv_path = Path(args.metrics_csv) if args.metrics_csv else (dataset_dir / "metrics.csv")
                row = {
                    'filename': Path(src_path).name,
                    'H': img.height,
                    'W': img.width,
                    'seconds': round(t1 - t0, 3),
                    'lpips': None if metrics is None else metrics['lpips'],
                    'ssim': None if metrics is None else metrics['ssim'],
                    'psnr': None if metrics is None else metrics['psnr'],
                    'id_metric': None if metrics is None else metrics['id_metric'],
                    'id_sim': None if metrics is None else metrics['id_sim'],
                }
                append_metrics_csv(csv_path, row)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RF-Edit')

    parser.add_argument('--name', default='flux-dev', type=str,
                        help='flux model')
    parser.add_argument('--source_img_dir', default='', type=str,
                        help='The path of the source image')
    parser.add_argument('--source_prompt', type=str,
                        help='describe the content of the source image (or leaves it as null)')
    parser.add_argument('--target_prompt', type=str,
                        help='describe the requirement of editing')
    parser.add_argument('--feature_path', type=str, default='feature',
                        help='the path to save the feature ')
    parser.add_argument('--guidance', type=float, default=5,
                        help='guidance scale')
    parser.add_argument('--num_steps', type=int, default=25,
                        help='the number of timesteps for inversion and denoising')
    parser.add_argument('--inject', type=int, default=20,
                        help='the number of timesteps which apply the feature sharing')
    parser.add_argument('--ddnm_inject', type=int, default=2,
                        help='the number of timesteps which apply the ddnm update')
    parser.add_argument('--output_dir', default='output', type=str,
                        help='the path of the edited image')
    parser.add_argument('--offload', action='store_true', help='set it to True if the memory of GPU is not enough')
    parser.add_argument('--degradation', type=str, default='super resolution',
                        help='degradation mode: super resolution, colorization, old photo restoration, inpainting')
    
    parser.add_argument('--compute_metrics', action='store_true',
                    help='Compute LPIPS/SSIM/PSNR and identity similarity vs input image.')
    parser.add_argument('--metrics_csv', type=str, default=None,
                        help='Path to dataset-level CSV file to append metrics. '
                            'Defaults to <dataset_dir>/metrics.csv')

    # identity choice (general = CLIP, faces = arcface)
    parser.add_argument('--id_metric', type=str, default='clip', choices=['none', 'clip', 'arcface'],
                        help='Identity metric to compute in addition to LPIPS/SSIM/PSNR.')
    
    # Batch mode arguments
    parser.add_argument('--input_dir', type=str, default=None,
                    help='Folder of degraded images (e.g., .../color/). If set, batch mode is used.')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Folder of ground-truth RGB images (same order as input_dir).')
    parser.add_argument('--source_prompts_file', type=str, default=None,
                        help='Path to source_prompts.txt (one line per image, in order).')
    parser.add_argument('--target_prompts_file', type=str, default=None,
                        help='Path to target_prompts.txt (one line per image, in order).')
    parser.add_argument('--recursive', action='store_true',
                        help='If set, search input_dir and gt_dir recursively.')
    parser.add_argument('--exts', type=str, default='jpg,jpeg,png',
                        help='Comma-separated list of image extensions to include.')
    parser.add_argument('--metrics_resize', type=str, default='gt_to_pred',
                        choices=['gt_to_pred','pred_to_gt','none'],
                        help='How to size-match images for metrics.')



    args = parser.parse_args()

    main(args)
