# edit_pnp.py
import os
import re
import time
import argparse
from glob import iglob

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ExifTags
from transformers import pipeline

# PnP-Flow sampler + helpers (use the versions you put in sampling_pnp.py)
from flux.sampling_pnp import denoise_pnp_fbs as denoise
from flux.sampling_pnp import get_schedule, prepare, unpack

# model/utility loaders
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5
import numpy as np
NSFW_THRESHOLD = 0.85


@torch.inference_mode()
def encode_np_rgb_to_latents(np_img, ae, device):
    """np_img: HxWx3 uint8 -> latents (B,C,H/8,W/8) in ae space."""
    x = torch.from_numpy(np_img).permute(2, 0, 1).float() / 127.5 - 1.0  # [-1,1]
    x = x.unsqueeze(0).to(device)
    lat = ae.encode(x).to(torch.bfloat16)
    return lat


def crop_to_16(np_img):
    h, w, _ = np_img.shape
    h2 = h - (h % 16)
    w2 = w - (w % 16)
    return np_img[:h2, :w2, :]


@torch.inference_mode()
def main(args):
    """
    PnP-Flow baseline runner:
      - builds measurement y in pixel space (supports 4x SR by default)
      - starts from token noise and runs denoise_pnp_fbs (3-step FBS)
      - decodes and saves result
    """
    # -------- device / seed ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    # -------- model selection ----------
    name = args.name
    if name not in configs:
        raise ValueError(f"Unknown model name: {name}. Choose from: {', '.join(configs.keys())}")

    # -------- load components ----------
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if args.offload else torch_device)
    ae = load_ae(name, device="cpu" if args.offload else torch_device)

    if args.offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.encoder.to(torch_device)  # keep encoder ready
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=0 if device=="cuda" else -1)

    # -------- read & prep image ----------
    np_img = crop_to_16(np.array(Image.open(args.source_img_dir).convert("RGB")))
    H, W, _ = np_img.shape
    print(f"input image (cropped to x16): {H}x{W}")

    # latents for shape/conditioning only (we start from noise for PnP)
    init_lat = encode_np_rgb_to_latents(np_img, ae, torch_device)

    # -------- measurement y (pixel space) ----------
    # Currently implemented: 4× super-resolution (bicubic downsample)
    y = torch.from_numpy(np_img).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0  # (1,3,H,W) in [-1,1]
    if args.degradation.lower().startswith("super"):
        H, W = y.shape[-2:]
        assert H % args.sr_scale == 0 and W % args.sr_scale == 0, \
            "Image size must be divisible by --sr_scale for adaptive avg pooling"
        y = F.adaptive_avg_pool2d(y, (H // args.sr_scale, W // args.sr_scale))

    elif "denois" in args.degradation.lower():
        # expecting a noisy input already; y is the noisy observation
        pass
    elif "inpaint" in args.degradation.lower():
        # expect a mask path (white=known). If not provided, fall back to identity.
        if args.mask_path and os.path.exists(args.mask_path):
            mask = Image.open(args.mask_path).convert("L").resize((W, H), Image.NEAREST)
            mask = torch.from_numpy((np.array(mask) > 127).astype("float32"))[None, None]  # (1,1,H,W)
        else:
            print("Warning: inpainting mask not provided; proceeding without a mask.")
            mask = torch.zeros(1, 1, H, W, dtype=torch.float32)
    elif "blur" in args.degradation.lower():
        print("Note: deblurring expects kernel in sampling_pnp info; not built here.")
    else:
        print("Unknown degradation type; proceeding without specialized DC.")

    y = y.to(torch_device)

    # -------- build conditioning & initial tokens (noise) ----------
    # Use target prompt to steer the prior; keep source prompt available if you reuse features later.
    inp = prepare(t5, clip, init_lat, prompt=args.target_prompt)
    # start from token noise (PnP-Flow noise->image)
    inp["img"] = torch.randn_like(inp["img"])

    # time schedule
    timesteps = get_schedule(args.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

    # -------- offload text encoders if needed ----------
    if args.offload:
        t5, clip = t5.cpu(), clip.cpu()
        torch.cuda.empty_cache()
        model = model.to(torch_device)

    # -------- run PnP-Flow (FBS) ----------
    info = {
        "gamma": args.gamma,      # DC step size
        "eta_dn": args.eta_dn,    # denoiser gain
        "sr_scale": args.sr_scale,
        "inject": False,    
    }
    if args.degradation.lower().startswith("super"):
        info["sr_scale"] = args.sr_scale  # (if you wire this into grad_step)

    if "inpaint" in args.degradation.lower():
        info["mask"] = mask.to(torch_device)

    z_extra = torch.empty(0, device=torch_device)  # API parity placeholder
    print(f"Running PnP-Flow for {args.num_steps} steps…")
    t0 = time.perf_counter()

    if args.offload:
    # We need both during the loop
        ae.encoder.to(torch_device)
        ae.decoder.to(torch_device)
        model = model.to(torch_device)

    x_tokens, _ = denoise(
        model,
        ae,
        img=inp["img"],
        img_ids=inp["img_ids"],
        txt=inp["txt"],
        txt_ids=inp["txt_ids"],
        vec=inp["vec"],
        timesteps=timesteps,
        y=y,
        z_extra=z_extra,
        degradation_type=args.degradation,
        width=W,
        height=H,
        inverse=False,
        info=info,
        name=name,
        offload=args.offload,
        guidance=args.guidance,
        device=device,
    )
    t1 = time.perf_counter()
    print(f"PnP-Flow done in {t1 - t0:.1f}s")

    # -------- decode & save ----------
    if args.offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(torch_device)

    batch_x = unpack(x_tokens.float(), H, W)  # latents -> (B,3,H,W) through ae.decode below
    os.makedirs(args.output_dir, exist_ok=True)
    pattern = os.path.join(args.output_dir, "img_pnp_{idx}.jpg")

    # quick assertion right before decode/encode
    assert next(ae.decoder.parameters()).is_cuda, "AE decoder must be on CUDA"

    for x in batch_x:
        x = x.unsqueeze(0)
        x = ae.decode(x.float())

        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x_hw3 = rearrange(x[0], "c h w -> h w c").cpu().numpy()
        pil = Image.fromarray((127.5 * (x_hw3 + 1.0)).astype("uint8"))

        # nsfw check
        nsfw_score = [d["score"] for d in nsfw_classifier(pil) if d["label"] == "nsfw"][0]
        if nsfw_score >= NSFW_THRESHOLD:
            print("Warning: NSFW score high; skipping save.")
            continue

        # pick index
        fns = [fn for fn in iglob(pattern.format(idx="*")) if re.search(r"img_pnp_[0-9]+\.jpg$", fn)]
        idx = max([int(os.path.basename(fn).split("_")[-1].split(".")[0]) for fn in fns], default=0)

        exif = Image.Exif()
        exif[ExifTags.Base.Software] = "AI generated;PnP-Flow;flux"
        exif[ExifTags.Base.Make] = "Black Forest Labs"
        exif[ExifTags.Base.Model] = name
        exif[ExifTags.Base.ImageDescription] = args.target_prompt or (args.source_prompt or "")

        out_path = pattern.format(idx=idx)
        pil.save(out_path, exif=exif, quality=95, subsampling=0)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PnP-Flow with FLUX (RF) prior")

    parser.add_argument("--name", type=str, default="flux-dev", help="flux model name")
    parser.add_argument("--source_img_dir", type=str, required=True, help="path to the source image")
    parser.add_argument("--source_prompt", type=str, default="", help="(optional) description of source image")
    parser.add_argument("--target_prompt", type=str, default="", help="target/edit prompt (conditioning)")
    parser.add_argument("--output_dir", type=str, default="output_pnp", help="where to save results")

    parser.add_argument("--guidance", type=float, default=2.0, help="CFG/guidance (keep modest for identity)")
    parser.add_argument("--num_steps", type=int, default=30, help="PnP iterations / time steps")

    # PnP-Flow knobs
    parser.add_argument("--gamma", type=float, default=0.8, help="DC gradient step size")
    parser.add_argument("--eta_dn", type=float, default=1.0, help="denoiser gain (multiplies (1 - t))")

    # Degradation options (currently implemented: super resolution, inpainting, denoising)
    parser.add_argument("--degradation", type=str, default="super resolution",
                        help="one of: 'super resolution', 'inpainting', 'denoising', 'deblurring'")
    parser.add_argument("--sr_scale", type=int, default=4, help="SR scale (only used if degradation startswith 'super')")
    parser.add_argument("--mask_path", type=str, default="", help="(inpainting) path to a binary mask image")

    parser.add_argument("--offload", action="store_true", help="CPU-offload big modules if GPU memory is tight")
    parser.add_argument("--seed", type=int, default=None, help="random seed")

    args = parser.parse_args()
    main(args)
