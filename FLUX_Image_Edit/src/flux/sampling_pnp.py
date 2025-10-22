import math
from typing import Dict, Callable
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder
from .util import load_ae
from PIL import Image, ImageDraw, ImageFont

def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

# --- helpers for debugging---
def tensor_chw_neg1to1_to_pil(x_chw: torch.Tensor) -> Image.Image:
    """
    x_chw: torch.Tensor with shape (C,H,W), values in [-1,1]
    returns: PIL RGB image
    """
    x = x_chw.detach().float().clamp(-1, 1)
    x = (x + 1.0) * 127.5  # to [0,255]
    x = x.round().clamp(0, 255).to(torch.uint8)
    x = rearrange(x, "c h w -> h w c").cpu().numpy()
    return Image.fromarray(x, mode="RGB")

def save_image_grid_with_labels(images, labels, out_path, cols=6, pad=8, caption_h=22, bg=(255,255,255)):
    """
    images: list of PIL Images (all same size)
    labels: list of strings (same length as images)
    cols:   number of columns in the grid
    pad:    padding between tiles (px)
    caption_h: reserved height under each tile for the label
    """
    assert len(images) == len(labels) and len(images) > 0
    w, h = images[0].size
    n = len(images)
    rows = math.ceil(n / cols)

    grid_w = pad + cols*(w + pad)
    grid_h = pad + rows*(h + caption_h + pad)

    canvas = Image.new("RGB", (grid_w, grid_h), bg)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    for idx, (im, text) in enumerate(zip(images, labels)):
        r = idx // cols
        c = idx % cols
        x0 = pad + c*(w + pad)
        y0 = pad + r*(h + caption_h + pad)

        canvas.paste(im, (x0, y0))

        # textbbox gives (left, top, right, bottom)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        tx = x0 + (w - tw)//2
        ty = y0 + h + (caption_h - th)//2
        draw.text((tx, ty), text, fill=(0,0,0), font=font)

    canvas.save(out_path, quality=95, subsampling=0)

# ----------------- helpers (same layout as your sampling.py) -----------------

def unpack(x: Tensor, H: int, W: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(H / 16), w=math.ceil(W / 16), ph=2, pw=2
    )

def repack(x_latent: Tensor) -> Tensor:
    return rearrange(x_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

# ----------------- data-consistency gradient steps in **pixel** space --------

def A_down(x: Tensor, s: int = 4) -> Tensor:
    return F.interpolate(x, scale_factor=1/s, mode="bicubic", align_corners=False, antialias=True)

def AT_up(z: Tensor, s: int = 4) -> Tensor:
    return F.interpolate(z, scale_factor=s, mode="bicubic", align_corners=False, antialias=True)

def PatchUpsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    n, c, h, w = x.shape
    x = x.view(n, c, h, 1, w, 1).expand(-1, -1, -1, scale, -1, scale)
    return x.contiguous().view(n, c, h * scale, w * scale)

def grad_step(x: Tensor, y: Tensor, degradation_type: str, info: Dict, gamma: float) -> Tensor:
    """one gradient step: z = x - gamma * ∇F(x) in pixel space."""
    typ = degradation_type.lower()

    if typ.startswith("super"):                    # 4x SR example
        s = int(info.get("sr_scale", 4))
        H, W = x.shape[-2:]
        assert H % s == 0 and W % s == 0, "H and W must be divisible by sr_scale"

        # Forward operator A: adaptive average pooling to (H/s, W/s)
        y_hat = F.adaptive_avg_pool2d(x, (H // s, W // s))  # A(x)

        # Residual in LR space
        resid = y_hat - y

        # Adjoint A^T for avg-pooling: replicate each LR residual to its s×s patch
        # NOTE: mathematically A^T includes a 1/s^2 factor because A averages.
        # We include it here; you can absorb it into gamma if you prefer.
        g = PatchUpsample(resid, s) #/ (s * s)

        return x - gamma * g

    if "inpaint" in typ:
        # F(x)=0.5||M x - y||^2 => ∇F = M(Mx - y) = Mx - y on known pixels
        M = info["mask"]  # (B,1,H,W) with 1 on known pixels
        g = M * (x - y)
        return x - gamma * g

    if "denois" in typ:
        # F(x)=0.5||x - y||^2 => ∇F = x - y
        g = x - y
        return x - gamma * g

    if "blur" in typ:
        # F(x)=0.5||k*x - y||^2 => ∇F = k^T (k*x - y)
        k = info["kernel"]       # (1,1,kh,kw) or (C,1,kh,kw)
        def conv2(x,k): return F.conv2d(x, k, padding="same", groups=x.shape[1])
        resid = conv2(x, k) - y
        g = conv2(resid, torch.flip(k, dims=[-1,-2]))
        return x - gamma * g

    # default: no-op
    return x

# ----------------- main PnP-FBS loop (3 steps) -------------------------------

@torch.no_grad()
def denoise_pnp_fbs(
    model: Flux,
    ae,
    # model inputs (tokens) at current iterate
    img: Tensor, img_ids: Tensor, txt: Tensor, txt_ids: Tensor, vec: Tensor,
    # schedule + IO
    timesteps: list[float],
    y: Tensor,                 # measurement in **pixel** space (LR for SR, masked image for inpaint, noisy y for denoise)
    z_extra: Tensor,           # unused here; kept for API parity with your sampling.py
    degradation_type: str,
    width: int, height: int,
    inverse: bool,
    info: Dict,
    name: str = "flux-dev",
    offload: bool = False,
    guidance: float = 4.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Implements the 3-step PnP-Flow (FBS) iteration at each t:
      1) Gradient step:          z = x - gamma * ∇F(x)
      2) Interpolation (reproj): z~ = (1 - t) * eps + t * z,  eps ~ P0
      3) PnP denoise:            x_{next} = D_t(z~)  with D_t built from Flux (rectified-flow)
    Everything denoise-facing is run in the **token** space that Flux expects.
    """
    assert not inverse, "PnP-FBS is intended for noise→image (inverse=False)."

    torch_device = torch.device(device)
    ae = load_ae(name, device="cpu" if offload else torch_device)
    # ensure AE is on the same device and fp32
    ae.encoder.to(torch_device)
    ae.decoder.to(torch_device)
    ae.to(dtype=torch.float32)

    # knobs (safe defaults; tune per task)
    gamma = float(info.get("gamma", 0.8))          # gradient step size
    eta_dn = float(info.get("eta_dn", 1.0))        # denoise strength multiplier
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=torch.float32)
    frames, labels = [], []

    # iterate over t (use the same schedule you already pass to sampling.py)
    for i, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        # ===== 1) GRADIENT STEP in pixel space =====
        # decode current tokens -> pixels in [-1,1]
        x_latent = unpack(img, height, width)                       # (B,C,H,W) latent
        x = ae.decode(x_latent.float()).clamp(-1, 1)                    # pixel

        z = grad_step(x, y, degradation_type, info, gamma=gamma)
        # z = x    # pixel

        # ===== 2) INTERPOLATION / REPROJECTION (in token space) =====
        # encode z back to token space, then mix with noise from P0
        z_lat = ae.encode(z.float())                                    # latent
        z_tok = repack(z_lat)                                       # tokens (B, HW, Cpatch)

        # sample eps ~ P0 in **token** space (matches Flux training)
        eps_tok = torch.randn_like(z_tok)                           # standard normal
        t = torch.as_tensor(t_curr, device=z_tok.device, dtype=z_tok.dtype)
        z_tilde_tok = (t) * eps_tok + (1.0-t) * z_tok               # reprojection to path X_t
        # z_tilde_tok = z_tok  # relaxed reprojection

        # ===== 3) PnP DENOISE via Flux velocity at time t =====
        # call Flux once to get velocity v(z~, t); then D_t(z~) ≈ z~ + η (1 - t) v(z~, t)
        t_vec = torch.full((img.shape[0],), t_curr, dtype=torch.float32, device=img.device)
        info['t'] = t_curr
        info['second_order'] = False

        # Ensure inputs match the model parameter dtype (bf16 in your case)
        model_dtype = next(model.parameters()).dtype

        z_tilde_tok_m = z_tilde_tok.to(model_dtype)
        txt_m        = txt.to(model_dtype)
        vec_m        = vec.to(model_dtype)
        img_ids_m    = img_ids.to(model_dtype)
        txt_ids_m    = txt_ids.to(model_dtype)
        guidance_vec_m = guidance_vec.to(model_dtype)
        t_vec        = t_vec.to(model_dtype)

        v_tok, info = model(
            img=z_tilde_tok_m,
            img_ids=img_ids_m,
            txt=txt_m,
            txt_ids=txt_ids_m,
            y=vec_m,                       # CLIP vec
            timesteps=t_vec,
            guidance=guidance_vec_m,
            info=info
        )

        img = z_tilde_tok + eta_dn * (1.0 - t) * v_tok              # tokens for next iterate

        img_debug = unpack(img, height, width) #[B,C,H,W]
        img_debug = ae.decode(img_debug.float())   #[B,C,H,W], ~[-1,1]
        
        # bring into PIL format and save
        x_debug = tensor_chw_neg1to1_to_pil(img_debug[0])
        if (not inverse):
            # save the the images from noise-> img path separately
            x_debug.save('test_itr%.2d.png'%(i), quality=95, subsampling=0)
        frames.append(x_debug)
        labels.append(f"itr{i}_t={info['t']}")
    
    # After the loop, save one grid image with captions
    save_image_grid_with_labels(frames, labels,out_path=f"debug_grid_{'img2noise' if inverse else 'noise2img'}.png",
                                cols=6, pad=8, caption_h=22)

    return img, info
