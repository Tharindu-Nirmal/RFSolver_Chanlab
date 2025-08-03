import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder

from .ddnm_degrads import ddnm_simple
from .util import (load_ae)
from PIL import Image


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


# Reorder the function definition to be used in denoise.
def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    y_enc: Tensor,
    y: Tensor,
    width,
    height,
    inverse,
    info,
    name: str = "flux-dev",
    offload: bool = False,
    guidance: float = 4.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu", 
    
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    # edits for ddnm update
    # ddnm_list = [True] * info['ddnm_step'] + [False] * (len(timesteps[:-1]) - info['ddnm_step'])
    # ddnm_list = [False] * (len(timesteps[:-1]) - info['ddnm_step']) + [True] * info['ddnm_step']
    ddnm_list =  [False]*(len(timesteps[:-1]) - 2) + [True]*1 + [False]*1

    torch_device = torch.device(device)
    ae = load_ae(name, device="cpu" if offload else torch_device)
    
    
    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
        ddnm_list = ddnm_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]
        info['ddnm'] = ddnm_list[i]

        #vhat_(ti) in algorithm 1 of the paper
        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        #Z_(ti + delta ti)
        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True

        #vhat_(ti + delta ti) in algorithm 1 of the paper
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )

        #Calculating acceleration (the derivate of velocity).
        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)

        #Second order update for the Latent.
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

        #ddnm update in latent space
        # if info['ddnm']:
        #     img = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=math.ceil(height / 16), w=math.ceil(width / 16), ph=2, pw=2,)
            
        #     # confirming the shape of the input degraded image (y= A img) is the measured version of the image (img).
        #     # print('img shape:',img.shape, 'y_shape:', y.shape)

        #     # Using 4x downsample for y
        #     print(f"img Tensor range: min={img.min().item():.4f}, max={img.max().item():.4f}")
        #     img = ddnm_simple(img, y_enc, lambda_t=0.01, IR_mode="super resolution embeds")
        #     img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        # NEED ALOT OF DEBUGGING HERE. y should be in image space. ==================================================
        #ddnm update in image space
        if info['ddnm']:
            # decode
            img = unpack(img, height, width) #[B,C,H,W]
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                img = ae.decode(img)
            
            # Debugging: Save the image as of this point
            # # bring into PIL format and save
            x = img.clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")
            img_x = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
            img_x.save('test.png', quality=95, subsampling=0)
            # assert 1==0

            print(f"img Tensor range: min={img.min().item():.4f}, max={img.max().item():.4f}")
            print(f"y Tensor range: min={y.min().item():.4f}, max={y.max().item():.4f}")
            img = ddnm_simple(img, y, lambda_t=0.01, IR_mode="super resolution") # both y and img are in [B,C,H,W]

            # The only relevant part from the encode() function
            img = ae.encode(img.to()).to(torch.bfloat16)
            print("encoded img shape:", img.shape)

            # The only relevant part in the prepare() function
            img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    return img, info



