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

NSFW_THRESHOLD = 0.85

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
    
    init_image = None
    init_image = np.array(Image.open(args.source_img_dir).convert('RGB'))
    
    shape = init_image.shape

    new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
    new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

    init_image = init_image[:new_h, :new_w, :]
    h, w, c = init_image.shape

    #====================edits start==============
    # Scales for average pooling
    print('init_image shape:',init_image.shape) # [320,480,3]
    scale_h = 4
    scale_w = 4
    if h >= w:
        scale_h,scale_w = scale_w,scale_h # swap if height is greater than width

    #Creating y tensor to repeatedly be used in sampling.py-->ddnm_simple
    y = torch.from_numpy(init_image).float().unsqueeze(0) # [1,320,480,3]
    y = rearrange(y, 'b h w c-> b c h w') # [1,3,320,480]
    y = y.to(torch_device)
    print('y shape:', y.shape)
    print(f"y Tensor range: min={y.min().item():.4f}, max={y.max().item():.4f}")

    #naive upsample the degraded image
    init_image = np.repeat( np.repeat(init_image, scale_h, axis=0), scale_w, axis=1)
    height, width = init_image.shape[0], init_image.shape[1]
    
    print('init_image to flux shape:',init_image.shape) # [4x320,4x480,3]


    #Convert the numpy image array into latent space tensor. init_image is used both in inversion and reconstruction
    init_image = encode(init_image, torch_device, ae) # 

    #Creating a copy y_enc tensor to repeatedly be used in inversion. sampling.py-->ddnm_simple
    y_enc = init_image # [1, 16, ?, ?]
    B,C,H,W = y_enc.shape
    assert H % scale_h == 0 and W % scale_w == 0 #Height and Width must be divisible by scale

    #Average pooling
    y_enc = rearrange(y_enc, 'b c (h s1) (w s2) -> b c h w s1 s2', s1=scale_h, s2=scale_w)
    y_enc = y_enc.mean(dim=(-1, -2))  # [1, 16, ?, ?]
    #didnt rescale the image, or move y_enc to torch_deivce, because enode already does it.
    print('y_enc shape:', y_enc.shape)
    print(f"y_enc Tensor range: min={y_enc.min().item():.4f}, max={y_enc.max().item():.4f}")
    

    #=======================edits end===============

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    if loop:
        opts = parse_prompt(opts)

    while opts is not None:
        if opts.seed is None:
            opts.seed = torch.randint(0, 2**32 - 1, (1,)).item()  # Generate a random seed
            torch.manual_seed(opts.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(opts.seed)
        print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")
        t0 = time.perf_counter()

        # opts.seed = None
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)

        info = {}
        info['feature_path'] = args.feature_path
        info['feature'] = {}
        info['inject_step'] = args.inject
        info['ddnm_step'] = args.ddnm_inject
        if not os.path.exists(args.feature_path):
            os.mkdir(args.feature_path)

        # prepare the input tensors for the VIT model by patching and changing shape
        inp = prepare(t5, clip, init_image, prompt=opts.source_prompt)
        
        inp_target = prepare(t5, clip, init_image, prompt=opts.target_prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        # offload TEs to CPU, load model to gpu
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # inversion to go from image latent to initial noise latent
        z, info = denoise(model, **inp, timesteps=timesteps, y_enc=y_enc, y=y, width=width, height=height, guidance=1, inverse=True, info=info)
        
        inp_target["img"] = z

        timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

        # denoise initial noise
        x, _ = denoise(model, **inp_target, timesteps=timesteps, y_enc=y_enc, y=y, width=opts.width, height=opts.height, guidance=guidance, inverse=False, info=info)
        
        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # Decode latents to pixel space.  
        # Look at the unpack function in flux/sampling.py. Previously they had switched width and height for some reason.
        batch_x = unpack(x.float(), opts.height, opts.width)

        for x in batch_x:
            x = x.unsqueeze(0)
            output_name = os.path.join(output_dir, "imgwddnm_v_{idx}.jpg")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                idx = 0
            else:
                fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"imgwddnm_v_[0-9]+\.jpg$", fn)]
                if len(fns) > 0:
                    idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
                else:
                    idx = 0

            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                x = ae.decode(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            fn = output_name.format(idx=idx)
            print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
            # bring into PIL format and save
            x = x.clamp(-1, 1)
            x = embed_watermark(x.float())
            x = rearrange(x[0], "c h w -> h w c")

            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
            nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]
            
            if nsfw_score < NSFW_THRESHOLD:
                exif_data = Image.Exif()
                exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                exif_data[ExifTags.Base.Model] = name
                if add_sampling_metadata:
                    exif_data[ExifTags.Base.ImageDescription] = source_prompt
                img.save(fn, exif=exif_data, quality=95, subsampling=0)
                idx += 1
            else:
                print("Your generated image may contain NSFW content.")

            if loop:
                print("-" * 80)
                opts = parse_prompt(opts)
            else:
                opts = None

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

    args = parser.parse_args()

    main(args)
