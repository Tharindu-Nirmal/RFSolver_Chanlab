import os
from PIL import Image
import math
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ------------- User Settings -------------------
data_folder = "/scratch/gilbreth/lwickrem/data/afhq_gt/val/cat_selected_1"       # Your input image folder
output_folder = "/scratch/gilbreth/lwickrem/data/afhq_degrads/deblur/cat_selected_1" # Where degraded images will be saved

# Select degradation mode to create data
IR_mode = "denoising"  # Options: "colorization", "inpainting", "super resolution", "super resolution embeds", "denoising", "deblurring", "old photo restoration"

# Used for super resolution
scale = 4                
scale_h = 4
scale_w = 4

# Used when IR_mode == "deblurring"
blur_sigma   = 2      # std dev of Gaussian PSF (in pixels)
kernel_size  = 11       # odd number, e.g., 11/15/21
wiener_lambda = 1e-3    # Tikhonov/Wiener regularizer for pinv(A); try 1e-4 .. 1e-2

# Used when IR_mode == "denoising"
sigma = 5             # noise level in [0,255] intensity units;

device = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------


# ---------- Degradation Operators -------------
#https://github.com/wyhuai/DDNM

def color2gray(x):
# Convert RGB image to grayscale by averaging the channels,but repeat it to maintain 3 channel structure.
    coef = 1/3
    gray = x[:, 0, :, :] * coef + x[:, 1, :, :] * coef + x[:, 2, :, :] * coef
    return gray.unsqueeze(1).repeat(1, 3, 1, 1)

def PatchUpsample(x, scale):
    n, c, h, w = x.shape
    x = x.view(n, c, h, 1, w, 1).expand(-1, -1, -1, scale, -1, scale)
    return x.contiguous().view(n, c, h * scale, w * scale)

def PatchUpsampleEmbeds(x, scale_h, scale_w):
    n, c, h, w = x.shape
    # Embedding dimensions change into [B, Cenc, Henc, Wenc]. So we need new scale_h and scale_w for embeddings.
    x = x.view(n, c, h, 1, w, 1).expand(-1, -1, -1, scale_h, -1, scale_w)
    return x.contiguous().view(n, c, h * scale_h, w * scale_w)

def gray2color(x):
    # Input shape: [N, 3, H, W] where all 3 channels are the same
    # First reduce to single channel
    x = x[:, 0, :, :]  # shape: [N, H, W]
    
    coef = 1/3
    # Reverse the averaging operation by scaling up
    base = coef**2 + coef**2 + coef**2  # sum of squared coefficients
    restored = x * coef / base  # scale back to plausible RGB value
    
    # Stack it back to 3 channels
    return torch.stack([restored, restored, restored], dim=1)  # shape: [N, 3, H, W]

def gaussian_2d_kernel(sigma: float, kernel_size: int, device="cpu", dtype=torch.float32):
    """
    Returns a (kernel_size x kernel_size) normalized 2D Gaussian kernel.
    kernel_size must be odd.
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd."
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k = k / k.sum()
    return k  # shape [K, K]

def _fft_filter_from_kernel(k2d: torch.Tensor, h: int, w: int, c: int, device, dtype):
    """
    Center the small kernel into an (h,w) canvas (per-channel), roll to align,
    and return its FFT. Output shape: [1, c, h, w] in frequency domain.
    """
    ks = k2d.shape[-1]
    filt = torch.zeros((1, c, h, w), device=device, dtype=dtype)
    filt[..., :ks, :ks] = k2d.to(device=device, dtype=dtype)
    filt = torch.roll(filt, shifts=(-(ks - 1)//2, -(ks - 1)//2), dims=(2, 3))
    return torch.fft.fft2(filt)
    

def set_operator(img_shape, IR_mode):
    if IR_mode == "colorization":
        _, _, h, w = img_shape
        return color2gray

    elif IR_mode == "inpainting":
        _, _, h, w = img_shape
        mask = torch.ones(img_shape, device=device)
        mask[:, :, h//4:h*3//4, w//4:w*3//4] = 0
        return lambda z: z * mask

    elif IR_mode == "super resolution":
        _, _, h, w = img_shape
        down = torch.nn.AdaptiveAvgPool2d((h // scale, w // scale))
        # #maintain input image size
        # up = torch.nn.Upsample(size=(h, w), mode="bilinear", align_corners=False)
        return lambda z: down(z)
    
    if IR_mode == "super resolution embeds":
        _, _, h, w = img_shape
        #  Embedding dimensions change into [B, Cenc, Henc, Wenc]. So we need new scale_h and scale_w for embeddings.
        down = torch.nn.AdaptiveAvgPool2d((h // scale_h, w // scale_w))
        return lambda z: down(z)
    
    # ---- NEW: Denoising ----
    elif IR_mode == "denoising":
        # Identity measurement: y = x
        return lambda z: z

    # ---- NEW: Deblurring (circular conv via FFT) ----
    elif IR_mode == "deblurring":
        # Build the blur PSF in frequency once for these image dims
        _, c, h, w = img_shape
        k2d = gaussian_2d_kernel(blur_sigma, kernel_size, device="cpu", dtype=torch.float32)

        # Create Ĥ(ω) on-the-fly using the *call-site* device/dtype for safety
        def A_blur(z: torch.Tensor):
            H_hat = _fft_filter_from_kernel(k2d, z.shape[-2], z.shape[-1], z.shape[1],
                                            device=z.device, dtype=z.dtype)
            return torch.fft.ifft2(torch.fft.fft2(z) * H_hat).real
        return A_blur

    elif IR_mode == "old photo restoration":
        _, _, h, w = img_shape
        mask = torch.ones(img_shape, device=device)
        mask[:, :, h//4:h*3//4, w//4:w*3//4] = 0
        A1 = lambda z: z * mask
        A2 = color2gray
        A3 = torch.nn.AdaptiveAvgPool2d((h // scale, w // scale))
        return lambda z: A3(A2(A1(z)))

    else:
        raise ValueError(f"Unknown IR_mode: {IR_mode}")
    

def set_pinv_operator(img_shape, IR_mode):
    _, _, h, w = img_shape
    if IR_mode == "colorization":
        return gray2color

    elif IR_mode == "inpainting":
        mask = torch.ones(img_shape, device=device)
        mask[:, :, h//4:h*3//4, w//4:w*3//4] = 0
        return lambda z: z * mask

    elif IR_mode == "super resolution":
        # The input to Ap is the downsampled image, so we upsample with PatchUpsample
        return lambda z: PatchUpsample(z, scale)
    
    elif IR_mode == "super resolution embeds":
        # The input to Ap is the downsampled image, so we upsample with PatchUpsample
        return lambda z: PatchUpsampleEmbeds(z, scale_h, scale_w)
    
    # ---- NEW: Denoising ----
    elif IR_mode == "denoising":
        # Pseudoinverse of Identity is Identity (both left and right inverse)
        return lambda z: z

    # ---- NEW: Deblurring pinv(A) via Wiener/Tikhonov deconvolution ----
    elif IR_mode == "deblurring":
        # A^\dagger y = F^{-1} { H*(ω) / (|H(ω)|^2 + λ) ⊙ F{y} }
        # This is a *regularized right-inverse*: A(A^\dagger y) ≈ y
        _, c, h, w = img_shape
        k2d = gaussian_2d_kernel(blur_sigma, kernel_size, device="cpu", dtype=torch.float32)

        def A_pinv_blur(y: torch.Tensor):
            H_hat = _fft_filter_from_kernel(k2d, y.shape[-2], y.shape[-1], y.shape[1],
                                            device=y.device, dtype=y.dtype)
            Y_hat = torch.fft.fft2(y)
            denom = (H_hat.conj() * H_hat).real + wiener_lambda  # avoid division by small |H|^2
            H_pinv = H_hat.conj() / denom
            x_est = torch.fft.ifft2(Y_hat * H_pinv).real
            return x_est

        return A_pinv_blur

    elif IR_mode == "old photo restoration":
        mask = torch.ones(img_shape, device=device)
        mask[:, :, h//4:h*3//4, w//4:w*3//4] = 0
        A1p = lambda z: z * mask
        A2p = gray2color
        A3p = lambda z: PatchUpsample(z, scale)
        return lambda z: A1p(A2p(A3p(z)))

    else:
        raise ValueError(f"Unknown IR_mode: {IR_mode}")
# -----------------------------------------------



# ------------ DDNM pipeline function to be imported to the edit.py script -----------
def ddnm_simple(xt, y, z, t, lambda_t=1, IR_mode="super resolution"):
# Refer: https://arxiv.org/pdf/2212.00490
# https://github.com/wyhuai/DDNM

    x0t = xt
    # x0t = (xt - (t)*z)/ ((1-t) + 1e-10)

    A = set_operator(x0t.shape, IR_mode)
    Ap = set_pinv_operator(x0t.shape, IR_mode)

    # x0t= x0t + lambda_t*Ap(y - A(x0t))
    # Seaprate lines for debugging:  
    # print(f"DDNM step input: x0t min={x0t.min().item():.4f}, max={x0t.max().item():.4f}; y min={y.min().item():.4f}, max={y.max().item():.4f}")    
    
    y0_hat = A(x0t)
    yres = y - y0_hat
    xres = Ap(yres)
    DDNM_x0t= x0t + lambda_t*xres
    # print(f"DDNM step: y0_hat min={y0_hat.min().item():.4f}, max={y0_hat.max().item():.4f}; yres min={yres.min().item():.4f}, max={yres.max().item():.4f}; xres min={xres.min().item():.4f}, max={xres.max().item():.4f}, x0t min={x0t.min().item():.4f}, max={x0t.max().item():.4f}")
    # x0t = torch.clamp(x0t, 0, 255)

    DDNM_xt = DDNM_x0t
    # DDNM_xt = (t)*z + (1-t)*DDNM_x0t
    
    return DDNM_xt

def ddnm_flow(x0, y, v, t, lambda_t=1, IR_mode="super resolution"):
    """
    v is the velocity at time t, x0 is the estimate for x0 (the clean image estimate given xt)
    """
# Refer: PNP flow https://arxiv.org/pdf/2410.02423

    # Expected value of x0, given xt
    x0t = x0

    A = set_operator(x0t.shape, IR_mode)
    Ap = set_pinv_operator(x0t.shape, IR_mode)

    # x0t= x0t + lambda_t*Ap(y - A(x0t))
    # Seaprate lines for debugging:  
    # print(f"DDNM step input: x0t min={x0t.min().item():.4f}, max={x0t.max().item():.4f}; y min={y.min().item():.4f}, max={y.max().item():.4f}")    
    
    y0_hat = A(x0t)
    yres = y - y0_hat
    xres = Ap(yres)
    DDNM_x0t= x0t + lambda_t*xres
    # print(f"DDNM step: y0_hat min={y0_hat.min().item():.4f}, max={y0_hat.max().item():.4f}; yres min={yres.min().item():.4f}, max={yres.max().item():.4f}; xres min={xres.min().item():.4f}, max={xres.max().item():.4f}, x0t min={x0t.min().item():.4f}, max={x0t.max().item():.4f}")
    # x0t = torch.clamp(x0t, 0, 255)

    # DDNM_xt = DDNM_x0t
    DDNM_xt = DDNM_x0t + t*v
    
    return DDNM_xt



# --------- Image Processing Pipeline to create the degraded images dataset----------
if __name__ == "__main__":
    print(f"Using device: {device} for generating degraded images.")
    print(f"Degradation mode: {IR_mode}")
    os.makedirs(output_folder, exist_ok=True)

    # Extra folder for the original (pre-upsample) y only for SR mode
    original_y_folder = None
    if IR_mode == "super resolution":
        original_y_folder = f"{output_folder}_original_y"
        os.makedirs(original_y_folder, exist_ok=True)

    transform = T.ToTensor()
    image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('jpg', 'png', 'jpeg'))]

    for fname in image_files:
        path = os.path.join(data_folder, fname)
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)  # (1,3,H,W)

        A = set_operator(img_tensor.shape, IR_mode)
        degraded = A(img_tensor)  # this is the measurement y

        # ---- ADD NOISE FOR DENOISING MODE (A = I) ----
        if IR_mode == "denoising":
            # sigma is assumed to be in [0,255] intensity units; convert to [0,1]
            noise_std = torch.as_tensor(sigma, dtype=img_tensor.dtype, device=img_tensor.device) / 255.0
            noise = torch.randn_like(img_tensor) * noise_std
            degraded = torch.clamp(img_tensor + noise, 0.0, 1.0)

        # If super-resolution, also save the *original y* before upsampling
        if IR_mode == "super resolution":
            # save low-res measurement y
            y_lr_img = TF.to_pil_image(torch.clamp(degraded.squeeze(0), 0, 1).cpu())
            y_lr_img.save(os.path.join(original_y_folder, fname))

            # then upsample so FLUX can handle the size
            degraded = PatchUpsample(degraded, scale)

        # Save the (possibly upsampled) degraded image to output_folder
        degraded_img = TF.to_pil_image(torch.clamp(degraded.squeeze(0), 0, 1).cpu())
        degraded_img.save(os.path.join(output_folder, fname))

    if IR_mode == "super resolution":
        print(f"Saved {len(image_files)} LR measurements to {original_y_folder}/")
    print(f"Saved {len(image_files)} degraded images to {output_folder}/")
