import os
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ------------- User Settings -------------------
data_folder = "/scratch/gilbreth/lwickrem/data/HandpickedDegrads/gt"       # Your input image folder
output_folder = "/scratch/gilbreth/lwickrem/data/HandpickedDegrads/" # Where degraded images will be saved
IR_mode = "super resolution"  # Select degradation mode
scale = 4                  # Used for super resolution
scale_h = 4
scale_w = 4
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
def ddnm_simple(xt, y, z, t, lambda_t=0.1, IR_mode="super resolution"):
# Refer: https://arxiv.org/pdf/2212.00490
# https://github.com/wyhuai/DDNM

    # x0t = xt
    x0t = (xt - (t)*z)/ ((1-t) + 1e-10)

    A = set_operator(x0t.shape, IR_mode)
    Ap = set_pinv_operator(x0t.shape, IR_mode)

    # x0t= x0t + lambda_t*Ap(y - A(x0t))
    # Seaprate lines for debugging:  
    # print(f"DDNM step input: x0t min={x0t.min().item():.4f}, max={x0t.max().item():.4f}; y min={y.min().item():.4f}, max={y.max().item():.4f}")    
    
    y0_hat = A(x0t)
    yres = y - y0_hat
    xres = Ap(yres)
    x0t= x0t + lambda_t*xres
    # print(f"DDNM step: y0_hat min={y0_hat.min().item():.4f}, max={y0_hat.max().item():.4f}; yres min={yres.min().item():.4f}, max={yres.max().item():.4f}; xres min={xres.min().item():.4f}, max={xres.max().item():.4f}, x0t min={x0t.min().item():.4f}, max={x0t.max().item():.4f}")
    # x0t = torch.clamp(x0t, 0, 255)

    # xt = x0t
    xt = (t)*z + (1-t)*x0t
    
    return xt



# --------- Image Processing Pipeline to create the degraded images dataset----------
if __name__== "__main__":
    print(f"Using device: {device} for generating degraded images.")
    print(f"Degradation mode: {IR_mode}")
    os.makedirs(output_folder, exist_ok=True)

    transform = T.ToTensor()
    image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('jpg', 'png', 'jpeg'))]

    for fname in image_files:
        path = os.path.join(data_folder, fname)
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)  # (1,3,H,W)

        A = set_operator(img_tensor.shape, IR_mode)
        degraded = A(img_tensor)

        #Ensure the size of degraded image matches the original, otherwise FLUX cant handle small inputs
        if IR_mode == "super resolution":
            degraded = PatchUpsample(degraded, scale)

        degraded_img = TF.to_pil_image(torch.clamp(degraded.squeeze(0), 0, 1).cpu())

        degraded_img.save(os.path.join(output_folder, fname))

    print(f"Saved {len(image_files)} degraded images to {output_folder}/")