import os
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ------------- User Settings -------------------
data_folder = "/scratch/gilbreth/lwickrem/data/HandpickedDegrads/gt"       # Your input image folder
output_folder = "/scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres" # Where degraded images will be saved
IR_mode = "super resolution"  # Select degradation mode
scale = 4                  # Used for super resolution
device = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------


# ---------- Degradation Operators -------------
#https://github.com/wyhuai/DDNM

def color2gray(x):
    coef = 1/3
    gray = x[:, 0, :, :] * coef + x[:, 1, :, :] * coef + x[:, 2, :, :] * coef
    return gray.unsqueeze(1).repeat(1, 3, 1, 1)

def PatchUpsample(x, scale):
    n, c, h, w = x.shape
    x = x.view(n, c, h, 1, w, 1).expand(-1, -1, -1, scale, -1, scale)
    return x.contiguous().view(n, c, h * scale, w * scale)

def set_operator(IR_mode, img_shape):
    _, _, h, w = img_shape
    if IR_mode == "colorization":
        return color2gray

    elif IR_mode == "inpainting":
        mask = torch.ones(img_shape, device=device)
        mask[:, :, h//4:h*3//4, w//4:w*3//4] = 0
        return lambda z: z * mask

    elif IR_mode == "super resolution":
        down = torch.nn.AdaptiveAvgPool2d((h // scale, w // scale))
        #maintain input image size
        up = torch.nn.Upsample(size=(h, w), mode="bilinear", align_corners=False)
        return lambda z: up(down(z))

    elif IR_mode == "old photo restoration":
        mask = torch.ones(img_shape, device=device)
        mask[:, :, h//4:h*3//4, w//4:w*3//4] = 0
        A1 = lambda z: z * mask
        A2 = color2gray
        A3 = torch.nn.AdaptiveAvgPool2d((h // scale, w // scale))
        return lambda z: A3(A2(A1(z)))

    else:
        raise ValueError(f"Unknown IR_mode: {IR_mode}")
# -----------------------------------------------


# --------- Image Processing Pipeline ----------
os.makedirs(output_folder, exist_ok=True)

transform = T.ToTensor()
image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('jpg', 'png', 'jpeg'))]

for fname in image_files:
    path = os.path.join(data_folder, fname)
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # (1,3,H,W)

    A = set_operator(IR_mode, img_tensor.shape)
    degraded = A(img_tensor)
    degraded_img = TF.to_pil_image(torch.clamp(degraded.squeeze(0), 0, 1).cpu())

    degraded_img.save(os.path.join(output_folder, fname))

print(f"Saved {len(image_files)} degraded images to {output_folder}/")