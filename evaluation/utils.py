import numpy as np
import torch
import cv2
from torchvision import utils
import lpips
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy.linalg import sqrtm


def save_image(image_path, image):
    assert image.dim() == 3 and image.shape[0] == 3

    utils.save_image(image, image_path)


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    assert img1.shape == img2.shape
    assert img1.ndim == 2 and img2.ndim == 2
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_psnr(img1, img2):
    assert img1.shape == img2.shape
    assert (img1 >= 0).all() and (img1 <= 1).all()
    assert (img2 >= 0).all() and (img2 <= 1).all()
    mse = ((img1 - img2) ** 2).mean(2).mean(1).mean(0)
    if mse == 0:
        return 100
    return 10 * torch.log10(1 / mse)


def calculate_lpips(img1, img2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)
    
    if img1.ndim == 2:
        img1 = img1.unsqueeze(0).repeat(3, 1, 1)
    elif img1.ndim == 3:
        if img1.shape[0] != 3 and img1.shape[2] == 3:
            img1 = img1.permute(2, 0, 1)
    if img2.ndim == 2:
        img2 = img2.unsqueeze(0).repeat(3, 1, 1)
    elif img2.ndim == 3:
        if img2.shape[0] != 3 and img2.shape[2] == 3:
            img2 = img2.permute(2, 0, 1)
    
    img1 = img1.unsqueeze(0).float()
    img2 = img2.unsqueeze(0).float()
    
    def normalize_img(tensor):
        if tensor.max() > 1:
            return tensor / 127.5 - 1.0
        else:
            return tensor * 2 - 1.0
    
    img1 = normalize_img(img1)
    img2 = normalize_img(img2)
    
    loss_fn = lpips.LPIPS(net='vgg').to(device)
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    with torch.no_grad():
        dist = loss_fn(img1, img2)
    
    return dist.item()

def total_variation_center_width(image):
    image = image.float()
    channels, height, width = image.shape

    start = int(width/2 - 0.1 * width)
    end = int(width/2 + 0.1 * width)
    
    sub_image = image[:, :, start:end]
    
    diff_h = sub_image[:, :, 1:] - sub_image[:, :, :-1]
    diff_v = sub_image[:, 1:, :] - sub_image[:, :-1, :]
    
    tv_h = torch.sqrt(diff_h ** 2 + 1e-6)
    tv_v = torch.sqrt(diff_v ** 2 + 1e-6)
    
    tv = torch.sum(tv_h) + torch.sum(tv_v)
    return tv.item()

def extract_style_from_description(description):

    description_lower = description.lower()
    
    if "mosaic" in description_lower:
        return "MOSAIC"
    elif "minecraft" in description_lower:
        return "MINECRAFT"
    elif "andy warhol" in description_lower or "warhol" in description_lower:
        return "ANDY WARHOL POP"
    elif "van gogh" in description_lower or "gogh" in description_lower:
        return "VAN GOGH"
    else:
        return "UNKNOWN"
