from utils import save_image, calculate_ssim, calculate_psnr, ssim, calculate_lpips, total_variation
import cv2
import torch
import numpy

def load_image(image_path):
    """
    Charge une image depuis le chemin fourni.
    L'image est lue en couleur (RGB) et retournée sous forme de numpy array.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Erreur lors du chargement de l'image : {image_path}")
    # Convertir de BGR (format OpenCV) en RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def main():
    # Remplacez ces chemins par les chemins réels de vos images
    image1_path = "chemin/vers/image1.jpg"
    image2_path = "chemin/vers/image2.jpg"
    
    # Charger les images
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)
    
    # Convertir les images numpy (H, W, C) en tenseurs torch (C, H, W)
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1)
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1)
    
    # Calculer LPIPS
    lpips_score = calculate_lpips(img1, img2)
    print("LPIPS:", lpips_score)
    
    # Calculer la variation totale sur la première image
    tv_score = total_variation(img1_tensor)
    print("Total Variation (image 1):", tv_score)

if __name__ == '__main__':
    main()