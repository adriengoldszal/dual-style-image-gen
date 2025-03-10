import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import glob

vgg = models.vgg19(pretrained=True).features.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def gram_matrix(features):
    """ Compute the Gram matrix of feature maps """
    b, c, h, w = features.shape
    features = features.reshape(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram.div(c * h * w)

def get_vgg_gram(image, layers=['5', '10', '19', '28']):
    """ Extracts Gram matrices from selected layers of VGG """
    features = []
    x = image
    for name, layer in vgg._modules.items():
        x = layer(x)
        if name in layers:  # Use selected layers
            features.append(gram_matrix(x))
    return features  # List of Gram matrices

def gram_similarity(image1, image2):
    """
    Computes Gram matrix similarity between a single image and a batch of images
    """
    gram1 = get_vgg_gram(image1)
    
    batch_size = image2.size(0)
    similarities = []
    
    for i in range(batch_size):
        ref_img = image2[i:i+1]
        gram2 = get_vgg_gram(ref_img)
        
        sim = sum(torch.norm(g1 - g2) for g1, g2 in zip(gram1, gram2))
        similarities.append(sim)
    
    return torch.tensor(similarities, device=image1.device)

def compute_style_similarity(generated_img, style_name):
    """
    Compute average gram similarity between generated image and all images in style folder
    """
    # Path to the style folder (same level as the file)
    style_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gram_styles", style_name)
    print(f'style folder {style_folder}')
    
    # Get all image files from the folder
    image_extensions = ['jpg', 'jpeg', 'png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(style_folder, f'*.{ext}')))

    # Load reference images in a batch
    ref_images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        ref_img = transform(img)
        print(ref_img.shape)
        ref_images.append(ref_img)

    # Stack images to create a batch (B, C, H, W)
    ref_images_batch = torch.stack(ref_images).to(generated_img.device)

    # Compute similarity for each reference image in the batch
    sim_batch = gram_similarity(generated_img.unsqueeze(0), ref_images_batch)
    print(f'Sim batch calculated ! {sim_batch.shape}')

    # Compute the average similarity
    avg_similarity = sim_batch.mean().item()
    
    # Print individual similarities
    for idx, sim in enumerate(sim_batch):
        print(f"Similarity with {os.path.basename(image_paths[idx])}: {sim.item()}")

    return avg_similarity