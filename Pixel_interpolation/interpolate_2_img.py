import os
import sys
sys.path.append('..')

import torch
from PIL import Image
from utils.transform_utils import CenterCropLongEdge
from torchvision.utils import save_image as torch_save_image
from torchvision import transforms
import json

from evaluation.translate_text import Evaluator

def interpolate_images(img1, img2, interpol_type):

    assert img1.shape == img2.shape, "Images must have the same dimensions"
    
    # Create a tensor that gradually changes from 0 to 1 horizontally
    _, h, w = img1.shape
    interpolation_weights = torch.linspace(0, 1, w).view(1, 1, w).repeat(1, h, 1)
    
    # Interpolate between the images using the weights
    
    if interpol_type == 'exp' : 
        sharpness = 50.0 
        interpolation_weights = torch.exp(sharpness * (interpolation_weights - 0.5)) / (torch.exp(sharpness * (interpolation_weights - 0.5)) + 1)
    interpolated = img1.unsqueeze(0) * (1 - interpolation_weights) + img2.unsqueeze(0) * interpolation_weights
    
    return interpolated.squeeze(0)

def save_interpolated_image(img, save_path):
    """Save the interpolated image to disk."""
    torch_save_image(img.clamp(0, 1), save_path)
    print(f"Saved interpolated image to {save_path}")
    return save_path

def main(encode_img, encode_text, decode_text, interpol_type):
    
    # Load images
    encode_img = Image.open(encode_img).convert('RGB')
    print(f"encode_img size: {encode_img.size}")
    print(type(encode_img))
    img1 = Image.open("img1.png").convert('RGB')
    img2 = Image.open("img2.png").convert('RGB')

    # Same transform as preprocessor
    transform = transforms.Compose([
            CenterCropLongEdge(),
            transforms.Resize(512),
            transforms.ToTensor()
        ])
    
    img1 = transform(img1)
    img2 = transform(img2)
    encode_img = transform(encode_img)
    # Interpolate images
    interpolated_img = interpolate_images(img1, img2, interpol_type)
    
    # Save the interpolated image
    interp_save_path = os.path.join('interpolated.png')
    save_interpolated_image(interpolated_img, interp_save_path)
    
    # Create dummy data for the evaluator
    dummy_data = [{
        'encode_text': encode_text,
        'decode_text': decode_text
    }]
    class MetaArgs:
        def __init__(self):
            self.output_dir = os.getcwd() 
            
    evaluator = Evaluator(args=None, meta_args=MetaArgs())
    
    eval_images = [(encode_img, interpolated_img)]
    
    # Call the evaluator
    summary = evaluator.evaluate(
        images=eval_images,
        model=None,  # Not used
        weighted_loss=[],  # Not used
        losses={},  # Not used
        data=dummy_data,
        split='eval'
    )
    
    print("Evaluation complete!")
    print("Summary:", summary)

if __name__ == "__main__":
    
    interpol_type = 'exp'
    
    with open('prompts.json', 'r') as file:
        data = json.load(file)

    # Since your JSON is an array with one object, access the first item
    item = data[0]

    # Now you can access each property
    encode_text = item["encode_text"]
    decode_text = item["decode_text"]
    encode_img = item["encode_img"]
    
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Print the values to verify
    print(f"Encode text: {encode_text}")
    print(f"Decode text: {decode_text}")
    print(f"Encode_img: {encode_img}")
    
    main(encode_img, encode_text, decode_text, interpol_type)