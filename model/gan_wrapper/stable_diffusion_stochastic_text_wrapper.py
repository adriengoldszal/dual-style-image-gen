import os
import argparse
import sys
sys.path.append(os.path.abspath('model/lib/stable_diffusion'))
import glob
from omegaconf import OmegaConf
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import autocast
from contextlib import contextmanager, nullcontext

from model.energy.clean_clip import DirectionalCLIP
from txt2img import load_model_from_config, DDIMSampler
from ..model_utils import requires_grad


def prepare_stable_diffusion_text(source_model_type):
    print('First of all, when the code changes, make sure that no part in the model is under no_grad!')

    config = OmegaConf.load(os.path.join('model/lib/stable_diffusion/configs/stable-diffusion/v1-inference.yaml'))
    ckpt = os.path.join('ckpts', 'stable_diffusion', source_model_type)

    return config, ckpt


def get_condition(model, text, bs):
    assert isinstance(text, list)
    assert isinstance(text[0], str)
    uc = model.get_learned_conditioning(bs * [""])
    print("model.cond_stage_key: ", model.cond_stage_key)
    c = model.get_learned_conditioning(text)
    print("c.shape: ", c.shape)
    print('-' * 50)
    return c, uc


def convsample_ddim_conditional(model, steps, shape, x_T, skip_steps, eta, eps_list, scale, text_right, text_left):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    c_right, uc_right = get_condition(model, text_right, bs)
    c_left, uc_left = get_condition(model, text_left, bs)
    samples, intermediates = ddim.sample_with_eps(steps,
                                                  eps_list,
                                                  conditioning_right=c_right,
                                                  conditioning_left=c_left,
                                                  batch_size=bs,
                                                  shape=shape,
                                                  eta=eta,
                                                  verbose=False,
                                                  x_T=x_T,
                                                  skip_steps=skip_steps,
                                                  unconditional_guidance_scale=scale,
                                                  unconditional_conditioning=uc_right,
                                                  log_every_t=10,
                                                  )
    return samples, intermediates


def make_convolutional_sample_with_eps_conditional(model, custom_steps, eta, x_T, skip_steps, eps_list,
                                                   scale, text_right, text_left):
    with model.ema_scope("Plotting"):
        sample, intermediates = convsample_ddim_conditional(model,
                                                            steps=custom_steps,
                                                            shape=x_T.shape,
                                                            x_T=x_T,
                                                            skip_steps=skip_steps,
                                                            eta=eta,
                                                            eps_list=eps_list,
                                                            scale=scale,
                                                            text_right=text_right,
                                                            text_left=text_left)
    
    
    print(f'In the SDS Wrapper:')
    print(f'- sample shape: {sample.shape}')
    print(f'- intermediates x_inter length: {len(intermediates["x_inter"])}')
    print(f'- intermediates x_inter[0] shape: {intermediates["x_inter"][0].shape}')
    print(f'- intermediates pred_x0 length: {len(intermediates["pred_x0"])}')
    print(f'- intermediates pred_x0[0] shape: {intermediates["pred_x0"][0].shape}')
    
    x_sample = model.decode_first_stage(sample)
    print(f'After decode_first_stage x_sample {x_sample.shape}')
    
    #We also decode the rest to be able to get them
    decoded_x_inter = []
    for x_t in intermediates['x_inter']:
        decoded_x_t = model.decode_first_stage(x_t)
        decoded_x_inter.append(decoded_x_t)

    # Decode all predicted x_0 states    
    decoded_pred_x0 = []
    for x_0 in intermediates['pred_x0']:
        decoded_x_0 = model.decode_first_stage(x_0)
        decoded_pred_x0.append(decoded_x_0)

    decoded_intermediates = {
        'x_inter': decoded_x_inter,
        'pred_x0': decoded_pred_x0
    }

    return x_sample, decoded_intermediates


def ddpm_ddim_encoding_conditional(model, steps, shape, eta, white_box_steps, skip_steps, x0, scale, text):
    """ returns z_list = [xT, ε_T, ε_T-1, ..., ε_1] """
    with model.ema_scope("Plotting"):
        ddim = DDIMSampler(model)
        bs = shape[0]
        shape = shape[1:]
        c, uc = get_condition(model, text, bs)

        z_list = ddim.ddpm_ddim_encoding(steps,
                                         conditioning=c,
                                         batch_size=bs,
                                         shape=shape,
                                         eta=eta,
                                         white_box_steps=white_box_steps,
                                         skip_steps=skip_steps,
                                         verbose=False,
                                         x0=x0,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc,
                                         )

    return z_list


class SDStochasticTextWrapper(torch.nn.Module):

    def __init__(self, source_model_type, custom_steps, eta, white_box_steps, skip_steps,
                 encoder_unconditional_guidance_scales=None, decoder_unconditional_guidance_scales=None,
                 n_trials=None):
        super(SDStochasticTextWrapper, self).__init__()
        print('==INITIALISATION==')
        self.encoder_unconditional_guidance_scales = encoder_unconditional_guidance_scales
        self.decoder_unconditional_guidance_scales = decoder_unconditional_guidance_scales
        self.n_trials = n_trials

        # Set up generator
        self.config, self.ckpt = prepare_stable_diffusion_text(source_model_type)

        print(self.config)

        self.generator = load_model_from_config(self.config, self.ckpt, verbose=True)
        self.precision = "full"

        print(75 * "-")

        self.eta = eta
        self.custom_steps = custom_steps
        self.white_box_steps = white_box_steps
        self.skip_steps = skip_steps

        self.resolution = 512
        print(f"resolution: {self.resolution}")

        print(f'Using DDIM sampling with {self.custom_steps} sampling steps and eta={self.eta}')

        # Freeze.
        # requires_grad(self.generator, False)

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

        # Directional CLIP score.
        self.directional_clip = DirectionalCLIP()

    def generate(self, z_ensemble, decode_text_right, decode_text_left):
        print("==GENERATE==")
        precision_scope = autocast if self.precision == "autocast" else nullcontext
        with precision_scope("cuda"):
            img_ensemble = []
            intermediates_ensemble = []
            for i, z in enumerate(z_ensemble):
                print(f'z.shape : {z.shape}')
                skip_steps = self.skip_steps[i % len(self.skip_steps)]
                bsz = z.shape[0]
                print(f'white_box_steps : {self.white_box_steps}')
                print(f'skip_steps : {skip_steps}')
                if self.white_box_steps != -1:
                    eps_list = z.view(bsz, (self.white_box_steps - skip_steps), self.generator.channels, self.generator.image_size, self.generator.image_size)
                else:
                    eps_list = z.view(bsz, 1, self.generator.channels, self.generator.image_size, self.generator.image_size)
                
                x_T = eps_list[:, 0]
                eps_list = eps_list[:, 1:]
                print(f'x_T.shape : {x_T.shape}')
                print(f'eps_list.shape {eps_list.shape}')
                print(f'custom_steps : {self.custom_steps}')
                print(f'eta : {self.eta}')
                # Uses the same epsilons for decoding !

                for decoder_unconditional_guidance_scale in self.decoder_unconditional_guidance_scales:
                    img, decoded_intermediates = make_convolutional_sample_with_eps_conditional(self.generator,
                                                                         custom_steps=self.custom_steps,
                                                                         eta=self.eta,
                                                                         x_T=x_T,
                                                                         skip_steps=skip_steps,
                                                                         eps_list=eps_list,
                                                                         scale=decoder_unconditional_guidance_scale,
                                                                         text_right=decode_text_right,
                                                                         text_left=decode_text_left)
                    img_ensemble.append(img)
                    intermediates_ensemble.append(decoded_intermediates)
        print(f'img end generate : {img_ensemble[0].shape}')
        print(f'intermediates : {intermediates_ensemble[0].keys()}')
        return img_ensemble, intermediates_ensemble

    def encode(self, image, encode_text):
        print("==ENCODE==")
        # Eval mode for the generator.
        self.generator.eval()

        precision_scope = autocast if self.precision == "autocast" else nullcontext

        # Normalize.
        image = (image - 0.5) * 2.0
        # Resize.
        assert image.shape[2] == image.shape[3] == self.resolution
        with precision_scope("cuda"):
            with torch.no_grad():
                # Encode.
                encoder_posterior = self.generator.encode_first_stage(image)
                z = self.generator.get_first_stage_encoding(encoder_posterior)
                x0 = z

        with precision_scope("cuda"):
            bsz = image.shape[0]
            z_ensemble = []
            for trial in range(self.n_trials):
                for encoder_unconditional_guidance_scale in self.encoder_unconditional_guidance_scales:
                    for skip_steps in self.skip_steps:
                        print(f'Image z_list for : trial {trial}/{self.n_trials}, encoder guidance scale {encoder_unconditional_guidance_scale}, skip step {skip_steps}')
                        with torch.no_grad():
                            # DDIM forward.
                            z_list = ddpm_ddim_encoding_conditional(self.generator,
                                                                    steps=self.custom_steps,
                                                                    shape=x0.shape,
                                                                    eta=self.eta,
                                                                    white_box_steps=self.white_box_steps,
                                                                    skip_steps=skip_steps,
                                                                    x0=x0,
                                                                    scale=encoder_unconditional_guidance_scale,
                                                                    text=encode_text)
                            z = torch.stack(z_list, dim=1).view(bsz, -1)
                            z_ensemble.append(z)
        print(f'z_ensemble.shape {z_ensemble[0].shape}')
        return z_ensemble

    def forward(self, z_ensemble, original_img, encode_text, decode_text_right, decode_text_left):
        # Eval mode for the generator.
        self.generator.eval()

        img_ensemble, intermediates_ensemble = self.generate(z_ensemble, decode_text_right, decode_text_left)
        assert len(img_ensemble) == len(self.decoder_unconditional_guidance_scales) * len(self.encoder_unconditional_guidance_scales) * len(self.skip_steps) * self.n_trials
     
        # Post process.
        img = self.post_process(img_ensemble[0])  # Just take the first (and only) image
       
        processed_intermediates = {
            'x_inter': [self.post_process(x) for x in intermediates_ensemble[0]['x_inter']],
            'pred_x0': [self.post_process(x) for x in intermediates_ensemble[0]['pred_x0']]
        }
        print(f'img.shape {img.shape}')
        print(f'intermediates keys {processed_intermediates.keys()}')
        return img, processed_intermediates

    @property
    def device(self):
        return next(self.parameters()).device




