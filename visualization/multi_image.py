import os
import math
import torch
import torch.nn.functional as F

from utils.file_utils import save_images


class Visualizer(object):
    def __init__(self, args):
        self.args = args

    def visualize(self,
                  images,
                  intermediates,
                  model, 
                  description: str,
                  save_dir: str,
                  step: int,
                  ):
        # Original visualization of input and final output
        bsz, c, h, w = images[0].shape
        images = torch.stack(images, dim=1).view(bsz * len(images), c, h, w)
        images = images[:100 * len(images), :, :, :]

        save_images(
            images,
            output_dir=save_dir,
            file_prefix=description,
            nrows=8,
            iteration=step,
        )

        # Lower resolution for original images
        images_256 = F.interpolate(
            images,
            (256, 256),
            mode='bicubic',
        )
        save_images(
            images_256,
            output_dir=save_dir,
            file_prefix=f'{description}_256',
            nrows=8,
            iteration=step,
        )

        # Save intermediate x_t states
        if intermediates is not None and 'x_inter' in intermediates:
            x_inter = intermediates['x_inter']
            if len(x_inter) > 0:
                x_inter = torch.stack(x_inter, dim=1).view(bsz * len(x_inter), c, h, w)
                x_inter = x_inter[:100 * len(x_inter), :, :, :]
                save_images(
                    x_inter,
                    output_dir=save_dir,
                    file_prefix=f'{description}_x_inter',
                    nrows=8,
                    iteration=step,
                )

        # Save predicted x_0 states
        if intermediates is not None and 'pred_x0' in intermediates:
            pred_x0 = intermediates['pred_x0']
            if len(pred_x0) > 0:
                pred_x0 = torch.stack(pred_x0, dim=1).view(bsz * len(pred_x0), c, h, w)
                pred_x0 = pred_x0[:100 * len(pred_x0), :, :, :]
                save_images(
                    pred_x0,
                    output_dir=save_dir,
                    file_prefix=f'{description}_pred_x0',
                    nrows=8,
                    iteration=step,
                )


