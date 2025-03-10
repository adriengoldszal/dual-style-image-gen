import os
import torch
from tqdm import tqdm
import pandas as pd
from model.energy.clean_clip import DirectionalCLIP
from .utils import save_image, calculate_ssim, calculate_psnr, calculate_lpips, total_variation_center_width
from .gram import compute_style_similarity

class Evaluator(object):

    def __init__(self, args, meta_args):
        self.args = args
        self.meta_args = meta_args

        self.directional_clip = DirectionalCLIP()

    def evaluate(self, images, model, weighted_loss, losses, data, split):
        """

        Args:
            images: list of images, or list of tuples of images
            model: model to evaluate
            weighted_loss: list of scalar tensors
            losses: dictionary of lists of scalar tensors
            data: list of dictionary
            split: str

        Returns:

        """
        assert split in ['eval', 'test']

        # Add metrics here.
        f_gen = os.path.join(self.meta_args.output_dir, 'temp_gen')
        f_ref = os.path.join(self.meta_args.output_dir, 'temp_ref')
        if os.path.exists(f_gen):
            os.remove(f_gen)
        os.mkdir(f_gen)
        if os.path.exists(f_ref):
            os.remove(f_ref)
        os.mkdir(f_ref)

        assert len(data) == len(images)
        n = len(images)
        all_psnr, all_ssim, all_l2 = 0, 0, 0
        all_clip, all_dclip = 0, 0
        sample_results = {
            'encode_text': [],
            'decode_text': [],
            'clip': [],
            'dclip': [],
            'psnr': [],
            'ssim': [],
            'l2': [],
            'style_right': [],
            'style_left': [],
        }
        idx = 0
        for original_img, img in tqdm(images):
            assert img.dim() == original_img.dim() == 3

            encode_text = data[idx]['encode_text']
            decode_text = data[idx]['decode_text']
            style_right = data[idx]['style_right']
            style_left = data[idx]['style_left']
            print('encode_text: {}'.format(encode_text))
            print('decode_text: {}'.format(decode_text))
            print('style right: {}'.format(style_right))
            print('style left: {}'.format(style_left))

            clip_score, dclip_score = self.directional_clip(img.unsqueeze(0),
                                                            original_img.unsqueeze(0),
                                                            [encode_text],
                                                            [decode_text],
                                                            )
            clip_score = clip_score.item()
            dclip_score = dclip_score.item()

            all_clip += clip_score
            all_dclip += dclip_score

            img = img.clamp(0, 1)
            original_img = original_img.clamp(0, 1)

            psnr = calculate_psnr(img, original_img).item()
            all_psnr += psnr
            ssim = calculate_ssim(
                (img.numpy() * 255).transpose((1, 2, 0)),
                (original_img.numpy() * 255).transpose((1, 2, 0)),
            )
            all_ssim += ssim
            l2 = torch.sqrt(
                ((img - original_img) ** 2).sum(2).sum(1).sum(0)
            ).item()
            all_l2 += l2
            
            lpips = calculate_lpips(img, original_img)
            
            total_variation_original = total_variation_center_width(original_img)
            total_variation_generated = total_variation_center_width(img)
            
            style_left_score = compute_style_similarity(img, style_left)
            style_right_score = compute_style_similarity(img, style_right)

            print('clip_score: {}'.format(clip_score))
            print('dclip_score: {}'.format(dclip_score))
            print('psnr: {}'.format(psnr))
            print('ssim: {}'.format(ssim))
            print('l2: {}'.format(l2))
            print('lpips: {}'.format(lpips))
            print('total_variation_original: {}'.format(total_variation_original))
            print('total_variation_generated: {}'.format(total_variation_generated))
            print('delta_total_variation: {}'.format(total_variation_generated - total_variation_original))
            print(f'Style left {style_left} score {style_left_score}')
            print(f'Style right {style_right} score {style_right_score}')
            print('-' * 50)

            sample_results['encode_text'].append(encode_text)
            sample_results['decode_text'].append(decode_text)
            sample_results['clip'].append(clip_score)
            sample_results['dclip'].append(dclip_score)
            sample_results['psnr'].append(psnr)
            sample_results['ssim'].append(ssim)
            sample_results['l2'].append(l2)
            sample_results['style_right'].append(style_right)
            sample_results['style_left'].append(style_left)

            assert img.shape == original_img.shape
            save_image(os.path.join(f_gen, '{}.png'.format(idx)), img)
            idx += 1

        summary = {
            "psnr": all_psnr / n,
            "ssim": all_ssim / n,
            "l2": all_l2 / n,
            "clip": all_clip / n,
            "d-clip": all_dclip / n,
            "clip_right": all_clip / n,
            "d-clip_right": all_dclip / n,
            "clip_left": all_clip / n,
            "d-clip_left": all_dclip / n,
        }

        # Save all results with pandas.
        # for key, value in sample_results.items():
        #     print(f"{key}: {len(value)}")

        df = pd.DataFrame(sample_results)
        df.to_csv(os.path.join(self.meta_args.output_dir, '{}_results.csv'.format(split)), index=False)

        return summary
