import os
import torch
from tqdm import tqdm
import pandas as pd
from model.energy.clean_clip import DirectionalCLIP
from .utils import save_image, calculate_ssim, calculate_psnr


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
        all_clip, all_dclip, all_clip_right, all_dclip_right, all_clip_left, all_dclip_left = 0, 0, 0, 0, 0, 0
        sample_results = {
            'encode_text': [],
            'decode_text_right': [],
            'decode_text_left': [],
            'clip': [],
            'dclip': [],
            'clip_right': [],
            'dclip_right': [],
            'clip_left': [],
            'dclip_left': [],
            'psnr': [],
            'ssim': [],
            'l2': [],
        }
        idx = 0
        for original_img, img in tqdm(images):
            assert img.dim() == original_img.dim() == 3

            encode_text = data[idx]['encode_text']
            decode_text_right = data[idx]['decode_text']['right_prompt']
            decode_text_left = data[idx]['decode_text']['left_prompt']
            decode_text = data[idx]['decode_text_total']
            print('encode_text: {}'.format(encode_text))
            print('decode_text_right: {}'.format(decode_text_right))
            print('decode_text_left: {}'.format(decode_text_left))
            print('decode_text: {}'.format(decode_text))
            clip_score, dclip_score, clip_score_right, dclip_score_right, clip_score_left, dclip_score_left = self.directional_clip(img.unsqueeze(0),
                                                            original_img.unsqueeze(0),
                                                            [encode_text],
                                                            [decode_text],
                                                            [decode_text_right],
                                                            [decode_text_left],
                                                            )
            clip_score = clip_score.item()
            dclip_score = dclip_score.item()

            all_clip += clip_score
            all_dclip += dclip_score
            
            clip_score_right = clip_score_right.item()
            dclip_score_right = dclip_score_right.item()

            all_clip_right += clip_score_right
            all_dclip_right += dclip_score_right
            
            clip_score_left = clip_score_left.item()
            dclip_score_left = dclip_score_left.item()

            all_clip_left += clip_score_left
            all_dclip_left += dclip_score_left

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

            print('clip_score: {}'.format(clip_score))
            print('dclip_score: {}'.format(dclip_score))
            print('clip_score_right: {}'.format(clip_score_right))
            print('dclip_score_right: {}'.format(dclip_score_right))
            print('clip_score_left: {}'.format(clip_score_left))
            print('dclip_score_left: {}'.format(dclip_score_left))
            print('psnr: {}'.format(psnr))
            print('ssim: {}'.format(ssim))
            print('l2: {}'.format(l2))
            print('-' * 50)

            sample_results['encode_text'].append(encode_text)
            sample_results['decode_text_right'].append(decode_text_right)
            sample_results['decode_text_left'].append(decode_text_left)
            sample_results['clip'].append(clip_score)
            sample_results['dclip'].append(dclip_score)
            sample_results['clip_right'].append(clip_score_right)
            sample_results['dclip_right'].append(dclip_score_right)
            sample_results['clip_left'].append(clip_score_left)
            sample_results['dclip_left'].append(dclip_score_left)
            sample_results['psnr'].append(psnr)
            sample_results['ssim'].append(ssim)
            sample_results['l2'].append(l2)

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