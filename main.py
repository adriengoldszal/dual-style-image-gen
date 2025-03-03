import logging
import os
import torch

import datasets
from transformers import (
    HfArgumentParser,
    set_seed,
)
from utils.config_utils import get_config
from utils.program_utils import get_model, get_preprocessor, get_evaluator, get_visualizer
from preprocess.to_model import get_multi_task_dataset_splits
from utils.training_arguments import CustomTrainingArguments
from trainer.trainer import Trainer

logger = logging.getLogger(__name__)


def get_dataset_splits(args):
    cache_root = os.path.join('output', 'cache')
    os.makedirs(cache_root, exist_ok=True)
    name2dataset_splits = dict()
    for name, arg_path in args.arg_paths:
        task_args = get_config(arg_path)
        task_raw_data_splits = datasets.load_dataset(
            path=task_args.raw_data.data_program,
            cache_dir=task_args.raw_data.data_cache_dir,
        )
        # Crée une instance de preprocessor de translate_text512.py
        task_preprocessor = get_preprocessor(task_args.preprocess.preprocess_program)
        # Crée notre instance de devdataset (aussi dans translate_text512.py)
        task_dataset_splits = task_preprocessor(task_args, args).preprocess(task_raw_data_splits, cache_root)

        name2dataset_splits[name] = task_dataset_splits
        
        for name, splits in name2dataset_splits.items():
            print(f"Dataset main: {name}")
            for split_name, split_data in splits.items():
                print(f"  Split: {split_name}, Size: {len(split_data)}")
                for i in range(len(split_data)):
                    example = split_data[i]
                    print(f"\nExample {i}:")
                    print(f"Encode text: {example['encode_text']}")
                    print(f"Decode text right: {example['decode_text_right']}")
                    print(f"Decode text left: {example['decode_text_left']}")
                    if 'original_image' in example:
                        img = example['original_image']
                        if torch.is_tensor(img):
                            print(f"Image shape: {img.shape}")
                            print(f"Image type: {img.dtype}")

    return get_multi_task_dataset_splits(meta_args=args, name2dataset_splits=name2dataset_splits)

def main():

    # Get training_args and args.
    parser = HfArgumentParser(
        (
            CustomTrainingArguments,
        )
    )
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    args = get_config(training_args.cfg)

    # Deterministic behavior of torch.addmm.
    # Please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)
    # cudnn.deterministic = True

    # Setup output directory.
    os.makedirs(training_args.output_dir, exist_ok=True)
    args.output_dir = training_args.output_dir
    # Build dataset splits.
    dataset_splits = get_dataset_splits(args)
    print(f'dataset_splits : {dataset_splits}')
    # Initialize evaluator.
    evaluator = get_evaluator(args.evaluation.evaluator_program)(args)
    # Initialize visualizer.
    visualizer = get_visualizer(args.visualization.visualizer_program)(args)

    # Initialize model : text_unsupervised_translation
    model = get_model(args.model.name)(args)

    # Initialize Trainer, only used for eval anyway
    trainer = Trainer(
        args=training_args,
        model=model,
        compute_metrics=evaluator.evaluate,
        eval_dataset=dataset_splits['dev'],
        visualizer=visualizer,
    )
    print(f'Rank {training_args.local_rank} Trainer build successfully.')

    # Evaluation after training
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate(
        metric_key_prefix="eval",
    )
    logger.info("*** End Evaluate ***")
    metrics["eval_samples"] = len(dataset_splits['dev'])

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)

    main()
