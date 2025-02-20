import os
from pathlib import Path
import time
import datetime
import json
import re
import logging
import warnings
import random
import math
import collections.abc
import shutil
from typing import Dict, Union, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from transformers.optimization import AdamW, Adafactor, get_scheduler
from transformers.trainer_pt_utils import (
    get_parameter_names,
    reissue_pt_warnings,
    ShardSampler,
    torch_pad_and_concatenate,
    numpy_pad_and_concatenate,
)
from transformers.trainer import TrainerState
from transformers.trainer_utils import (
    IntervalStrategy,
    denumpify_detensorize,
)
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)
WEIGHTS_NAME = "pytorch_model.bin"
PREFIX_CHECKPOINT_DIR = "checkpoint"


def distributed_concat(tensor: Union[Tuple, List, torch.tensor], num_total_examples: Optional[int] = None):
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(distributed_concat(t, num_total_examples) for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({k: distributed_concat(v, num_total_examples) for k, v in tensor.items()})
        elif tensor is None:
            return None
        output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(output_tensors, tensor)
        output_tensors = [t if len(t.shape) > 0 else t[None] for t in output_tensors]
        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, dict):
        assert set(tensors.keys()) == set(new_tensors.keys())
        return type(tensors)({k: nested_concat(tensors[k], new_tensors[k], padding_index=padding_index) for k in tensors.keys()})
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif tensors is None:
        return None
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")


def nested_cpu(tensors):
    "CPU `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    elif isinstance(tensors, dict):
        return type(tensors)({k: distributed_concat(v) for k, v in tensors.items()})
    elif tensors is None:
        return None
    return tensors.cpu()


def nested_truncate(tensors, limit):
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    elif isinstance(tensors, dict):
        return type(tensors)({k: nested_truncate(v, limit) for k, v in tensors.items()})
    elif tensors is None:
        return None
    return tensors[:limit]


def _secs2timedelta(secs):
    """
    convert seconds to hh:mm:ss.msec, msecs rounded to 2 decimals
    """

    msec = int(abs(secs - int(secs)) * 100)
    return f"{datetime.timedelta(seconds=int(secs))}.{msec:02d}"


def speed_metrics(split, start_time, num_samples=None, num_steps=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:

    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    """
    runtime = time.time() - start_time
    result = {f"{split}/runtime": round(runtime, 4)}
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}/samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}/steps_per_second"] = round(steps_per_second, 3)
    return result


class Trainer:
    optimizer = None
    scheduler = None
    state = None

    def __init__(self,
                 args,
                 model,
                 compute_metrics,
                 eval_dataset,
                 visualizer,
                 wandb_run_dir=None,
                 ):

        # force device and distributed setup init explicitly
        logging.info(f'Rank {args.local_rank} device = {args.device}')

        self.args = args
        self.output_interval = 50
        self.model = model
        self.compute_metrics = compute_metrics
        self.eval_dataset = eval_dataset
        self.visualizer = visualizer
        self.wandb_run_dir = wandb_run_dir

        # Build training state tracker.
        self.state = TrainerState()

        # CUDA and distributed training.
        self.model = self.model.to(args.device)
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=self.args.ddp_find_unused_parameters,
        )

        if self.args.verbose and self.is_world_process_zero():
            print(self.model)

            # Setup output directory.
            if self.args.overwrite_output_dir:
                shutil.rmtree(self.args.output_dir)
            os.makedirs(self.args.output_dir, exist_ok=True)
        dist.barrier()

    def is_world_process_zero(self) -> bool:
        """Whether this process is the global main process"""
        return self.args.process_index == 0

    def is_local_process_zero(self) -> bool:
        """Whether this is the local main process"""
        return self.args.local_process_index == 0
    
    def _prepare_inputs(self, inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)
                
    def visualize(self, images, description):
        if not self.is_world_process_zero():
            return

        save_dir = self.args.output_dir
        self.visualizer.visualize(
            images=images,
            model=self.model.module,
            description=description,
            save_dir=save_dir,
            step=self.state.global_step,
        )
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        if not self.is_world_process_zero():
            return

        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)

        # wandb
        wandb.log(logs)
        
    def metrics_format(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Reformat Trainer metrics values to a human-readable format

        Args:
            metrics (:obj:`Dict[str, float]`):
                The metrics returned from train/evaluate/predict

        Returns:
            metrics (:obj:`Dict[str, float]`): The reformatted metrics
        """

        metrics_copy = metrics.copy()
        for k, v in metrics_copy.items():
            if "_mem_" in k:
                metrics_copy[k] = f"{ v >> 20 }MB"
            elif "_runtime" in k:
                metrics_copy[k] = _secs2timedelta(v)
            elif k == "total_flos":
                metrics_copy[k] = f"{ int(v) >> 30 }GF"
            elif type(metrics_copy[k]) == float:
                metrics_copy[k] = round(v, 4)

        return metrics_copy
    
    def save_metrics(self, split, metrics, combined=True):
        """
        Save metrics into a json file for that split, e.g. ``train_results.json``.

        Under distributed environment this is done only for a process with rank 0.

        Args:
            split (:obj:`str`):
                Mode/split name: one of ``train``, ``eval``, ``test``, ``all``
            metrics (:obj:`Dict[str, float]`):
                The metrics returned from train/evaluate/predict
            combined (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Creates combined metrics by updating ``all_results.json`` with metrics of this call

        To understand the metrics please read the docstring of :meth:`~transformers.Trainer.log_metrics`. The only
        difference is that raw unformatted numbers are saved in the current method.

        """
        if not self.is_world_process_zero():
            return

        path = os.path.join(self.args.output_dir, f"{split}_results.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

        if combined:
            path = os.path.join(self.args.output_dir, "all_results.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}

            all_metrics.update(metrics)
            with open(path, "w") as f:
                json.dump(all_metrics, f, indent=4, sort_keys=True)
        
    def log_metrics(self, split, metrics):
        """
        Log metrics in a specially formatted way

        Under distributed environment this is done only for a process with rank 0.

        Args:
            split (:obj:`str`):
                Mode/split name: one of ``train``, ``eval``, ``test``
            metrics (:obj:`Dict[str, float]`):
                The metrics returned from train/evaluate/predictmetrics: metrics dict
        """
        if not self.is_world_process_zero():
            return

        print(f"***** {split} metrics *****")
        metrics_formatted = self.metrics_format(metrics)
        k_width = max(len(str(x)) for x in metrics_formatted.keys())
        v_width = max(len(str(x)) for x in metrics_formatted.values())
        for key in sorted(metrics_formatted.keys()):
            print(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")
        
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_sampler = ShardSampler(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            num_processes=self.args.world_size,
            process_index=self.args.process_index,
        )

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.evaluation_loop
        metrics, num_samples = eval_loop(
            eval_dataloader,
            description="eval",
            metric_key_prefix=metric_key_prefix,
        )

        if self.is_world_process_zero():
            total_batch_size = self.args.eval_batch_size * self.args.world_size
            metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=num_samples,
                    num_steps=math.ceil(num_samples / total_batch_size),
                )
            )

            self.log(metrics)

        return metrics
    
    def prediction_step(
            self,
            inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        """
        self._prepare_inputs(inputs)
        
        logger.info('\n ***Doing a prediction step***')
        with torch.no_grad():
            # Calls the forward method of text_unsupervised_translation
            # images is the tuple (input_img, generated_img)
            images, weighted_loss, losses = self.model(**inputs)

        return images, weighted_loss, losses
    
    def evaluation_loop(
                self,
                dataloader: DataLoader,
                description: str,
                metric_key_prefix: str = "eval",
        ) -> Tuple[Dict[str, float], int]:
            """
            Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

            Works both with or without labels.
            """

            batch_size = dataloader.batch_size
            logger.info(f"***** Running {description} *****")
            if isinstance(dataloader.dataset, collections.abc.Sized):
                logger.info(f"  Num examples = {len(dataloader.dataset)}")
            else:
                logger.info("  Num examples: Unknown")
            logger.info(f"  Batch size = {batch_size}")

            self.model.eval()

            # Do this before wrapping.
            eval_dataset = dataloader.dataset

            # Initialize containers
            # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
            prediction_outputs_host = None
            # losses/preds/labels on CPU (final containers)
            all_prediction_outputs = None
            # Will be useful when we have an iterable dataset so don't know its length.

            # Main evaluation loop
            for step, inputs in tqdm(enumerate(dataloader)):
                
                # Prediction step : getting the new image
                prediction_outputs = self.prediction_step(inputs)

                # Update containers on host
                if prediction_outputs is not None:
                    prediction_outputs = distributed_concat(prediction_outputs)
                prediction_outputs_host = (
                    prediction_outputs if prediction_outputs_host is None else
                    nested_concat(prediction_outputs_host, prediction_outputs, padding_index=-100)
                )

            # Gather all remaining tensors and put them back on the CPU
            if prediction_outputs_host is not None:
                prediction_outputs = nested_cpu(prediction_outputs_host)
                all_prediction_outputs = (
                    prediction_outputs if all_prediction_outputs is None else
                    nested_concat(all_prediction_outputs, prediction_outputs, padding_index=-100)
                )

            # Number of samples
            num_samples = len(eval_dataset)

            # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
            # samplers has been rounded to a multiple of batch_size, so we truncate.
            if all_prediction_outputs is not None:
                all_prediction_outputs = nested_truncate(all_prediction_outputs, num_samples)

            images, weighted_loss, losses = all_prediction_outputs

            # Metrics!
            if self.is_world_process_zero():
                if self.compute_metrics and all_prediction_outputs:
                    metrics = self.compute_metrics(images,
                                                self.model.module,
                                                weighted_loss,
                                                losses,
                                                dataset=eval_dataset,
                                                split=metric_key_prefix,
                                                )
                else:
                    metrics = {}

                # To be JSON-serializable, we need to remove numpy types or zero-d tensors
                metrics = denumpify_detensorize(metrics)

                # Prefix all keys with metric_key_prefix + '/'
                for key in list(metrics.keys()):
                    if not key.startswith(f"{metric_key_prefix}/"):
                        metrics[f"{metric_key_prefix}/{key}"] = metrics.pop(key)

                # Weighted loss and losses
                metrics[f"{metric_key_prefix}/weighted_loss"] = weighted_loss.mean(0).item()
                for key, value in losses.items():
                    metrics[f"{metric_key_prefix}/{key}"] = value.mean(0).item()

                # Save images.
                self.visualize(images, description)
            else:
                metrics = None

            return metrics, num_samples