# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for protein sequences and existing PaccmannProteomics framework by flp@ibm.zurich.ch
""" 
Finetuning transformer language models for sequence classification and regression.
Based on GLUE.
"""
import dataclasses
import os
import glob
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, List, Tuple
import numpy as np
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from utils.metrics_clf import glue_compute_metrics
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from loguru import logger
from paccmann_proteomics.data.datasets.seq_clf import SeqClfDataset
from paccmann_proteomics.data.processors.seq_clf import (
    seq_clf_output_modes, seq_clf_tasks_num_labels, seq_clf_processors
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    config_name: Optional[str] = field(
        default=None, metadata={'help': 'Pretrained config name or path if not the same as model_name'}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={'help': 'Where do you want to store the pretrained models downloaded from AWS'}
    )
    continue_from_checkpoint: bool = field(
        default=False,
        metadata={'help': 'Whether to continue training from `model_name_or_path/checkpoint-<Trainer.global_step>/`'},
    )

def _sorted_checkpoints(
    model_dir, checkpoint_prefix='checkpoint', use_modification_time=False
) -> List[str]:
    """
    TODO: refactor to paccmann_proteomics/utils
    Private method discovers model checkpoint directories within the global model_dir, 
    returns a sorted list with directory paths.

    Supports sorting by checkpoint directory modification time (use with care!) or directory suffix 
    which contains the global step at which checkpoint directory was saved.

    Args:
        model_dir ([type]): location of model with checkpoint directories
        checkpoint_prefix (str, optional): directory prefix concatenated 
            with "-<Trainer.global_step>" . Defaults to "checkpoint".
        use_mtime (bool, optional): sort by directory modification time. Defaults to False.

    Returns:
        List[str]: [description]
    
    Dependancies:
        Glob, Re
    """
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(model_dir, '{}-*'.format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_modification_time:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    parser = HfArgumentParser(dataclass_types=(ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.'
        )

    logger.warning(
        'Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s',
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info('Training/evaluation parameters %s', training_args)

    # Set seed
    set_seed(training_args.seed)


    # Load task-specific number of labels (==1 if regression) and output modes)
    try:
        num_labels = seq_clf_tasks_num_labels[data_args.task_name]
        logger.info('number of labels: ', num_labels)
        output_mode = seq_clf_output_modes[data_args.task_name]
        logger.info('task output mode: ', output_mode)
    except KeyError:
        raise ValueError('Task not found: %s' % (data_args.task_name))


    # Load pretrained model and tokenizer
    if model_args.config_name:
        logger.info('config_name provided as: %s', model_args.config_name)
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    
    elif model_args.model_name_or_path:
        logger.info('model_name_or_path provided as: %s', model_args.model_name_or_path)
        
        if model_args.continue_from_checkpoint:
            logger.info('checking for the newest checkpoint directory %s/checkpoint-<Trainer.global_step>', model_args.model_name_or_path)
            sorted_checkpoints = _sorted_checkpoints(model_args.model_name_or_path)
            logger.info('checkpoints found: %s', sorted_checkpoints) 
            if len(sorted_checkpoints) == 0:
                raise ValueError('Used --continue_from_checkpoint but no checkpoint was found in --model_name_or_path.')
            else:
                model_args.model_name_or_path = sorted_checkpoints[-1]
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        # use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool('.ckpt' in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset =(
        SeqClfDataset(
            args = data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir
        ) 
        if training_args.do_train
        else None
    )
    eval_dataset = (
        SeqClfDataset(
            args = data_args, tokenizer=tokenizer, mode='dev', cache_dir=model_args.cache_dir
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        SeqClfDataset(
            args = data_args, tokenizer=tokenizer, mode='test', cache_dir=model_args.cache_dir
        )
        if training_args.do_predict
        else None
    )


    # Metrics computation for a task
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction) -> Dict:
            """computes metrics

            Args:
                p (EvalPrediction): NamedTuple with predictions and label ids

            Returns:
                Dict: a dict with metrics
            """
            if output_mode == 'classification':
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == 'regression':
                preds = np.squeeze(p.predictions)  # see x = np.array([[[0], [1], [2]]]) x.shape np.squeeze(x).shape
            # logger.info('DEBUGGING testing: ')
            # logger.info('preds: ', '\n', preds)
            # logger.info('p.label_ids: ', '\n', p.label_ids)
            return glue_compute_metrics(task_name, preds, p.label_ids)
        return compute_metrics_fn


    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info('*** Evaluate ***')

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == 'mnli':
            mnli_mm_data_args = dataclasses.replace(data_args, task_name='mnli-mm')
            eval_datasets.append(GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, evaluate=True))

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f'eval_results_{eval_dataset.args.task_name}.txt'
            )

            if trainer.is_world_master():
                with open(output_eval_file, 'w') as writer:
                    logger.info('***** Eval results {} *****'.format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info('  %s = %s', key, value)
                        writer.write('%s = %s\n' % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info('*** Test ***')
        test_datasets = [test_dataset]
        if data_args.task_name == 'mnli':
            mnli_mm_data_args = dataclasses.replace(data_args, task_name='mnli-mm')
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode='test', cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == 'classification':
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f'test_results_{test_dataset.args.task_name}.txt'
            )
            if trainer.is_world_master():
                with open(output_test_file, 'w') as writer:
                    logger.info('***** Test results {} *****'.format(test_dataset.args.task_name))
                    writer.write('index\tprediction\n')
                    for index, item in enumerate(predictions):
                        if output_mode == 'regression':
                            writer.write('%d\t%3.3f\n' % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write('%d\t%s\n' % (index, item))
    return eval_results


if __name__ == '__main__':
    main()
