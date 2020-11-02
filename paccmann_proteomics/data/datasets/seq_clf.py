import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union
import torch
from filelock import FileLock
from transformers import PreTrainedTokenizer, RobertaTokenizer, RobertaTokenizerFast, XLMRobertaTokenizer
from transformers.data.datasets import GlueDataset
from transformers.data.datasets import GlueDataTrainingArguments
from transformers.data.processors.glue import glue_convert_examples_to_features
from transformers.data.processors.utils import InputFeatures
from loguru import logger
from ..processors.seq_clf import seq_clf_output_modes, seq_clf_processors, seq_clf_tasks_num_labels


class Split(Enum):
    train = 'train'
    dev = 'dev'
    test = 'test'


class SeqClfDataset(GlueDataset):
    """
    Why this class even exists?
    `class GlueDataset(Dataset)` has  a constructor `def __init__()` with   
    `processor = glue_processors[args.task_name]()`, however I want to expand `glue_processors` 
    with protein clf task names. The line `processor = glue_processors[args.task_name]()` in parent 
    class doesn't accomodate this.
    """
    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = seq_clf_processors[args.task_name]()
        self.output_mode = seq_clf_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError('mode is not a valid split name')
        
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            'cached_{}_{}_{}_{}'.format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        label_list = self.processor.get_labels()
        if args.task_name in ['mnli', 'mnli-mm'] and tokenizer.__class__ in (
            RobertaTokenizer,
            RobertaTokenizerFast,
            XLMRobertaTokenizer,
            BartTokenizer,
            BartTokenizerFast,
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + '.lock'
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f'Loading features from cached file {cached_features_file} [took %.3f s]', time.time() - start
                )
            else:
                logger.info(f'Creating features from dataset file at {args.data_dir}')

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]

                # Load a data file into a list of ``InputFeatures``
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    'Saving features into cached file %s [took %.3f s]', cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list
