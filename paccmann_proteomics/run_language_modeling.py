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
Pre-training or Fine-tuning transformer models.
"""
import math
import os
import glob
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from transformers import (
    CONFIG_MAPPING,
    # mapping containing all the PyTorch models that have an LM head; 
    # OrderedDict([(ModelConfig, ModelForMaskedLM), (RobertaConfig, RobertaForMaskedLM), ..])
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    # preprocess batches of tensors for MLM
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from loguru import logger
from paccmann_proteomics.data.datasets.language_modeling import (
    LineByLineTextDatasetCached, LineByLineTextDatasetChunksCached, LineByLineTextDataset
)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())  # [ModelConfig, RobertaConfig, ...]
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)  # ('roberta', 'bert', ...)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.'
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={'help': 'If training from scratch, pass a model type from the list: ' + ', '.join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={'help': 'Pretrained config name or path if not the same as model_name'}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={'help': 'Where to store the pretrained models downloaded from S3'}
    )
    continue_from_checkpoint: bool = field(
        default=False,
        metadata={'help': 'Whether to continue training from `model_name_or_path/checkpoint-<Trainer.global_step>/`'},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={'help': 'The input training data file (a text file).'}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={'help': 'An optional input evaluation data file to evaluate the perplexity on (a text file).'},
    )
    line_by_line: bool = field(
        default=False,
        metadata={'help': 'Whether distinct lines of text in the dataset are to be handled as distinct sequences.'},
    )
    chunk_length: Optional[int] = field(
        default=1000000, metadata={'help': 'Length of chunks when batch tokenizing the dataset.'}
    )
    mlm: bool = field(
        default=False, metadata={'help': 'Train with masked-language modeling loss instead of language modeling.'}
    )
    mlm_probability: float = field(
        default=0.15, metadata={'help': 'Ratio of tokens to mask for masked language modeling loss'}
    )

    block_size: int = field(
        default=-1,
        metadata={
            'help': 'Optional input sequence length after tokenization.'
            'The training dataset will be truncated in block of this size for training.'
            'Default to the model max input length for single sentence inputs (take into account special tokens).'
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={'help': 'Overwrite the cached training and evaluation sets'}
    )


def get_dataset(args: DataTrainingArguments, 
                tokenizer: PreTrainedTokenizer, 
                evaluate=False
                ) -> 'Optional[Dataset]':
    """Reads in the dataset depending on its original text formatting, tokenizes it and saves to cache.

    Args:
        args (DataTrainingArguments): specify how text is formatted (line by line, or uniform), 
            whether to save cache or not, etc.
        tokenizer (PreTrainedTokenizer): tokenizer object
        evaluate (bool, optional): preprocesses evaluation (dev) file. Defaults to False.

    Returns:
        Optional[Dataset]: batch tokenized (encoded) torch.Dataset with token IDs
    """
    if evaluate:
        file_path = args.eval_data_file
    else:
        file_path = args.train_data_file

    if args.line_by_line:
        # return LineByLineTextDatasetChunksCached(
        return LineByLineTextDatasetChunksCached(
            tokenizer=tokenizer,
            file_path=file_path, 
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
            chunk_length=args.chunk_length,
        )
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, 
            block_size=args.block_size, 
            overwrite_cache=args.overwrite_cache
        )
    

def _sorted_checkpoints(
    model_dir, checkpoint_prefix="checkpoint", use_modification_time=False
) -> List[str]:
    """
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
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            'Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file '
            'or remove the --do_eval argument.'
        )

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


    # Load pretrained language model
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
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning('You are instantiating a new config instance from scratch.')


    # Load tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, 
            cache_dir=model_args.cache_dir, 
            use_fast=True, 
            add_prefix_space=False,
            model_max_length = data_args.block_size
        )
        logger.info('Tokenizer dir was specified. The maximum length (in number of tokens) for'+
                    'the inputs to the transformer model, `model_max_length` is: %d', 
                    tokenizer.model_max_length
        )

    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, 
            cache_dir=model_args.cache_dir, 
            use_fast=True, 
            add_prefix_space=False,
            model_max_length = data_args.block_size
        )
        logger.info('Tokenizer dir was NOT specified. The maximum length (in number of tokens) for'+ 
                    'the inputs to the transformer model, `model_max_length` is: %d', 
                    tokenizer.model_max_length
        )
    else:
        raise ValueError(
            'You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,'
            'and load it from here, using --tokenizer_name'
        )
     

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool('.ckpt' in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info('Training new model from scratch')
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ['bert', 'roberta', 'distilbert', 'camembert'] and not data_args.mlm:
        raise ValueError(
            'BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm '
            'flag (masked language modeling).'
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)
    logger.info('Param check: block_size is: %d', data_args.block_size)

    
    # Get datasets
    logger.info('Preparing training dataset from: %s', data_args.train_data_file)
    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None

    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
    )

    # Initialize our Trainer
    model = model.to(training_args.device)  # BUG https://github.com/huggingface/transformers/issues/4240
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=False,  # in evaluation and prediction, only return the loss
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info('*** Evaluate ***')

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output['eval_loss'])
        result = {'perplexity': perplexity}

        output_eval_file = os.path.join(training_args.output_dir, 'eval_results_lm.txt')
        if trainer.is_world_master():
            with open(output_eval_file, 'w') as writer:
                logger.info('***** Eval results *****')
                for key in sorted(result.keys()):
                    logger.info('  %s = %s', key, str(result[key]))
                    writer.write('%s = %s\n' % (key, str(result[key])))

        results.update(result)

    return results


if __name__ == '__main__':
    main()
