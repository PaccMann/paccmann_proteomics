""" 
Processors for protein sequence tasks, based on GLUE processors and helpers
"""
import os
from enum import Enum
from typing import List, Optional, Union
from transformers import PreTrainedTokenizer
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers.data.processors.glue import OutputMode, glue_convert_examples_to_features
from loguru import logger
from .seq_clf_utils import ProteinDataProcessor


class LocalizationProcessor(ProteinDataProcessor):
    """
    Processor for the Localization dataset in `.fa` format, originally from Armenteros 2017 paper
    "Deeploc: prediction of protein subcellular localization using deep learning. Bioinformatics"
    """

    def get_train_examples(self, data_dir) -> List[InputExample]:    
        train_examples = self._create_examples(self._read_fasta_localization(os.path.join(data_dir, 'train.fa')), 'train')
        return train_examples

    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """
        Gets a collection (list) of `InputExample`s for the dev set
        
        Args:
            data_dir (str): where data `.tsv` files are located

        Returns:
            List[InputExample]: list with all examples from dev dataset, in `InputExample` format.
        """
        dev_examples = self._create_examples(lines=self._read_fasta_localization(os.path.join(data_dir, 'dev.fa')), set_type='dev')
        return dev_examples

    def get_labels(self):
        """See base class."""
        return ['Nucleus', 'Cytoplasm', 'Extracellular', 'Mitochondrion', 'Cell.membrane', 'Endoplasmic.reticulum', 'Plastid', 'Golgi.apparatus', 'Lysosome/Vacuole', 'Peroxisome']


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = '%s-%s' % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class SolubilityProcessor(ProteinDataProcessor):
    """
    Processor for the Solubility data set in `.fa` format, originally from Khurana 2018 paper
    "Deepsol: a deep learning framework for sequence-based protein solubility"
    """

    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Gets a collection (list) of `InputExample`s for the train set

        Args:
            data_dir (str): where training data `.fa` files are located

        Returns:
            List[InputExample]: list with all examples from train dataset, in `InputExample` format.
        """        
        train_examples = self._create_examples(self._read_fasta_tab(os.path.join(data_dir, 'train.fa')), 'train')
        return train_examples

    def get_dev_examples(self, data_dir) -> List[InputExample]:
        dev_examples = self._create_examples(lines=self._read_fasta_tab(os.path.join(data_dir, 'dev.fa')), set_type='dev')
        return dev_examples

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = '%s-%s' % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class PairwiseInteractionProcessor(DataProcessor):
    """Processor for the PairwiseInteraction dataset, similar to MRPC data set in GLUE"""

    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict['idx'].numpy(),
            tensor_dict['sentence1'].numpy().decode('utf-8'),
            tensor_dict['sentence2'].numpy().decode('utf-8'),
            str(tensor_dict['label'].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info('Getting train examples from: {}'.format(os.path.join(data_dir, 'train.tsv')))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'train.tsv')), 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'dev.tsv')), 'dev')

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[4]
            text_b = line[5]
            label = line[3]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class PairwiseStringProcessor(ProteinDataProcessor):
    """
    Processor for the String PPI dataset, similar to Pairwise data set, but uses .txt files
    instead of .tsv, and label, text_a and text_b are at positions 2, 3 and 4 
    """

    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict['idx'].numpy(),
            tensor_dict['sentence1'].numpy().decode('utf-8'),
            tensor_dict['sentence2'].numpy().decode('utf-8'),
            str(tensor_dict['label'].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info('Getting train examples from: {}'.format(os.path.join(data_dir, 'train.tsv')))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'train.tsv')), 'train')
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'dev.tsv')), 'dev')

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = '%s-%s' % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RemoteHomologyProcessor(ProteinDataProcessor):
    """
    Processor for the Localization dataset in `.fa` format, originally from Armenteros 2017 paper
    "Deeploc: prediction of protein subcellular localization using deep learning. Bioinformatics"
    """

    def get_train_examples(self, data_dir) -> List[InputExample]:    
        train_examples = self._create_examples(self._read_fasta_remote_homology(os.path.join(data_dir, 'train.fasta')), 'train')
        return train_examples

    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """
        Gets a collection (list) of `InputExample`s for the dev set
        
        Args:
            data_dir (str): where data `.tsv` files are located

        Returns:
            List[InputExample]: list with all examples from dev dataset, in `InputExample` format.
        """
        dev_examples = self._create_examples(lines=self._read_fasta_remote_homology(os.path.join(data_dir, 'dev.fasta')), set_type='dev')
        return dev_examples

    def get_labels(self):
        """See base class. List of labels stores items of type int, but model accepts type str"""
        labels = list(range(0, 1195))

        return list(map(str, labels))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = '%s-%s' % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples



seq_clf_tasks_num_labels = {
    'cola': 2,
    'mnli': 3,
    'mrpc': 2,
    'sst-2': 2,
    'solubility': 2,
    'localization': 10,
    'remote-homology': 1195,
    'pairwise-interaction': 2,
    'pairwise-string': 2,
    'sts-b': 1,
    'qqp': 2,
    'qnli': 2,
    'rte': 2,
    'wnli': 2,
}

seq_clf_processors = {
    'solubility': SolubilityProcessor,
    'localization': LocalizationProcessor,
    'remote-homology': RemoteHomologyProcessor,
    'pairwise-interaction': PairwiseInteractionProcessor,
    'pairwise-string': PairwiseStringProcessor,
}

seq_clf_output_modes = {
    # protein classification tasks
    'solubility': 'classification',
    'localization': 'classification',
    'remote-homology': 'classification',
    'pairwise-interaction': 'classification',
    'pairwise-string': 'classification',
    # original GLUE tasks
    'cola': 'classification',
    'mnli': 'classification',
    'mnli-mm': 'classification',
    'mrpc': 'classification',
    'sst-2': 'classification',
    'sts-b': 'regression',
    'qqp': 'classification',
    'qnli': 'classification',
    'rte': 'classification',
    'wnli': 'classification',
}
