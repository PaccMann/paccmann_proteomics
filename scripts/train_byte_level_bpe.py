import argparse
import glob
from os.path import join
from loguru import logger
from tokenizers import ByteLevelBPETokenizer
from tokenizers import pre_tokenizers
"""
Byte-Level BPE as used in GPT-2 (Radford 2019 Language Models are Unsupervised Multitask Learners 
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
see `Section 2.2. Input Representation`,
and RoBERTa (Liu 2018  RoBERTa: A Robustly Optimized BERT Pretraining Approach, https://arxiv.org/abs/1907.11692)
see `Section 4.4 Text Encoding`

Algorithm explanation:
    BPE vocabulary sizes typically range from 10K-100K subword units. However, unicode characters 
    can account for a sizeable portion of this vocabulary when modeling large and diverse corpora, 
    like Wiki, news, reddit. Radford et al. (2019) introduce a clever implementation of BPE that 
    uses bytes instead of unicode characters as the base subword units. 
    Using bytes makes it possible to learn a subword vocabulary of a modest size (50K units) 
    that can still encode any input text without introducing any '<unk>' tokens.

Important:
    - 'Ġ' token is used to delimit word prefix: 
    ['MSKGEE LFTGVVPILVELDGDVNGHKFSVSGEGEG DAT']
	['ĠM', 'SK', 'GEE', 'Ġ', 'LF', 'TG', 'VV', 'PI', 'LVE', 'LD', 'GDV', 'N', 'GH', 'KF', 'SV', 'SGE', 'GE', 'G', 'Ġ', 'D', 'AT']
    - Set `add_prefix_space` flag to True, to conserve the absence of a space at 
    the beginning of a string
    - No need to set '<unk>' tokens, but we'll set it just in case
    - '<s>' and '</s>' are begining and end of sequence tokens
    - Initial alphabet: it will have 256 characters; see in `ByteLevelBPETokenizer` class `initial_alphabet` is set to 
    `pre_tokenizers.ByteLevel.alphabet()`,
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    '--files',
    default=None,
    metavar='path',
    type=str,
    required=True,
    help='The files to use as training; accept `**/*.txt`you type of patterns if enclosed in quotes',
)
parser.add_argument(
    '--out',
    #default='./',
    type=str,
    required=True,
    help='Path to the output directory, where the files will be saved',
)
parser.add_argument(
    '--name', 
    default='byte-level-bpe', 
    type=str, 
    help='The name of the output vocab files'
)
parser.add_argument(
    '--vocab_size', 
    default=30000, 
    type=int,
    required=True,
    help='Vocabulary size',
)
parser.add_argument(
    '--empty_initial_alphabet', 
    default=False, 
    type=bool,
    required=False,
    help="By default BPE has 256 char alphabet, most of such chars don't appear in proteins",
)

args = parser.parse_args()

files = glob.glob(args.files)
if not files:
    logger.info(f'File does not exist: {args.files}')
    exit(1)


# Initialize an empty tokenizer
tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)

# Initial alphabet with 256 chars or much less?
if args.empty_initial_alphabet:
    alphabet = []
else:
    alphabet = pre_tokenizers.ByteLevel.alphabet()  # 256 chars

logger.info('Initial alphabet for ByteLevel BPE as defined in pre_tokenizers.ByteLevel.alphabet(): ', alphabet)
# And then train
tokenizer.train(
    files,
    vocab_size=args.vocab_size,
    min_frequency=2,
    show_progress=True,
    special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'],
)

# Save the files
tokenizer.save(args.out, args.name)

# Restoring model from learned vocab/merges
tokenizer = ByteLevelBPETokenizer(
    join(args.out, '{}-vocab.json'.format(args.name)),
    join(args.out, '{}-merges.txt'.format(args.name)),
    add_prefix_space=True,
)

# Test encoding
logger.info('Tokens and their ids from ByteLevelBPETokenizer with GFP protein sequence: \n MSKGEE LFTGVVPILVELDGDVNGHKFSVSGEGEG DAT')
encoded = tokenizer.encode('MSKGEE LFTGVVPILVELDGDVNGHKFSVSGEGEG DAT', pad_to_max_length=True)
logger.info(encoded.tokens)
logger.info(encoded.ids)
logger.info('done!')
