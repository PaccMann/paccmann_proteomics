import argparse
import glob
from os.path import join
from loguru import logger
from tokenizers import SentencePieceBPETokenizer
"""
SentencePiece BPE Tokenizer 
as outlined in Kudo 2018 Subword Regularization: Improving Neural Network Translation Modelswith Multiple Subword Candidates

The central idea is to `virtually augment training data with on-the-fly subword  sampling`, 
which  helps to improve the accuracy as well as robustness of NMT models. 
For better subword sampling they use the unigram language model, which unlike the greedy BPE approach 
(takes two tokens, looks at the frequency of each pair and then merges the pairs that have 
the highest combined frequency count) chooses the most likely likely combination. 
    
Algorithm is performed in Expectation Maximization (EM) setting: 
    0) convert all the input into unicode, even spaces (as underscores, '_')
    1) calculate probabilities (frequency-based) of each subword token (can seed the subword token set with BPE)
    2) with EM estimate a loss which would result if each subword token was discarded
    3) discard tokens with the largest loss (can adjust the fraction of the worst tokens to drop with param )
    <-- insert fraction param
    4) repeat steps 1-3 until reached final vocabulary size or until there is no change in token numbers after successive iterations

Pecularities:
    - spaces encoded as "_", or symbol U+2581
"""


parser = argparse.ArgumentParser()
parser.add_argument(
    '--files',
    default=None,
    metavar='path',
    type=str,
    required=True,
    help='The files to use as training; accept a string in format `"**/*.txt"`'
)
parser.add_argument(
    '--out',
    # default='./',
    type=str,
    required=True,
    help='Path to the output directory, where the files will be saved'
)
parser.add_argument(
    '--name', 
    default='sentencepiece', 
    type=str, 
    help='The name of the output vocab files',
)
parser.add_argument(
    '--vocab_size', 
    default=30000, 
    type=int,
    required=True,
    help='Vocabulary size',
)
parser.add_argument(
    '--limit_alphabet',
    default=1000,
    type=int,
    help='The size of alphabet character set (e.g., for English, |alphabet|=26)',
)

args = parser.parse_args()

files = glob.glob(args.files)
if not files:
    logger.info(f'File does not exist: {args.files}')
    exit(1)


# Initialize an empty tokenizer
tokenizer = SentencePieceBPETokenizer(add_prefix_space=True)

# And then train
tokenizer.train(
    files,
    vocab_size=args.vocab_size,
    min_frequency=2,
    show_progress=True,
    special_tokens=['<unk>'],
    limit_alphabet=1000
)

# Save the files
tokenizer.save(args.out, args.name)

# Restoring model from learned vocab/merges
tokenizer = SentencePieceBPETokenizer(
    join(args.out, '{}-vocab.json'.format(args.name)),
    join(args.out, '{}-merges.txt'.format(args.name)),
    add_prefix_space=True
)

# Test encoding
logger.info('Tokens and their ids from SentencePiece with GFP protein sequence: \n MSKGEE LFTGVVPILVELDGDVNGHKFSVSGEGEG DAT')
encoded = tokenizer.encode('MSKGEE LFTGVVPILVELDGDVNGHKFSVSGEGEG DAT')
logger.info(encoded.tokens)
logger.info(encoded.ids)
logger.info('done!')
