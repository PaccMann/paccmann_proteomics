import argparse
import glob
from os.path import join
from loguru import logger
from tokenizers import CharBPETokenizer
"""
The original BPE tokenizer, as proposed in Sennrich, Haddow and Birch, Neural Machine Translation 
of Rare Words with Subword Units. ACL 2016
https://arxiv.org/abs/1508.07909
https://github.com/rsennrich/subword-nmt 
https://github.com/EdinburghNLP/nematus

BPE algorithm explanation:
    BPE first splits the whole sentence intoindividual characters. 
    The most frequent adjacent pairs of characters are then consecutively 
    merged until reaching a desired vocabulary size. Subword segmentation is 
    performed by applying the same merge operations to the test sentence.
    
    Frequent sub-strings will be joined early, resulting in common words 
    remaining as one unique symbol. Words consisting of rare character 
    combinations will be split into smaller units - substrings or characters

    For example, given a Dictionary with following word frequencies:
        ```
        5 low
        2 lower
        6 newest
        3 widest
        ```
    - a starting Vocabulary with all the characters is initialized:
    `{l,o,w,e,r,n,w,s,t,i,d}`
    - `es` is the most common 2-byte (two character) subsequence, it appears 9 times, so add it to vocab:
    `{l,o,w,e,r,n,w,s,t,i,d, es}`
    - `es t` is now the most common subseq, append it to Vocabulary too:
    `{l,o,w,e,r,n,w,s,t,i,d, es, est}`
    - then `lo` appears 7 times:
    `{l,o,w,e,r,n,w,s,t,i,d, es, est, lo}`
    - then `lo w`:
    `{l,o,w,e,r,n,w,s,t,i,d, es, est, lo, low}`
    - continue indefintitely until we reach a pre-defined vocabulary length

Example usage: 
python train_char_level_bpe.py --files /Users/flp/Box/Molecular_SysBio/data/paccmann/paccmann_proteomics/uniprot_sprot/uniprot_sprot_100_seq.txt --out /Users/flp/Box/Molecular_SysBio/data/paccmann/paccmann_proteomics/tokenized_uniprot_sprot/tests --vocab_size 

Important:
- Adds special end-of-word token (or a suffix) "</w>", 
e.g., word `tokenization` becomes [‘to’, ‘ken’, ‘ization</w>’]
- If needed, can limit initial alphabet size with `limit_alphabet: int`

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
    default='char-bpe', 
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
    default=100,
    type=int,
    help='The size of alphabet character set (e.g., for English, |alphabet|=26)',
)

args = parser.parse_args()

files = glob.glob(args.files)
if not files:
    logger.info(f'File does not exist: {args.files}')
    exit(1)


# Initialize an empty tokenizer
# ANY ARGS?
tokenizer = CharBPETokenizer()

# And then train
tokenizer.train(
    files,
    vocab_size=args.vocab_size,
    min_frequency=2,
    show_progress=True,
    special_tokens=['<unk>'],
    suffix='</w>',
    limit_alphabet=args.limit_alphabet,
)

# Save the files
tokenizer.save(args.out, args.name)

# Restoring model from learned vocab/merges
tokenizer = CharBPETokenizer(
    join(args.out, '{}-vocab.json'.format(args.name)),
    join(args.out, '{}-merges.txt'.format(args.name)),
)

# Test encoding
logger.info('Tokens and their ids from CharBPETokenizer with GFP protein sequence: \n MSKGEE LFTGVVPILVELDGDVNGHKFSVSGEGEG DAT')
encoded = tokenizer.encode('MSKGEE LFTGVVPILVELDGDVNGHKFSVSGEGEG DAT')
logger.info(encoded.tokens)
logger.info(encoded.ids)
logger.info('done!')
