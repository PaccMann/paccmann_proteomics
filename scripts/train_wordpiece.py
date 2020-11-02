import argparse
import glob
from loguru import logger
from tokenizers import BertWordPieceTokenizer
"""
Wordpiece as used by Google to train BERT and DistilBERT

Algorithm:
    - convert all the input into unicode characters (language/character agnostic)
    Similar to BPE and uses frequency occurrences to identify potential merges,
    but makes the final decision based on the likelihood of the merged token.

Important:
    - prepends a word prefix '##' (`wordpieces_prefix`) for sub-words of less common (unknown in the Vocabulary) words,
    e.g., `hypatia = h ##yp ##ati ##a`
    - support for Chinese characters
    - special tokens: `"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"`
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files",
    default=None,
    metavar="path",
    type=str,
    required=True,
    help="The files to use as training; accept `**/*.txt` type of patterns if enclosed in quotes",
)
parser.add_argument(
    "--out",
    default="./",
    type=str,
    help="Path to the output directory, where the files will be saved",
)
parser.add_argument(
    "--name", default="bert-wordpiece", 
    type=str, 
    help="The name of the output vocab files"
)
parser.add_argument(
    '--vocab_size', 
    default=30000, 
    type=int,
    required=True,
    help='Vocabulary size',
)
args = parser.parse_args()

files = glob.glob(args.files)
if not files:
    logger.info(f"File does not exist: {args.files}")
    exit(1)

# CHINESE CHARACTERS???!!!
# Initialize an empty tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True, handle_chinese_chars=False, 
    strip_accents=True, lowercase=True,
)

# And then train
trainer = tokenizer.train(
    files,
    vocab_size=10000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=1000,
    wordpieces_prefix="##",
)

# Save the files
tokenizer.save(args.out, args.name)


# Restoring model from learned vocab/merges
tokenizer = BertWordPieceTokenizer(vocab_file=
    join(args.out, '{}-vocab.txt'.format(args.name)),
    prefix=wordpieces_prefix
)

# Test encoding
logger.info('Testing BertWordPieceTokenizer with GFP protein sequence: \n MSKGEE LFTGVVPILVELDGDVNGHKFSVSGEGEG DAT')
encoded = tokenizer.encode('MSKGEE LFTGVVPILVELDGDVNGHKFSVSGEGEG DAT')
logger.info(encoded.tokens)
logger.info(encoded.ids)
logger.info('done!')
