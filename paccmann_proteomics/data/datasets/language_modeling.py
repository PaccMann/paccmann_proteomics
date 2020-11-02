import logging
import os
import pickle
import time
import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from filelock import FileLock
from ..processors.lm_utils import split_into_chunks

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Basic dataset loader, supports fast tokenizers and caching
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDatasetCached(Dataset):
    """
    Similar to ``transformers.data.datasets.language_modeling.LineByLineTextDataset``,
    but additionally supports caching as seen in TextDataset class

    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False):
        assert os.path.isfile(file_path)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info("Loading features from cached file: {cached_features_file}")
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")
                start = time.time()
                with open(file_path, encoding="utf-8") as f:
                    lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
                logger.info(
                    f"Generated lines object [took %.3f s]. Now batch_encoding...", time.time()-start
                )
                
                start = time.time()
                # compatible with Tokenizers <= 0.7
                # batch_encoding = tokenizer.batch_encode_plus(
                #     lines, 
                #     add_special_tokens=True, 
                #     max_length=block_size,
                #     pad_to_max_length=True
                # )

                # new in Tokenizers 0.8 version
                batch_encoding = tokenizer(
                        lines_item, add_special_tokens=True, padding='longest',
                        truncation=True, max_length=block_size,
                    )

                self.examples = batch_encoding["input_ids"]
                logger.info(f"Batch encoding done [took %.3f s]", time.time() - start)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)



class LineByLineTextDatasetChunksCached(Dataset):
    """
    Similar to ``transformers.data.datasets.language_modeling.LineByLineTextDataset``,
    but additionally supports batch caching as seen in TextDataset class

    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, chunk_length: int, block_size: int, overwrite_cache=False):
        assert os.path.isfile(file_path)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )
        
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info("Loading features from cached file: {cached_features_file}")
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")
                start = time.time()

                # return `lines` list object
                with open(file_path, encoding="utf-8") as f:
                    lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
                logger.info(
                    f"Generated lines object [took %.3f s].", time.time()-start
                )

                # chunk the dataset into `lines_list` generator
                lines_list = split_into_chunks(chunk_length, lines)
                logger.info(
                    f"Dataset divided in %d line chunks for batch tokenizing. Total # of lines: %d. Expected # of chunks: %d", 
                    chunk_length, len(lines), 1+len(lines)/chunk_length
                )

                start_batch_encoding = time.time()
                self.examples = []
                for chunk_counter, lines_item in enumerate(lines_list, 1): 
                    start = time.time()

                    # compatible with Tokenizers <= 0.7
                    # returns transformers.tokenization_utils.BatchEncoding object
                    # batch_encoding = tokenizer.batch_encode_plus(
                    #     lines_item, 
                    #     add_special_tokens=True, 
                    #     max_length=block_size,
                    #     pad_to_max_length=True,
                    #     return_attention_masks=False,
                    # )
                    
                    # new in Tokenizers == 0.8
                    batch_encoding = tokenizer(
                        lines_item, add_special_tokens=True, padding='longest',
                        truncation=True, max_length=block_size,
                    )


                    self.examples.extend(batch_encoding["input_ids"])
                    logger.info(f"  Encoding batch %d complete! [took %.3f s]", chunk_counter, time.time() - start)

                logger.info(f"Batch encoding for all %d batches is done! [took %.3f s]", chunk_counter, time.time() - start_batch_encoding)

                # save to cache
                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )
    
        # for i in self.examples:
        #     print("example length, and tokens ids: ", len(i), i, '\n')


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)



class LineByLineTextDataset(Dataset):
    """
    Ingest dataset line-by-line and pad to ``block_size`` length
    Not padding at the batch level!
    Not caching features
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path)

        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
        # compatible with Tokenizers <= 0.7
        # batch_encoding = tokenizer.batch_encode_plus(
        #     lines, 
        #     add_special_tokens=True, 
        #     max_length=block_size,
        #     pad_to_max_length=True,
        #     return_overflowing_tokens=False)

        # new in Tokenizers == 0.8
        # from tokenization_utils_base.py:
        # The `pad_to_max_length` argument is deprecated and will be removed in a future version
        # use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or
        # use `padding='max_length'` to pad to a max length. In this case, you can give a specific
        # length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the
        # maximal input size of the model (e.g. 512 for Bert).
        batch_encoding = tokenizer(
                            lines, add_special_tokens=True, padding='longest',
                            truncation=True, max_length=block_size
        )
        
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)