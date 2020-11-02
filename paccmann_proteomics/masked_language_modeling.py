"""Utilities for masked language modeling."""
import torch
import torch.nn as nn
from transformers.tokenization_utils import PreTrainedTokenizer
from loguru import logger


class MaskedLanguage(object):
    """Masked language."""

    def __init__(self, tokenizer: PreTrainedTokenizer, model: nn.Module):
        """
        Initialize the masked language.
        
        Args:
            tokenizer (PreTrainedTokenizer): a tokenizer.
            model (nn.Module): a model for language modeling.
        """
        self.tokenizer = tokenizer
        self.lm_model = model

    def get_tokenized_masked(self, a_string: str, mask_index: int) -> torch.tensor:
        """
        Get a tensor from a tokenized string where a mask is applied.
        
        Args:
            a_string (str): a string to tokenize.
            mask_index (int): index of the mask in the tokenized string.
        
        Returns:
            torch.tensor: the masked tokenized string.
        """
        tokenized_text = self.tokenizer.tokenize(a_string)
        logger.debug('Masking token: {} at index {}'.format(tokenized_text[mask_index], mask_index))
        tokenized_text[mask_index] = '[MASK]'
        logger.debug('Tokenized and masked text is: ', tokenized_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return torch.tensor([indexed_tokens])
    

    def predict_masked(self, tokenized_masked_tensor: torch.tensor, lm_model, 
        mask_index: int, k: int) -> list:
        """
        Predict `k` most likely tokens for a masked-out protein sequence at positions `mask_index`
        """            
        lm_model.eval()
        with torch.no_grad():
            outputs = lm_model(tokenized_masked_tensor)
            predictions = outputs[0]

        # describe predictions
        predicted_index_topk = torch.topk(predictions[0, mask_index], k)
        return self.tokenizer.convert_ids_to_tokens(predicted_index_topk[1])
