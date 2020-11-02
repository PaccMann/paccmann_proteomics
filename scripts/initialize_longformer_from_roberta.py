import os
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from transformers import RobertaForMaskedLM, RobertaTokenizerFast 
from transformers import TrainingArguments, HfArgumentParser
from transformers.modeling_longformer import LongformerSelfAttention
from loguru import logger
from paccmann_proteomics.data.datasets.language_modeling import LineByLineTextDatasetChunksCached


# RobertaLongForMaskedLM represents the "long" version of the RoBERTa model. It replaces 
# BertSelfAttention with RobertaLongSelfAttention, which is a thin wrapper around 
# LongformerSelfAttention
class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)


class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)



def convert_to_long_model(model_name, tokenizer_name, save_model_to, attention_window, max_pos):
    """
    Starting from the roberta-base checkpoint, the following function converts it into an instance 
    of RobertaLong.

    Args:
        save_model_to (str): path to output dir
        attention_window (int): 
        max_pos (int): max model position before adding extra 2 tokens for roberta models

    Returns:
        transformers.RobertaForMaskedLM: RoBERTa model with LM head on top
    """
    model = RobertaForMaskedLM.from_pretrained(model_name)
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name, model_max_length=max_pos)
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos

    # allocate a larger position embedding matrix
    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = layer.attention.self.query
        longformer_self_attn.key_global = layer.attention.self.key
        longformer_self_attn.value_global = layer.attention.self.value

        layer.attention.self = longformer_self_attn

    logger.info(f'      saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer


def update_global_projection_layers(model):
    """
    Pretraining on Masked Language Modeling (MLM) doesn't update the global projection layers. 
    After pretraining, the following function copies query, key, value to their global counterpart 
    projection matrices. For more explanation on "local" vs. "global" attention, please refer to 
    the documentation:
    `https://huggingface.co/transformers/model_doc/longformer.html#longformer-self-attention`

    Args:
        model (transformers.RobertaForMaskedLM): RoBERTa model with LM head on top

    Returns:
        model (transformers.RobertaForMaskedLM): RoBERTa model with LM head on top, this time with
        fixed projection layers
    """
    for i, layer in enumerate(model.roberta.encoder.layer):
        layer.attention.self.query_global = layer.attention.self.query
        layer.attention.self.key_global = layer.attention.self.key
        layer.attention.self.value_global = layer.attention.self.value
    return model



@dataclass
class ModelArguments:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=2048, metadata={"help": "Maximum position"})
    model_name_or_path: str = field(
        default=None, 
        metadata={"help": "Path to pretrained RoBERTa model"}
    )
    tokenizer_name: str = field(
        default=None,
        metadata={"help": "Path to pretrained BPE tokenizer"}
    )
    copy_projection_layers: bool = field(
        default=False,
        metadata={"help": "Whether to copy local projection layers to global ones"},
    )


def main():

    parser = HfArgumentParser((TrainingArguments, ModelArguments))
    training_args, model_args = parser.parse_args_into_dataclasses()


    # As descriped in convert_to_long_model, convert a roberta-base model into roberta-base-2048 which is
    # an instance of RobertaLong, then save it to the disk
    model_path = f'{training_args.output_dir}/roberta{model_args.max_pos}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.info(f'Converting roberta-base into roberta{model_args.max_pos}')
    model, tokenizer = convert_to_long_model(
        model_name = model_args.model_name_or_path, 
        tokenizer_name = model_args.tokenizer_name,
        save_model_to = model_path,
        attention_window = model_args.attention_window, 
        max_pos = model_args.max_pos
    )

    # Load roberta-base-2048 from the disk. This model works for long sequences even without pretraining
    # If you don't want to pretrain, you can stop here and start finetuning your roberta-base-4096 
    # on downstream tasks
    logger.info(f'Loading the model from {model_path}')
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = RobertaLongForMaskedLM.from_pretrained(model_path)


    if model_args.copy_projection_layers:
        # Copy global projection layers. MLM pretraining doesn't train global projections, so we need to
        # call update_global_projection_layers to copy the local projection layers to the global ones
        logger.info(f'  Copying local projection layers into global projection layers ... ')
        model = update_global_projection_layers(model)

        model_path = f'{training_args.output_dir}/roberta{model_args.max_pos}-with-global-projections'
        logger.info(f'      Saving model after copying global projection layers to {model_path}')
        model.save_pretrained(model_path)


if __name__ == "__main__":
    main()
