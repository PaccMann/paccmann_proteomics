#!/usr/bin/env python3
"""
For a given file with one amino acid sequence and an index for the desired token mask location,
predict the masked out token.
"""
import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM, BertForMaskedLM
import os
import argparse
from pathlib import Path
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def get_tokenized_masked(aa_seq, mask):
    """
    Tokenize the input aa seq, masks with token 3, output a tensor with token indices
    ['MS', 'KGE', 'EL', 'FTG', 'VVP', 'IL', 'VEL', 'DG', 'DVNG', ..] 
    ['MS', 'KGE', 'EL', '[MASK]', 'VVP', 'IL', 'VEL', 'DG', 'DVNG', ..]
    [375, 861, 266, 3, 1210, 274, 641, 310, 25004, ..]
    tensor([[  375,   861,   266,  3,  1210,   274,   641,   310, 25004, ..]])
    """
    tokenized_text = tokenizer.tokenize(aa_seq)
    print('Masking token: {} at index {}'.format(tokenized_text[mask], mask))
    tokenized_text[mask] = '[MASK]'

    print('Tokenized and masked text is: ', tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    return torch.tensor([indexed_tokens])


def predict_mask(tokens_tensor, model, k):
    """
    Predict the masked token with BertForMaskedLM to predict a masked token
    Returns a vector of top k predictions for each mask
    `predictions` is a (1 x N_tokens x N_vocab) ten
    """
    
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    #describe(predictions) # 
    predicted_index_topk = torch.topk(predictions[0, args.masked_index], args.k)
    return tokenizer.convert_ids_to_tokens(predicted_index_topk[1])


def describe(x):
    """
    Helper function to describe a tensor object
    """
    print("------ BEGIN Describe tensor -----------------")
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))
    print("------ END Describe tensor -------------------")



if __name__ == '__main__':

    # parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_tokenizer', type=str, default='./models/BERT-uniprot')
    parser.add_argument('--dir_model', type=str, default='../../../Box/Molecular_SysBio/data/paccmann/paccmann_proteomics/roberta-uniprot-v3-100k')
    # parser.add_argument('--text_file', type=str, default='MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK')
    parser.add_argument('--text_file', type=str, default='./models/BERT-uniprot/aa_seq.txt')
    parser.add_argument('--masked_index', type=int, default=5)
    parser.add_argument('--k', type=int, default=10, help='number of top predicted tokens to be displayed')
    args = parser.parse_args()

    # load pre-trained tokenizer
    tokenizer = RobertaTokenizer(vocab_file=os.path.join(args.dir_tokenizer, 'vocab.json'), merges_file=os.path.join(args.dir_tokenizer, 'merges.txt'))

    # read in amino acid seaquence
    with open (args.text_file, 'r') as myfile:
        text=myfile.read()

    # tokenize aa sequence, mask selected tokens, and convert to a tensor
    print('----------- BEGIN --------------')
    tokens_tensor = get_tokenized_masked(aa_seq=text, mask=args.masked_index)

    # predict k best tokens for the masked token
    model_masked = RobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path=args.dir_model)
    model_masked.eval()
    predicted_token_topk = predict_mask(tokens_tensor, model_masked, args.k)

    print('Top {} predictions for masked index {} are: {}'.format(args.k, args.masked_index,predicted_token_topk))
    print('----------- END ---------------')