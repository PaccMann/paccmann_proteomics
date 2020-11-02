"""Utilities for working with protein embeddings."""
import torch
from transformers.tokenization_utils import PreTrainedTokenizer
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from tape.tokenizers import IUPAC_VOCAB


class EmbeddingToolkit(object):
    """Retrieve, preprocess and visualize embeddings
    
    Args:
        object ([type]): [description]
        language_model (transformers.modeling_roberta.RobertaModel): pre-trained language model
    """

    def __init__(self, model: nn.Module, tokenizer: PreTrainedTokenizer):
        """Initialize EmbeddingTools
        
        Args:
            model (nn.Module): pretrained model
            tokenizer (PreTrainedTokenizer): a tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    def get_embedding(self, manually_selected_vocab: list, export_to_csv: bool, export_filepath: str, return_full_vocab: bool) -> pd.core.frame.DataFrame:
        """Get embeddings for a corresponding list of tokenizer vocabulary items as a Pandas dataframe object"
        
        Args:
            model (nn.Module): [description]
            manually_selected_vocab (list): desired amino acid vocabulary, corresponds to keys in a dict returned from calling tokenizer.get_vocab()
            return_full_vocab (bool): whether to return an embedding for all tokens, or just the ones from manually_selected_vocab
        Returns:
            pandas.core.frame.DataFrame: [description]

        Example:
            manually_selected_vocab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            # Create model and tokenizer objects
            tokenizer = RobertaTokenizer.from_pretrained(dir_tokenizer)
            model = RobertaModel.from_pretrained(pretrained_model_name_or_path=dir_model)

            Toolkit = EmbeddingToolkit(model=model, tokenizer=tokenizer)
            df = Toolkit.get_embedding(manually_selected_vocab=manually_selected_vocab)
            logger.info(df)
                      0         1         2         3  ...        766       767  
            A -0.012096 -0.042020 -0.027793 -0.006360  ...   0.001138 -0.000997 
            B  0.018540  0.060982  0.055752  0.012910  ...   0.049360  0.013828
            C -0.003167  0.001412 -0.026587 -0.040021  ...  -0.033149 -0.006456
                .          .        .         .        ...    .          .
                .          .        .         .        ...    .          .
            Y 0.026067	 0.002919 -0.032527	 0.025508		-0.018694  0.037993
            Z -0.002928	 0.016255  0.033822 -0.028604        0.000767  -0.035366
        """
        # logger.info('Retrieving embeddings for ', manually_selected_vocab, '\n')
        embedding = self.model.get_input_embeddings()
        tokens = self.tokenizer.get_vocab()  # could be moved outside to speed up, if the dict is huge

        if return_full_vocab == True:
            tokens_aa = tokens
            logger.info('Returning embeddings for all %d vocabulary tokens', len(tokens))
        else:
            tokens_aa = {k: tokens[k] for k in manually_selected_vocab}  # dict {'A': 37, 'B': 38, ..., 'Y': 61, 'Z': 62}

        embedded_tokens_df = pd.DataFrame(data = [embedding.weight[token].tolist() for token in tokens_aa.values()], index=tokens_aa.keys())
        # logger.info('Head and Tail of embeddings dataframe: ')
        # logger.info(embedded_tokens_df.head(), '\n')
        logger.debug(embedded_tokens_df.tail())
        
        if export_to_csv == True:
            embedded_tokens_df.to_csv(path_or_buf=export_filepath)
            logger.info('Exported model to: %s', export_filepath)
        
        return embedded_tokens_df
    
    
    def get_tsne(self, embedding_df: pd.core.frame.DataFrame, tsne_dim: int, export_to_csv: bool, export_filepath: str) -> pd.core.frame.DataFrame:
        """Compresses high dimensional word embedding into `tsne_dim` embedding, and return a pandas df. Used for visualization only.
        recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD for sparse data) 
        to reduce the number of dimensions to a reasonable amount (e.g. 50) if the number of embedding dimensions is very high.
        
        Args:
            embedding_df (pd.core.frame.DataFrame): Dataframe with embeddings of size [number_amino_acids x embedding_dimension]
            tsne_dim (int): see more at https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        
        Returns:
            pd.core.frame.DataFrame: a dataframe with t-sne of embeddings, dimensions [number_amino_acids x tsne_dim]
                0          1
            A  13.062521   9.171124
            C  36.266224 -11.948713
            D -36.986889  14.661242
            .   .           .
            .   .           .
            Y -26.306509 -23.310379
            Z  -3.202672  35.785797
        """
        tsne_result = TSNE(
            n_components=tsne_dim, 
            perplexity=5, 
            min_grad_norm=1E-7, 
            n_iter=250,
            learning_rate=40,
            init='pca',
            verbose=1
            ).fit_transform(embedding_df)
        
        df = pd.DataFrame(data=tsne_result, index=embedding_df.index)
        logger.debug(embedded_tokens_df.tail())

        if export_to_csv == True:
            df.to_csv(path_or_buf=export_filepath)

            logger.info('Exported TSNE model to: ', export_filepath, '\n')
        return df


    def plot_tsne(self, df_tsne: pd.core.frame.DataFrame, save_figure=True, figure_path='figures/aa_embedding_by_property_tsne.pdf'):
        """Plots t-sne (or any 2-dimensional graph) of aa embeddings, returns a pdf at `figure_path`
        
        Args:
            df_tsne (pd.core.frame.DataFrame): 2-dimensional 
            figure_path (str, optional): [description]. Defaults to 'figures/aa_embedding_by_property.pdf'.
        """
        aa_family = pd.DataFrame(data=list(IUPAC_VOCAB.values()), index=list(IUPAC_VOCAB.keys()), columns=['property'] )

        df_tsne = df_tsne.merge(aa_family, how='inner', left_index=True, right_index=True)
        logger.info(df_tsne)
        # function to add amino acid characters next to the datapoints in the plot
        def label_point(x, y, val, ax):
            a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
            logger.info(a)
            for i, point in a.iterrows():
                ax.text(point['x']+.1, point['y']+.1, str(' '+point['val']))

        logger.info('list(df_tsne.columns)[1] is: ', list(df_tsne.columns)[1])
        # ax = sns.scatterplot(x=list(df_tsne.columns)[1], y=list(df_tsne.columns)[2], 

        ax = sns.scatterplot(x=df_tsne['0'], y=df_tsne['1'], 
                            # hue=list(df_tsne.index), 
                            hue=list(df_tsne.property), 
                            data=df_tsne, 
                            legend='brief')

        plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0)
        plt.title('t-SNE plot of 20 embedded amino acids (colored by chemical family)', fontweight='bold')
        # Set x-axis label
        plt.xlabel('t-SNE dim 0')
        # Set y-axis label
        plt.ylabel('t-SNE dim 1')
        
        # DEBUG
        logger.info(list(df_tsne.index))

        # label_point(x=df_tsne['0'], y=df_tsne['1'], val=df_tsne['index'], ax=plt.gca())  
        label_point(x=df_tsne['0'], y=df_tsne['1'], val=df_tsne.index.to_series(), ax=plt.gca())

        if save_figure:
            plt.savefig(figure_path, bbox_inches='tight', pad_inches=2)
            logger.info('Successfully saved t-sne plot at: ', figure_path)
        else:
            plt.show()


    def plot_dendrogram(self, embedding_df: pd.core.frame.DataFrame, linkage_method: str, linkage_metric: str,
                        figsize=(20, 10), save_figure=True,
                        figure_path='figures/aa_embedding_by_property_dendrogram.pdf'):
        """
        Perform hierarchical clustering on a 2D dataset (shape: obs x features) with 
        single, complete, average (UPGMA), weighted, centroid or Ward linkage distance methods.
        Metrics: see scipy.spatial.distance.pdist https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist

        Is a wrapper of:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy-cluster-hierarchy-linkage
        
        Args:
            embeding_df (pd.core.frame.DataFrame): [description]
            linkage_method (str): single, complete, average (UPGMA), weighted, centroid or Ward linkage distance methods
            linkage_metric (str): euclidian, minkowski, manhattan
            figure_path (str, optional): Where to save the figure. Defaults to 'figures/aa_embedding_by_property_dendrogram_'+linkage_method+'_'+linkage_metric+'.pdf'.
        """
        aa_property_labels = (';'.join(x + ', ' + y for x, y in IUPAC_VOCAB.items())).split(';')
        Z = linkage(embedding_df, method=linkage_method, metric=linkage_metric)
        fig = plt.figure(figsize=figsize)
        dn = dendrogram(Z, labels=aa_property_labels)

        if save_figure == True:
            plt.savefig(figure_path, bbox_inches='tight', pad_inches=2)
            logger.info('Successfully saved dendrogram at: ', figure_path)
        else:
            plt.show()
        plt.close(fig)
