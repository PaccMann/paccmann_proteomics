import dataclasses
import json
import csv
from dataclasses import dataclass
from typing import List, Optional, Union
import torch
from torch.utils.data import TensorDataset
from transformers.data.processors.utils import DataProcessor


class ProteinDataProcessor(DataProcessor):
    """Enhanced DataProcessor class with Fasta parsers"""
    
    @classmethod
    def _read_irregular_whitespace_delim(cls, input_file, quotechar=None):
        """
        Reads an irregular whitespace separated value file.
        Irregular whitespaces can appear when exporting pd.DataFrame to .txt with delim='\t' 
        """
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            output_list = []
            for line in f:
                row = line.split()
                output_list.append(row)
            return output_list

    @classmethod
    def _read_fasta_tab(cls, input_file):
        """Reads a fasta file, label is in fasta header and preceded by '\t'."""

        def parse_fasta(file):
            with open(file, 'r') as f:
                # Create variables for storing the identifiers and the sequence.
                label = ''
                sequence = ''
                for line in f:
                    line = line.strip()  # Remove trailing newline characters.
                    if line.startswith('>'):
                        # Split on tab char that precedes the seq label
                        line = line.split('\t')
                        line = line[1]

                        # Start by yielding the entry that has been built up.
                        yield sequence, label

                        # Reinitialise the identifier and sequence variables to build up a new record.
                        label = line
                        sequence = []
                    else:
                        sequence.append(line)
        
        output_list = []
        generator = parse_fasta(input_file)
        next(generator)
        for entry in generator:
            entry[0].append(entry[1])
            output_list.append(entry[0])
        output_list.insert(0, ['sentence', 'label'])
        return output_list

    @classmethod
    def _read_fasta_localization(cls, input_file):
        """
        Reads a fasta file, label is in fasta header, preceded by " ", and succeeded by "-X", 
        where X in {"U", "M", "S"}.
        
        """
        def parse_fasta(file):
            with open(file, 'r') as f:
                # Create variables for storing the identifiers and the sequence.
                label = ''
                sequence = ''
                for line in f:
                    line = line.strip()  # Remove trailing newline characters.
                    if line.startswith('>'):
                        # Split on whitespace char that precedes the seq label, returns a list
                        line = line.split(' ')

                        # don't select the 1st item seq_id '>PROTXXX', but rather the 2nd item with label returns a str
                        # remove info about membrane location: Membrane, Soluble, Unknown
                        line = line[1].rstrip('-U -M -S')

                        # Start by yielding the entry that has been built up.
                        yield sequence, label

                        # Reinitialise the identifier and sequence variables to build up a new record.
                        label = line
                        sequence = []
                    else:
                        sequence.append(line)

        output_list = []
        generator = parse_fasta(input_file)
        next(generator)
        for entry in generator:
            entry[0].append(entry[1])
            output_list.append(entry[0])
        output_list.insert(0, ['sentence', 'label'])
        return output_list


    @classmethod
    def _read_fasta_remote_homology(cls, input_file):
        """
        Converts a two line fasta with fold labels and sequences into a list of lists:
            ```
            25 <unknown description>
            APVLSKDVADIESILALNPRTQSHAALHSTLAKKLDKKHWKRNPDKNCFHCEKLENNFDDMP
            1109 <unknown description>
            GLLSRLRKREPISIYDKIGGHEAIEVVVEDFYVRVLADDQLSAFFSGTNMSRLKGKQVEFLAVDVTS
            ```
        into:
            ```
            [   
                ['APVLSKDVADIESILALNPRTQSHAALHSTLAKKLDKKHWKRNPDKNCFHCEKLENNFDDMP', 25],
                ['GLLSRLRKREPISIYDKIGGHEAIEVVVEDFYVRVLADDQLSAFFSGTNMSRLKGKQVEFLAVDVTS', 1109]
            ]
            ```
        `input_file` is prepared by by running: 
        `scripts/lmdb_to_fasta_homology.py in.lmdb out.fasta True`
        
        Note, in the script above, if `Bio.SeqIO.FastaIO.SeqRecord` object has attribute `format('fasta')`, 
        replace it to `format('fasta-2line')` to get a 2-line fasta.
        
        Alternatively, keep the 40 char/line fasta format, and run the following command to get a 2-line fasta:
        `awk '/^>/ {printf("%s%s\n",(N>0?"\n":""),$0);N++;next;} {printf("%s",$0);} END {printf("\n");}' < in.fasta > input_file.fasta`
        """
        def parse_fasta(file):
            with open(file, 'r') as f:
                # Create variables for storing the identifiers and the sequence.
                label = ''
                sequence = []
                for line in f:
                    line = line.strip()  # Remove trailing newline characters.
                    if line.startswith('>'):
                        # Split on whitespace char that succeeds the seq label, returns a list
                        # >0 <unknown description> 
                        # ['>0', '<unknown description>'] 
                        line = line.split(' ')

                        line = line[0].strip('>')

                        # Start by yielding the entry that has been built up.
                        yield sequence, label

                        # Reinitialise the identifier and sequence variables to build up a new record.
                        label = line
                        sequence = []
                    else:
                        sequence.append(line)

        output_list = []
        generator = parse_fasta(input_file)
        next(generator)
        for entry in generator:
            entry[0].append(entry[1])
            output_list.append(entry[0])
        output_list.insert(0, ['sentence', 'label'])
        return output_list