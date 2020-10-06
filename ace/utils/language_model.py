import argparse
import gzip
import string
import sys
import time
from collections import Counter
from os import path

import numpy as np
import pandas as pd
from spellchecker import SpellChecker
import wordninja
from nltk.corpus import stopwords
stopwords_list = [x.upper() for x in set(stopwords.words('english'))]

from ace.utils.ace_exception import AceException

""" Creates a language model for use with the wordninja package"""
def keep_correctly_spelled(original_tokens, spell):
    """ Only keep words that are correctly spelled
    params: 
    * original tokens: list of words
    * spell: spellchecker.SpellChecker object
    """
    corrected_text = []
    mispelled_words = spell.unknown(original_tokens.split())
    for word in original_tokens.split():
        if word.lower() not in mispelled_words:
            corrected_text.append(word.upper())
    return " ".join(corrected_text)

def read_in_data(code):
    """ Reads in data for preprocessing
    params:
    * code: either 'sic' or 'soc'
    """
    data_file = f'./data/{code}/training_data.pkl.bz2'
    try:
        X, y = pd.read_pickle(data_file)
    except ValueError:
        print(f"{data_file} does not exist")
    spell = SpellChecker(distance=1)     
    X = X.str.translate(str.maketrans('', '', string.punctuation))
    df = pd.DataFrame({'words' :[keep_correctly_spelled(token, spell) for token in X]})

    return df

def create_language_model(df, word_col = 'words', output_filepath='./config/my_lang.txt.gz'):
    """ Creates a language model, containing words ordered by frequency
    params:
    * df: pandas.DataFrame
    * word_col: column containing words
    * output_filepath: location to save language model (must have a .txt.gz extension)
    """
    word_count = dict(Counter(" ".join(df[word_col]).split(" "))) 
    word_count_df = pd.DataFrame.from_dict(word_count,orient='index').reset_index()
    word_count_df.columns= ['words', 'n_appearances']
    
    #only keep actual words
    word_count_df['wordlength'] = word_count_df['words'].str.len()
    word_count_df = word_count_df[(word_count_df['wordlength'] >=3) | (word_count_df['words'].isin(stopwords_list))]
    word_count_df = word_count_df.sort_values('n_appearances',ascending=False).reset_index(drop=True)
    word_count_df['words'] = word_count_df['words'].str.lower()
    word_count_df['words'].to_csv(output_filepath,index=None, header=False,compression='gzip',encoding='utf-8')


def get_args(command_line_arguments):
    parser = argparse.ArgumentParser(description="an app to classify text in a convenient way",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # include defaults in help
                                     conflict_handler='resolve')  # allows overriding of arguments
    parser.add_argument("-c", "--code", default='both', help='Include either only sic or soc codes')
    args = parser.parse_args(command_line_arguments)
    return args


def main(supplied_args):
    args = get_args(supplied_args)
    if args.code == 'both':
        soc_df = read_in_data('soc')
        sic_df = read_in_data('sic')
        language_df = pd.concat([soc_df, sic_df]).reset_index(drop=True)
    elif args.code in ['sic','soc']:
        language_df = read_in_data(args.code)
    else:
        raise ValueError('Code must be "sic", "soc", or left to default as "both"')
    output_filepath = './config/my_lang.txt.gz'
    create_language_model(language_df, output_filepath)
    print("Language model saved to: " + output_filepath)
        
if __name__ == '__main__':
    try:
        start = time.time()
        main(sys.argv[1:])
        end = time.time()
        diff = int(end - start)
        hours = diff // 3600
        minutes = (diff % 3600) // 60
        seconds = diff % 60

        print('')
        print(f"Language model took {hours}:{minutes:02d}:{seconds:02d} to complete")
    except AceException as err:
        print(f"ACE error: {err.message}")