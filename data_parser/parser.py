''' 
This code uses the same dictionaries and order of parsing steps to the original SIC-SOC-CODING-TOOL

'''

from collections import OrderedDict
import os
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

def Exceptions(code, exceptions, suffix_list=None):
    '''
        Loads a list of expcetion words and suffixs not to use when apply parsing function.
        Load the lists from pickled dictionaries or from .txt files.

    '''
    # Load expcetions folder
    parser_dir = f"./ace/data_parser/{code}/"
    # if there are dictionaries available
    if exceptions is not None:
        # load list from text file
        if exceptions.endswith(".txt"):
            with open(parser_dir + exceptions) as f:
                exceptions = [line.strip() for line in f]

        # load list from dictionaries
        else:
            # load dictionaries using parser class
            dictionaries = LoadParseDicts(code).load_dicts()

            # get dictionary containing list of suffix excpetions
            if suffix_list is not None:
                exceptions = tuple(dictionaries[exceptions])

            # get dictionary containing word exceptions
            else:
                exceptions = [w.replace("\\1", "").replace("\\2", "")
                              for w in dictionaries[exceptions].values()]

    # else load an empty list
    else:
        exceptions = []
    return exceptions

class LoadParseDicts(): 
    def __init__(self, code): 
        self.__code = code
        
    def __load_pickle(self, path_to_file):
        with open(path_to_file, 'rb') as handle:
            return pickle.load(handle)

    def __load_prs_dicts(self):
        """Loads a locally saved parser dictionaries from a given type"""

        path = f"ace/data_parser/{self.__code}"

        def __load(path, filename):
            try:
                return self.__load_pickle(f"{path}/{filename}")
            except IsADirectoryError: 
                pass

        try:
            return {
                    filename.split('.')[0]:__load(path, filename) 
                    for filename in os.listdir(path)
                   }
        except FileNotFoundError:
            raise FileNotFoundError(f"parser files not found in {path}")

    def load_dicts(self):
        type_parser = self.__code.lower()
        if self.__code in ['sic','soc']:
            return self.__load_prs_dicts()
        else:
            raise ValueError('Type Parser should either be sic or soc.')
            
class ParseSoc(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dicts = LoadParseDicts("soc").load_dicts() 
    
    def fit(self, x, y=None):
        return self
        
    def transform(self, x):
        print("Starting parsing steps")
        
        def _parse_soc(original_tokens):
            """Perform a series of manipulations on the Pandas Series passed as 
            input.

            Args:
                column: input Pandas series
                dicionaries: dictionary of parsing dictionaries defining 
                              word replacements

            Returns:
                The parsed Pandas series.       
            """
            dictionaries = self.dicts

            parsed = (original_tokens.str.strip() # Introduce extra step of stripping leading and trailing ...
              .str.replace(r"\s+", " ") # ...and double white spaces
              .str.replace(r"\s?-+\s?", "-") # trim hypens
              .replace(dictionaries["RSTR_dict"], regex=True)
              .replace(dictionaries["DSTR_dict"], regex=True)
              .str.replace("[^A-Z0-9&'-]", " ") # word breaking 
              .str.replace("'", "") # aphostrophe or not? I would say not
              .str.strip()
              .str.replace("\s+", " ") # redo as previous step might have introduced extra spaces
              .replace(dictionaries["RWRD_dict"], regex=True)
              .replace(dictionaries["HWRD_dict"], regex=True)
              .str.replace("-", " ") # break all hyphens words
              .replace(dictionaries["DWRD_dict"], regex=True)
              .replace(dictionaries["EWRD_dict"], regex=True)
              .str.strip()
              .str.replace("\s+", " "))

            parsed = parsed.apply(lambda s: " ".join(OrderedDict.fromkeys(s.split(" ")).keys()))
            return parsed
        out = _parse_soc(x)
        return out
    
class ParseSic(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dicts = LoadParseDicts("sic").load_dicts() 
    
    def fit(self, x, y=None):
        return self
        
    def transform(self, x):
        print("Starting parsing steps")
        
        def _parse_sic(original_tokens):
            """Perform a series of manipulations on the Pandas Series passed as 
            input.

            Args:
                column: input Pandas series
                dicionaries: dictionary of parsing dictionaries defining 
                              word replacements

            Returns:
                The parsed Pandas series.       
            """
            dictionaries = self.dicts
            parsed = (original_tokens.str.strip() # Introduce extra step of stripping leading and trailing ...
              .str.replace(r"\s+", " ") # ...and double white spaces
              .str.replace(r"\s?-+\s?", "-") # trim hypens
              .replace(dictionaries["RSTR_dict"], regex=True)
              .replace(dictionaries["DCLS_dict"], regex=True)
              .str.replace("[^A-Z0-9'-]", " ") # Word Breaking
              .str.strip()
              .str.replace("\s+", " ") # redo as previous step might have introduced extra spaces
              .replace(dictionaries["HWRD_dict"], regex=True)
              .str.replace("-", " ") # break all hyphens words
              .replace(dictionaries["DWRD_dict"], regex=True)
              .replace(dictionaries["RWRD_dict"], regex=True)
              .replace(dictionaries["EWRD_dict"], regex=True)#note called EXCP off-DAP
              .str.strip()
              .str.replace("\s+", " "))
            parsed = parsed.apply(lambda s: " ".join(OrderedDict.fromkeys(s.split(" ")).keys()))
            return parsed
        out = _parse_sic(x)
        return out