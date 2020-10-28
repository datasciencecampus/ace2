import os
import pickle

from os import path


def create_load_balance_hist(yin):
    d = {}
    y = yin.tolist()
    for i in range(len(yin)):
        key = y[i]
        if key in d:
            d[key] += 1
        else:
            d[key] = 1
    return d


def strip_file_ext(file_path, file_ext='.csv'):
    return file_path.replace(file_ext, '')


def check_and_create(base_path):
    if not path.exists(base_path):
        os.makedirs(base_path)


def load_corrections(parser_dir, dict_files_list=None):
    """
    Loads a series of dictionaries of manually defined regex replacements/alterations to be made to the text.
    :param parser_dir: path to folder with the pickled dicts in
    :param dict_files_list: list of filenames, if None all files in folder will be read
    :return a single bigger dict of all corrections
    """
    def __load(path, filename):
        try:
            with open(f"{path}/{filename}", 'rb') as handle:
                return pickle.load(handle)
        except IsADirectoryError:
            pass

    def __load_dicts(folder, filenames):
        try:
            return {
                filename.split('.')[0]: __load(folder, filename)
                for filename in filenames
            }
        except FileNotFoundError:
            raise FileNotFoundError(f"parser files not found in {path}")

    if not dict_files_list:
        dict_files_list = os.listdir(parser_dir)

    return __load_dicts(parser_dir, dict_files_list)
