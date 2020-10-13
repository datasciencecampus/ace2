import os
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
    return file_path.replace(file_ext,'')

def check_and_create(base_path):
    if not path.exists(base_path):
        os.makedirs(base_path)