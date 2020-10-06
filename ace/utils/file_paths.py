import json
import os

from ace.utils.utils import strip_file_ext

config_path = os.path.abspath('./config/config.json')

with open(config_path, 'r') as f:
    config = json.load(f)
    
file_path_stems = {
    'sic': strip_file_ext(config['hdfs_path']['2011_sample']['sic']),
    'soc': strip_file_ext(config['hdfs_path']['2011_sample_v2']['soc'])
}

file_path_suffixes = {
    "training": {
        #balanced
        False: '_training'
    },
    "validation": {
        #balanced
        True: "_validation_balanced",
        False: "_validation"
    },
    "full": {
        #balanced
        False: ''
    }
}

def get_file_path(code, dataset_type, balanced):
    """
    Creates full file path
    code: {'sic', 'soc'}
    dataset_type: {'original','training','validation'}
    balanced: bool
    """
    return file_path_stems[code] + file_path_suffixes[dataset_type][balanced] + '.csv'
