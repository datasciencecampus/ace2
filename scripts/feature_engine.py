import argparse
import json
import sys
import time
import ast
from os import path, makedirs

import pandas as pd

from ace.utils.ace_exception import AceException
from ace.utils import utils
from ace.pipelines.pipeline_feature import PipelineFeatures

feature_selection_algs = ['Logistic', "X2", "lsvc", "ExtraTrees"]
features = ['frequency_matrix', 'word_features', 'nmf', 'embeddings']
run_types =['training', 'validation', 'validation_balanced']


def get_args(command_line_arguments):
    parser = argparse.ArgumentParser(description="an app to create and cache a features matrix",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # include defaults in help
                                     conflict_handler='resolve')  # allows overriding of arguments

    # suppressed:________________________________________
    parser.add_argument("-ntf", "--num-tfidf-features", default='all', help='number of tf-idf features')
    parser.add_argument("-mwc", "--min-word-count", type=int, default=2, help="Number of times a word has to appear to be included in the features")
    parser.add_argument("-sf", "--suffix", default='', help='The suffix for the run')
    parser.add_argument("-dpt", "--data-path", default='data', help=argparse.SUPPRESS)
    parser.add_argument("-c", "--code", default='soc', help='Either sic or soc')
    parser.add_argument("-rt", "--run-types", type=str, nargs='+', default=run_types,
                        help=(", ".join([f"{index}. {value}" for index, value in enumerate(run_types)]))
                        + "; multiple inputs are allowed.\n")
    parser.add_argument("-fs", "--feature-selection", type=int, default=1,
                        help=(", ".join([f"{index}. {value}" for index, value in enumerate(feature_selection_algs)]))
                        + "; single input.\n")
    parser.add_argument("-ft", "--features", type=int, nargs='+', default=[0],
                        help=(", ".join([f"{index}. {value}" for index, value in enumerate(features)]))
                        + "; multiple inputs are allowed.\n")
    parser.add_argument("-sc", "--spell-correct", default=True, action="store_true", help='spell correction')
    parser.add_argument("-p", "--parse", default=False, action="store_true", help='parsing steps')
    parser.add_argument("-e", "--exceptions", default='{"suffix":None, "except":None}', type=str, action = "store", help='exceptions to parsing')
    parser.add_argument("-m", "--memory", default=False, action="store_true", help='Save memory consumption when running script')
    parser.add_argument("-f", "--filename", type=str, default=path.basename(__file__)[:-3], help='Returns string of the name of current file')
    
    args = parser.parse_args(command_line_arguments)

    return args


def main(supplied_args):
    args = get_args(supplied_args)

    # FEATURE SELECTION CHOICE
    feature_select_type = feature_selection_algs[args.feature_selection]

    # Feature choice
    feature_set = [features[x] for x in args.features]

    # set up file paths
    data_directory = path.join(args.data_path, args.code)
    features_directory_name = args.code + args.suffix
    print(data_directory)
    
    # TRAINING FILE
    training_filename = "training_data.pkl.bz2"
       
    makedirs(path.join('cached_features', features_directory_name), exist_ok=True)
    makedirs(path.join('models', features_directory_name), exist_ok=True)
    model_obj = PipelineFeatures(data_dirname = data_directory, data_filename = training_filename, feature_set = feature_set, idf=False, num_features=args.num_tfidf_features, feature_selection_type=feature_select_type, spell=args.spell_correct, parser=args.parse, min_df=args.min_word_count, exceptions = ast.literal_eval(args.exceptions))
    
    model_obj.save_data(features_directory_name, 'training')
    print("Training features saved")
    
    for run_type in args.run_types:
        if run_type in ['validation','validation_balanced']:
            valid_filename = f"{run_type}_data.pkl.bz2"
            X, y = pd.read_pickle(path.join(data_directory, valid_filename))
            X = model_obj.reuse_pipeline(X)
            model_obj.save_data(features_directory_name, run_type, X, y)
            print(run_type + " features saved")

    # save config
    pipeline_config = {
        "project": {
            "code" : args.code,
            "data_file_path" : data_directory
        },
        "features" : {
            "feature_select_type" : feature_select_type,
            "feature_set" : feature_set,
            "file_path": features_directory_name,
            'num_features' : args.num_tfidf_features,
            "spell" : args.spell_correct, 
            "parser": args.parse
        } 
    }
                       
    with open('pipeline_config.json','w') as outfile:
        json.dump(pipeline_config, outfile)


if __name__ == '__main__':
    try:
        start = time.time()
        args = get_args(sys.argv[1:])
        if args.memory: 
            main = utils.MemoryMonitor(main, args).save_memory_output()
        main(sys.argv[1:])
        end = time.time()
        diff = int(end - start)
        hours = diff // 3600
        minutes = (diff % 3600) // 60
        seconds = diff % 60

        print('')
        print(f"Feature engineering took {hours}:{minutes:02d}:{seconds:02d} to complete")
    except AceException as err:
        print(f"ACE error: {err.message}")