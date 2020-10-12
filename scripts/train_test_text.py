import argparse
import json
import sys
import time
from os import makedirs, path

from ace.pipelines.pipeline_ml import MLPipeline
from ace.utils.ace_exception import AceException
from ace.utils import utils

algorithm_names = ['RandomForestClassifier', "RandomForestClassifier_104", "RandomForestClassifier_128",
                   'LogisticRegression']


# logistic regression, decision tree, XGBoost

# todo: suffix


def get_args(command_line_arguments):
    parser = argparse.ArgumentParser(description="an app to classify text in a convenient way",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # include defaults in help
                                     conflict_handler='resolve')  # allows overriding of arguments

    # suppressed:_______________________________  _________
    parser.add_argument("-d", "--dirname", default='soc', help='The directory name for the cached feature matrix')
    parser.add_argument("-ucm", "--use-cached-model", default=False, action="store_true", help='Multi-labeled data')
    parser.add_argument("-ml", "--multi-label", default=True, action="store_true", help='Multi-labeled data')
    parser.add_argument("-r", "--train-test-ratio", default=0.75, help='The train-test ratio')
    parser.add_argument("-a", "--algorithms", type=int, nargs='+', default=[3],
                        help=(", ".join([f"{index}. {value}" for index, value in enumerate(algorithm_names)]))
                        + "; multiple inputs are allowed.\n")
    parser.add_argument("-m", "--memory", default=False, action="store_true", help='Save memory consumption when running script')
    parser.add_argument("-f", "--filename", type=str, default=path.basename(__file__)[:-3], help='Returns string of the name of current file')
    parser.add_argument("-acc", "--accuracy", type=float, default=None, help='The desired accuracy per class')
    parser.add_argument("-t", "--threshold", default=0.5, help='The desired threshold per class')
    
    args = parser.parse_args(command_line_arguments)

    return args


def main(supplied_args):
    args = get_args(supplied_args)

    multilabel = args.multi_label

    ####################################################################

    # ALGORITHMS CHOICE
    alg_types = [algorithm_names[x] for x in args.algorithms]
    outputs_name = args.dirname
    data_filename ="training_data.pkl.bz2"
    
    makedirs(path.join('outputs', outputs_name), exist_ok=True)
    # initialize pipeline object
    model_obj = MLPipeline(
        alg_types, multi=multilabel, dirname=outputs_name, output_dirname= outputs_name,
        data_filename=data_filename, train_test_ratio=float(args.train_test_ratio), use_cached_model=args.use_cached_model
    , threshold=args.threshold, memory = args.memory, accuracy=args.accuracy)
    
    with open('pipeline_config.json') as json_file:
        pipeline_config = json.load(json_file)
    project = pipeline_config["project"]
    features = pipeline_config["features"]
    
    full_pipeline_config = {
        "project" : project,
        "features": features,
        "model" : {
            "alg_types" : alg_types,
            "multi" : multilabel
        }
    }
    
    with open('pipeline_config.json','w') as outfile:
        json.dump(full_pipeline_config, outfile)
    
    # run pipeline
    model_obj.test_models()
    model_obj.validate_models()
    model_obj.validate_models(balanced=True)


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
        print(f"training took {hours}:{minutes:02d}:{seconds:02d} to complete")
    except AceException as err:
        print(f"ACE error: {err.message}")