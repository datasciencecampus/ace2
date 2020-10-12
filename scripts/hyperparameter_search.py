import argparse
import sys
import time
import json
from os import makedirs, path

from ace.pipelines.pipeline_ml import HyperparameterPipeline
from ace.utils.ace_exception import AceException


def get_args(command_line_arguments):
    """
    Fun fact!  Apart from directory info, all the settings for the
    hyperparameter search are to be found in the relevant config file.
    """
    
    parser = argparse.ArgumentParser(description="an app to classify text in a convenient way",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # include defaults in help
                                     conflict_handler='resolve')  # allows overriding of arguments

    # suppressed:_______________________________  _________
    parser.add_argument("-d", "--dirname", default='soc', help='The directory name for the cached feature matrix')
    parser.add_argument("-c", "--config", default='hyperparameter_test.json', help='The name of the search config file')
    parser.add_argument("-m", "--memory", default=False, action="store_true", help='Save memory consumption when running script')
    parser.add_argument("-f", "--filename", type=str, default=path.basename(__file__)[:-3], help='Returns string of the name of current file')
    
    args = parser.parse_args(command_line_arguments)

    return args


def main(supplied_args):
    args = get_args(supplied_args)

    ####################################################################
    
    outputs_name = args.dirname
    makedirs(path.join('outputs', outputs_name), exist_ok=True)
    
    parameters_path = path.abspath(path.join('./config', args.config))
    
    # Get the hyperparameters over which to search
    with open(parameters_path, "r") as f:
        parameters = json.load(f)
    
    algorithm = parameters['algorithm']
    hyperparameters = parameters['hyperparameters']
    cv = parameters['cv']
    
    """
    FOR LOGISTIC REGRESSION
    C = inverse of regularisation strength, therefore smaller = stronger regularisation
    l1_ratio = Elastic-net mixing parameter, relative weights of l1 and l2 regularisation
    
    if l1_ratio = 1.0, method == LASSO regression
    if l1_ratio = 0.0, method == ridge regression
    """
    
    # Initialize pipeline object
    model_obj = HyperparameterPipeline(algorithm,
                                       dirname=outputs_name,
                                       hyperparameters=hyperparameters,
                                       cv=cv)
    
    # run pipeline
    model_obj.test_parameters()


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