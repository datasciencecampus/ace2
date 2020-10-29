from os import path as path, makedirs
import argparse
import bz2
import pickle
import random
import sys
import time
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.data_access import DataLoader
from ace.utils import utils
from ace.utils.ace_exception import AceException


def get_args(command_line_arguments):
    parser = argparse.ArgumentParser(description="an app to ingest data from various datafile formats and create a "
                                                 "pickled DataFrame",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # include defaults in help
                                     conflict_handler='resolve')  # allows overriding of arguments

    # suppressed:________________________________________
    parser.add_argument("-dpt", "--data-path", default='./data', help=argparse.SUPPRESS)
    parser.add_argument("-c", "--code", default='soc', help='Either sic or soc')
    parser.add_argument("-lm", "--limit", type=int, default=None, help='Optional argument to limit the size of the dataset to n rows. Defaults to no limits')
    parser.add_argument("-rt", "--run-type", default="training", help="Either load training data or validation data")
    parser.add_argument("-b", "--balanced", default=False, action="store_true", help="Use a balanced version of the dataset. At the moment this is only possible when --run-type='validation'")

    parser.add_argument("-mnc", "--min-label-count", type=int, default=0, help='Minimum number of class members allowed without dropping class.')
    parser.add_argument("-mxc", "--max-label-count", type=int, default=sys.maxsize, help='Maximum number of class members allowed without dropping class.')
    parser.add_argument("-s", "--sample", default=1.0, help='A sample of the data')

    parser.add_argument("-lh", "--label-header", default='SOC2020_code', help='The label header: {soc: SOC2020_code, sic: industry_code}')
    parser.add_argument("-th", "--text-header", default='job_title', help='The text header')
    parser.add_argument("-oth", "--other-text-header", default='employer_text', help='The text header for sic codes')
    parser.add_argument("-ih", "--index-header", default='id', help='A unique index column, identifying the samples')
    
    parser.add_argument("-gd", "--group-doubts", default=False, action="store_true", help='Group all doubts together in one class. False means that doubts are merged with their closest numerical code')
    parser.add_argument("-du", "--drop-unknowns", default=False, action="store_true", help='Drop unknown classes')
    
    parser.add_argument("-m", "--memory", default=False, action="store_true", help='Save memory consumption when running script')
    parser.add_argument("-f", "--filename", type=str, default=os.path.basename(__file__)[:-3], help='Returns string of the name of current file')
    
    args = parser.parse_args(command_line_arguments)

    return args


def plot_histogram(labels_hist, classes, dir_name, num_red_classes, title='load_balance'):
    """
    For reporting the load balancing of the data.
    """
    n_classes = len(classes)
    label_counts = [labels_hist[x] for x in classes]
    sorted_counts_indices = sorted(range(len(label_counts)), key=lambda k: label_counts[k])
    sorted_classes = [classes[x] for x in sorted_counts_indices]
    sorted_label_counts = [label_counts[x] for x in sorted_counts_indices]
    nticks = range(n_classes)

    classes_colours = ['red' if i < num_red_classes else 'orange' for i in range(n_classes)]
    plt.bar(nticks, sorted_label_counts, width=2, alpha=0.2, color=classes_colours)

    plt.title('load balancing graph')
    plt.ylabel('Label Counts')

    plt.xticks(nticks, sorted_classes, rotation='vertical', fontsize=4)
    plt.xlabel('Labels')

    plt.savefig(path.join(dir_name, title))


def main(supplied_args):
    args = get_args(supplied_args)
    label_header = args.label_header
    max_labels = int(args.max_label_count)
    dirname = path.join(args.data_path, args.code)
    makedirs(dirname, exist_ok=True)
    if args.balanced:
        filename = f'{args.run_type}_balanced_data.pkl.bz2'
    else:
        filename = f'{args.run_type}_data.pkl.bz2'
    filepath = path.join(args.data_path, args.code, filename)

    # pick desired features, then drop nas
    df = DataLoader(code=args.code, spark=utils.spark(), lim=args.limit, dataset_type=args.run_type, balanced=args.balanced).df
    num_all = df.shape[0]
    print("Size: " + str(num_all))
    df = df.dropna(subset=[label_header, args.text_header])
    num_non_na = df.shape[0]
    print("Nulls dropped :" + str(num_all-num_non_na))

    print(df.head())
    dropclasses = []
    reduce_classes = []
    dropped_count = 0
    
    # returns dict of counts of classes
    hist = utils.create_load_balance_hist(df[args.label_header])
    
    # Figure out which classes need to be dropped, and which reduced in number
    for key in hist:
        if hist[key] < args.min_label_count:
            dropped_count += hist[key]
            dropclasses.append(key)
        if hist[key] > args.max_label_count:
            reduce_classes.append(key)
    plot_histogram(hist, list(set(df[label_header])), dirname, len(dropclasses), 'before_load_balance')
    
    # For classes with too MANY samples, randomly drops samples down to limit
    indices_drop = []
    for label in reduce_classes:
        indices = df[df[label_header] == label].index.tolist()
        num_to_drop = len(indices) - max_labels
        indices_to_drop = random.sample(indices, num_to_drop)
        indices_drop.extend(indices_to_drop)
    df.drop(indices_drop, inplace=True)
    
    # Drops the classes with too FEW samples
    num_classes_dropped = len(dropclasses)
    print('dropping ' + str(len(dropclasses)) + ' small classes')
    for cls in dropclasses:
        df.drop(df[df[label_header] == cls].index, inplace=True)
    num_remaining = df[label_header].nunique()
    print(str(num_remaining) + ' classes remaining')
    
    # Handles doubtful/unknown classes
    rows_before = df.shape[0]
    classes_before = df[label_header].nunique()
    if args.group_doubts:
        df.loc[~df[label_header].str.isdigit(), label_header] = '-1'
        if args.drop_unknowns:
            df.drop(df[df[label_header] == '-1'].index, inplace=True)
            print(str(rows_before-df.shape[0]) + ' rows removed')
    else:
        df[label_header] = df[label_header].str.replace(r'[^0-9]','')
        print(str(classes_before-df[label_header].nunique()) + ' classes merged')
        if args.drop_unknowns:
            df.drop(df[df[label_header] == ''].index, inplace=True)
            
            print(str(rows_before-df.shape[0]) + ' rows removed')
        else:
            df.loc[df[label_header] == '', label_header] = '-1'
            
    print(str(df[label_header].nunique()) + ' classes remaining after cleaning')

    hist = utils.create_load_balance_hist(df[args.label_header])

    plot_histogram(hist, list(set(df[label_header])), dirname, 0, 'after_load_balance')
    
    # smaller sample if specified by fraction, happens with replacement
    if float(args.sample) < 0.9:
        df = df.sample(frac=float(args.sample), replace=True, random_state=1)

    num_after_sample = df.shape[0]
    
    # Assign id column to index of data
    if args.index_header:
        df.index = df[args.index_header]
        
    if args.other_text_header:
      df[args.text_header] += '. '+ df[args.other_text_header]

    obj = df[args.text_header], df[label_header]
    # TODO: report file
    print(filepath)
    with bz2.BZ2File(filepath, 'wb') as pickle_file:
        pickle.dump(obj, pickle_file, protocol=4, fix_imports=False)

    output_text = (
            "******* Data Reader Report ************" + "\n" +
            "specified load limit: " + str(args.limit) + "\n" +
            "num all data: " + str(num_all) + "\n" +
            "min label count: " + str(args.min_label_count) + "\n" +
            "num classes all: " + str(len(set(df[label_header]))) + "\n" +
            "num all data na dropped: " + str(num_non_na) + "\n" +
            "num classes dropped: " + str(num_classes_dropped) + "\n" +
            "num class data  dropped: " + str(dropped_count) + "\n" +
            "num  data  remained: " + str(num_remaining) + "\n" +
            "num data sampled: " + str(num_after_sample) + "\n" +
            "**************************************\n")
    
    print(output_text)
    filename_out = path.join(dirname, 'report.txt')
    with open(filename_out, "w") as f:
        f.write(output_text)


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
        minutes = diff // 60
        seconds = diff % 60

        print('')
        print(f"data ingest took {hours}:{minutes:02d}:{seconds:02d} to complete")
    except AceException as err:
        print(f"Ace error: {err.message}")