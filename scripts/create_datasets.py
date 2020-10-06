import argparse
import string
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from scripts.data_access import DataLoader, check_classes, compare_codes
from ace.utils.file_paths import get_file_path
from ace.factories.data_factory import save_data_to_hdfs
from ace.utils.ace_exception import AceException
from ace.utils.utils import spark

warnings.warn(
    "If you try to run this script and save the data, you will receive an error as the filepath already exists. You are unable to overwrite data already saved in Hue"
)

# Large enough for validation, whilst not removing too much from the training data
# More data is taken for sic for validation as the data has come from the census data, although the quality of the data is poorer, we are able to take more of it
# TODO: we are temporarily NOT taking SIC data from the full census data, and have reduced the sample size to 10000 from 20000
random_state=42
config = {
    'soc': {
        'sample_size' : 20000,
        'sample_size_validation': 10000,
        'class_size': 5, 
        'id_col': 'id',
        'label_header' : 'SOC2020_code',
        'title_header' : 'job_title', 
        'sic_coding_tool_lh' : 'SIC2007_predicted',
    },
    'sic': {
        'sample_size' : 20000,
        'sample_size_validation': 10000,
        'class_size': 5, 
        'id_col':'id',
        'label_header' : 'industry_code',
        'title_header' : 'business',
        'sic_coding_tool_lh' : 'SIC2007_predicted',
        'full' : None
    }
}
full_census_config = {
        'balanced_sample_size' : 10000,
        'census_sample_fraction' : 0.01,
        'validation_sample_fraction' : 0.1,
        'id_col' : 'id'
    }          


def test_access(code, spark): 
    """
    Test basic access of a pyspark dataframe in HDFS.
    """
    if code not in ['sic','soc']:
        raise ValueError(f"Code {code} is not recognised. Please use either 'sic' or 'soc'.")    
    else:   
        for dataset_type in ['validation','training','full']:
            try:
                file_path = get_file_path(code, dataset_type, balanced=False)
                df = spark.read.csv(file_path)
            except Exception as e:
                raise ValueError(f"Error in loading {filepath}")
        try:
            file_path = get_file_path(code, 'validation', balanced=True)
            df = spark.read.csv(file_path)
        except Exception as e:
            raise ValueError(f"Error in loading {filepath}")
            
    print("All dataframes loaded successfully")
    
def check(training, validation, balanced_validation, id_col, code):
    if list(set(training[id_col]) & set(validation[id_col]) & set(balanced_validation[id_col])):
        print("There is overlap in dataframes")
        print("Number of common items: " + str(len(list(set(training[id_col]) & set(validation[id_col]) & set(balanced_validation[id_col])))))
    else: 
        print("No overlap in dataframes")
    run_type = {"training":training, "validation":validation, "balanced_validation":balanced_validation}
    for k,v in run_type.items():    
        print("The length of " + code + " " + k + " dataframe is " + str(len(v)))

def create_datasets(code, spark):
    """
    Creates a training, validation and balanced validation set
    code: {'sic','soc'}
    """
    label_header = config[code]['label_header']
    title_header = config[code]['title_header']
    id_col = config[code]['id_col']
    
    dl = DataLoader(code=code,spark=spark, sample=True, lim=None)
    dl_full = DataLoader(code=code,spark=spark, sample=False, lim=None)
          
    if code == 'sic':
        additional_header = 'employer_text'
        use_full_census_data = config[code]['full']
    else:
        use_full_census_data = None

    if use_full_census_data:
        balanced_sample_size, census_sample_fraction, validation_sample_fraction, id_col = full_census_config.values()
        df_to_be_sampled = dl_full.df.sample(False,census_sample_fraction,seed=random_state)
        code_list=[i[label_header] for i in df_to_be_sampled.select(label_header).distinct().collect()]
       
    # census data large enough to assume smaller classes will not need to be filtered out as in 1% EA subsample
        balanced_validation = dl_full.balance_validation(
                size_of_validation = balanced_sample_size, 
                code_list=code_list
            )
        balanced_validation = balanced_validation.withColumnRenamed("employer", "employer_text")

        df_to_be_sampled = df_to_be_sampled.join(balanced_validation, on=[id_col], how='left_anti')
        validation = df_to_be_sampled.sample(False,validation_sample_fraction, seed = random_state)
        training = df_to_be_sampled.join(validation, on=[id_col], how='left_anti')
        training = training.dropna(subset=[title_header])
        validation = validation.toPandas()
        training = training.toPandas()
        balanced_validation = balanced_validation.toPandas()
    else:
        df_to_be_sampled = dl.remove_small_classes(min_class_size=config[code]['class_size'])
        #returns list of SOC codes in the full census data not in 1 pc sample to QA
        codes_in_1pc, column_names = compare_codes(code=code,full_df = dl_full.df, df=df_to_be_sampled)
        
        balanced_validation = dl.balance_validation(
            size_of_validation = config[code]['sample_size_validation'], 
            code_list=codes_in_1pc
        )
        
        df_to_be_sampled = df_to_be_sampled[~df_to_be_sampled[id_col].isin(balanced_validation[id_col])]
        validation = df_to_be_sampled.sample(n=config[code]['sample_size'], random_state = random_state)
        training = df_to_be_sampled.loc[~df_to_be_sampled[id_col].isin(validation[id_col])].dropna(subset=[title_header])   
                                             
    check(training, validation, balanced_validation, id_col, code)
    
    for data in [validation, balanced_validation]: 
        data[config[code]['sic_coding_tool_lh']] = data[config['sic']['label_header']]        
    return training, validation, balanced_validation
                                                                                      
def save_file(df, code, dataset_type, spark_session, balanced=False):
    file_path = get_file_path(code, dataset_type,balanced)
    for col in df:
        df[col] = df[col].astype('str')
    save_data_to_hdfs(spark_session.createDataFrame(df), file_path = file_path)

def get_args(command_line_arguments):
    parser = argparse.ArgumentParser(description="an app to create training and validation datasets for sic and soc",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # include defaults in help
                                     conflict_handler='resolve')  # allows overriding of arguments
    parser.add_argument("-c", "--code", default='both', help='Pass either sic or soc to generate only datasets for that code')
    
    args = parser.parse_args(command_line_arguments)

    return args

def main(supplied_args):
    
    spark_session = spark()
    args = get_args(supplied_args)
    
    if (args.code !='both') & (args.code in ['sic','soc']):
        code_list = [args.code]
    elif args.code == 'both':
        code_list = ['sic','soc']
    else:
        raise AceException(f'Unrecognised code argument {args.code}')
    print(f'Creating training, validation and balanced validation datasets for {code_list}')
    for code in code_list:
        training, validation, balanced_validation = create_datasets(code, spark_session)
        
        check_classes(training, validation, balanced_validation, code = code)
        
        save_file(training, code=code,dataset_type='training',spark_session=spark_session)
        save_file(validation, code=code,dataset_type='validation',spark_session=spark_session)
        save_file(balanced_validation, code=code,dataset_type='validation', spark_session=spark_session, balanced=True)
        
        test_access(code, spark_session)
    print(f'Created datasets for {code_list}')
        
if __name__ == '__main__':
    try:
        start = time.time()
        main(sys.argv[1:])
        end = time.time()
        diff = int(end - start)
        hours = diff // 3600
        minutes = diff // 60
        seconds = diff % 60

        print('')
        print(f"Data set creation took {hours}:{minutes:02d}:{seconds:02d} to complete")
    except AceException as err:
        print(f"Ace error: {err.message}")