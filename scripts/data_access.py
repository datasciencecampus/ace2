import os 
import bz2
import json
import pdb
import pickle
import string
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql import Window

from ace.utils.file_paths import get_file_path
from ace.factories.data_factory import load_data

#------------------------
# Create DataLoader class
#------------------------

class DataLoader:
    def __init__(self, spark=None, code = None, lim = None, dataset_type='full', sample=True, balanced=False):
        
        """
        Defines the arguments to be used in the data loader class: 
        code: the census type of data you are using, {"sic", "soc"}
        lim: integer limit of dataframe when loading it from Hue. 
        sample: data taken from the 1% sample
            
        """
        self.code = code

        if not sample:
            if self.code== 'sic':
                source_string = "SELECT * FROM 2011_census_identified.cen_ident_2011_stage_0_writein_industry_std"
            elif self.code == 'soc':
#                 source_string = "SELECT * FROM 2011_census_identified.cen_ident_2011_stage_0_writein_occupation_std where occ_job_title is not null and occupation_code is not null"
                source_string = "SELECT * FROM 2011_census_identified.cen_ident_2011_stage_0_writein_occupation_std"
            data_source = 'sql'
            return_as_pandas = False
        elif self.code in ['sic','soc']:
            source_string = get_file_path(self.code, dataset_type, balanced)
            data_source = 'file'
            return_as_pandas = True
        else:
             raise ValueError("Code is not recognised. Please use either sic or soc")
        self.df = load_data(source_string, spark, data_source, lim, return_as_pandas)
        if not sample:
            # add index
             self.df = self.df.withColumn("id",F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))-1)

    def balance_validation(self, size_of_validation, code_list, random_state=42):
        '''
        Get a dataset with balanced classes: 
        SIC - use full census data 
        SOC - use training data 
        
        code_list: list of codes to filter when creating the validation data (e.g. classes with instances above threshold set)
        
        '''
        startTime = datetime.now()
        # threshold to filter classes for validation
        size_of_classes = int(size_of_validation/len(code_list))
        
        if self.code == "soc":
            col_name='SOC2020_code'
        elif self.code == "sic": 
            col_name = "industry_code"
        else: 
            raise ValueError("Code is not recognised. Please use either 'sic' or 'soc'.")
        
        if isinstance(self.df, pd.DataFrame):
            # ensure codes are not present in validation but not in training
            # this is not ideal as we will not be performing well on smaller classes and will not be measuring this
            appearances = self.df[col_name].value_counts()
            sampled_df = self.df[self.df[col_name].isin(appearances[appearances>(2*size_of_classes)].index)]
            balanced_validation = sampled_df.groupby(col_name).apply(lambda s: s.sample(size_of_classes, random_state=random_state))

        else:
            # census data large enough to assume smaller classes will not need to be filtered out as above
            # only select full census data where the code can be found in the training data
            # TODO: not sure this is the ideal process
            filtered_df = self.df.filter(F.col(col_name).isin(code_list))
            n_classes = filtered_df.select(F.countDistinct(col_name).alias('n_classes')).collect()[0]['n_classes']
            size_of_classes = int(size_of_validation/n_classes)
            w = Window.partitionBy(F.col(col_name)).orderBy(F.col("rnd_"))

            balanced_validation = (filtered_df
                       .withColumn("rnd_", F.rand(seed=random_state))  # Add random numbers column
                       .withColumn("rn_", F.row_number().over(w))  # Add row number over window
                       .where(F.col("rn_") <= size_of_classes)  # Take n observations
                       .drop("rn_")  # drop helper columns
                       .drop("rnd_"))
            
            print("The script took " + str(datetime.now() - startTime) + " minutes") 
        
        return balanced_validation
                  
    def remove_small_classes(self, min_class_size=5):
        '''
        Filters classes with small class size from dataframe before sampling
        
        min_class_size:  integer to use to set the minimum instances in the data of each class before sampling 
        '''
        
        if self.code == "sic": 
            col_name = "industry_code"
        elif self.code == "soc": 
            col_name = "SOC2020_code"
        else: 
            raise ValueError("Code is not recongised. Please use either 'sic' or 'soc'")
        
        #get value counts of each class
        classes = pd.DataFrame(self.df[col_name].value_counts()).reset_index()
        
        #set column names
        classes.columns = [col_name,'Count']
        
        #filter classes with instances > class_size
        classes = classes[classes["Count"] > min_class_size]
        sample_classes = classes[col_name].unique()
        
        #filter dataframe so only classess with a sample greater than class size are included
        sample_df = self.df[self.df[col_name].isin(sample_classes)]
        
        return sample_df    

      
def compare_codes(code, full_df, df): 
    '''
    Compare codes from full census data with 1pc sample
    
    code: 'sic' or 'soc'
        
    '''
    #Define column names of full and sample codes based on input code
    if code == "soc": 
        col_name_full = "occupation_code"
        col_name_sample = "SOC2020_code"
    elif code == "sic": 
        col_name_full = "industry_code"
        col_name_sample = "industry_code"
    else: 
        raise ValueError("Code is not recognised. Please use either 'sic' or 'soc")

    # Check to ensure codes in all codes in census are in 1pc sample
    codes_in_1pc = list(df[col_name_sample].unique())
    codes_in_full = [i[col_name_full] for i in full_df.select(col_name_full).distinct().collect()]

    # Remove None items and punctuation to compare lists
    for x in [codes_in_full, codes_in_1pc]:
        if None in x:
            x.remove(None)

    codes_in_full = [''.join(c for c in s if c not in string.punctuation) for s in codes_in_full]

    # Compare length of lists
    if set(codes_in_full) != set(codes_in_1pc): 
        print("Difference in the codes in the full and the sample data")

        # Compare items in lists
        full_not_1pc_codes = np.setdiff1d(codes_in_full,codes_in_1pc)
        print("The different codes are: ")
        print(full_not_1pc_codes)
    else: 
        print("Number of codes in full data and sample data are equal")
    return codes_in_1pc, df.columns

  
def check_classes(df_training, df_validation, df_balanced_validation, code): 
    '''
        Function to check that all classes in the validation and balanced validation datasets can be found in the training data
    
    '''
    
    if code == "sic": 
        col_name = "industry_code"
    elif code == "soc": 
        col_name = "SOC2020_code"
    else: 
        raise ValueError("Code is not recognised. Please use either sic or soc")
    
    unique_classes_training = df_training[col_name].unique()
    unique_classes_validation = df_validation[col_name].unique()
    unique_classes_balanced_validation = df_balanced_validation[col_name].unique()
    
    bool_classes_validation = bool(set(unique_classes_validation) - set(unique_classes_training))
    bool_classes_validation_balanced = bool(set(unique_classes_balanced_validation) - set(unique_classes_training))
    
    if bool_classes_validation: 
        raise ValueError("There are classes in validation dataset that are not in the training data")
    if  bool_classes_validation_balanced:
        raise ValueError("There are classes in validation balanced dataset that are not in the training data")                 
    