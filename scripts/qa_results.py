from os import path

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from sklearn.metrics import classification_report, accuracy_score

from ace.utils.utils import spark

spark_session = spark()

def evaluate_sic_soc_by_class(code, balanced=False, matched_only=True):
    """
    Creates a classification report showing the results per class of the sic soc coding tool
    Outputs have been saved on HDFS
    
    params:
    code: {'sic','soc'}
    balanced: Use validation set with evenly balanced classes, bool, default: False
    matched_only: Include/Exclude those under the threshold
    """
    
    if code == 'sic':
        true_y_label = 'industry_code'
        pred_y_label = 'SIC2007_prototype'
        error_codes_list = ['VVVV','XXXX','YYYY','ZZZZ']
    elif code == 'soc':
        true_y_label = 'SOC2020_code'
        pred_y_label = 'SOC2020_predicted'
        error_codes_list = None
    else:
        raise('Code should be either sic or soc')
    hdfs_filepath = '/dapsen/workspace_zone/sic_soc/dsc/benchmark/' + code
    if balanced:
        hdfs_filepath = hdfs_filepath + '_b'
        
    benchmark_df = spark_session.read.parquet(hdfs_filepath)
    
    new_true_y_label = true_y_label + '_2'
    new_pred_y_label = pred_y_label + '_2'
    benchmark_df = benchmark_df.withColumn(new_true_y_label,F.rpad(F.col(true_y_label),4,'0'))
    if error_codes_list:
        benchmark_df = benchmark_df.withColumn(new_true_y_label,F.when(F.col(new_true_y_label).isin(error_codes_list), '-1').otherwise(F.col(new_true_y_label)))
    benchmark_df = benchmark_df.withColumn(new_pred_y_label, F.when(F.col(pred_y_label).isin(['-6','-1']), '-1').otherwise(F.col(pred_y_label)))
    
    if matched_only:
        benchmark_df = benchmark_df.filter(F.col(new_pred_y_label) != '-1')
    y_true = np.array(benchmark_df.select(new_true_y_label).collect())
    y_pred = np.array(benchmark_df.select(new_pred_y_label).collect())
    
    return evaluation_results(y_true, y_pred)

  
def evaluate_ml_by_class(file_path, matched_only):
    """
    Creates a classification report showing the results per class of the ml coding tool

    params:
    file_path: string of file location where prediction results are found e.g. ./outputs/soc/...
    matched_only: Include/Exclude those under the threshold
    """
    ml_df = pd.read_csv(file_path,index_col=0)
    
    if matched_only:
        ml_df = ml_df[ml_df['matched'] == 1]
    y_true = ml_df['true_label']
    y_pred = ml_df['prediction_labels']
    
    def _rpad_ml(column_name):
        ml_df[column_name] = ml_df[column_name].astype('str').str.pad(4, side='right',fillchar='0')
        ml_df.loc[ml_df[column_name] == '-100', column_name] = '-1'
    _rpad_ml('true_label')
    _rpad_ml('prediction_labels')
    
    return evaluation_results(y_true, y_pred)

  
def evaluation_results(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    results = pd.DataFrame(report).transpose().reset_index()
    results.rename(columns={'index':'code'},inplace=True)
    results = results[~results.code.isin(['accuracy','macro avg','weighted avg'])]
    return results


def qa_results_by_class(code, ml_file_path, weighted=True, balanced=False, matched_only = True, cols_to_rank = ['precision','recall','f1-score']):
    """
    Compares sic soc tool output with ml output by class
    
    params:
    code: {'sic','soc'}, str
    ml_file_path: file location where prediction results are found e.g. ./outputs/soc/.., str
    weighted: prioritises codes with more records, bool
    balanced: use random or balanced validation set, bool
    matched_only: Include/Exclude those under the threshold, bool
    cols_to_rank:[{'precision','recall','f1-score'}] list of columns to include in the rank, list
    """
    suffixes = ['_sic-soc','_ml']
    ml_results = evaluate_ml_by_class(ml_file_path, matched_only)
    sic_soc_results = evaluate_sic_soc_by_class(code=code, balanced=balanced, matched_only=matched_only)
    qa_results = pd.merge(sic_soc_results, ml_results, on='code',suffixes=suffixes)
    
    # Creates a rank, with those underperforming relative to sic soc having the highest rank
    rank_cols = []
    for col in cols_to_rank:
        rank_col = col + suffixes[1] + '_rank'
        qa_results[rank_col] = (qa_results[col + suffixes[1]]-qa_results[col + suffixes[0]]).rank(ascending=False)
        rank_cols.append(rank_col)
    
    # Rank prioritises codes with more records
    if weighted:
        rank_col = 'support' + suffixes[0] + '_rank'
        rank_cols.append(rank_col)
        qa_results[rank_col] = qa_results['support' + suffixes[0]].rank(ascending=True)
        
    qa_results['overall_rank'] = qa_results[rank_cols].mean(axis=1)
    performance = qa_results.sort_values('overall_rank', ascending=False).reset_index(drop=True)
    return performance


def join_compare_systems(code,
                         ml_file_path,
                         output_path,
                         classifier,
                         balanced=False,
                         id_col="id"):
    """
    Compares results of ML tool to results of SIC SOC tool by creating a combined dataframe
    then finding the degree of overlap.
    
    Assumes existence of SIC SOC tool coded data.
    params:
    code: {'sic','soc'}, str
    ml_file_path: file location where prediction results are found e.g. ./outputs/soc/.., str
    output_path: directory where all output graphs and tables are dumped
    classifier: str specifying the model being evaluated
    balanced: use random or balanced validation set, bool
    id_col: the column of unique record identifiers, should be the same column name for SIC SOC coding tool predictions and for ML predictions
    
    """    
    # Load the ML tool output for the comparison
    ml_sdf = spark_session.createDataFrame(pd.read_csv(ml_file_path))
    #              .withColumnRenamed(id_col, "mlid")
    
    # Load the benchmark SIC SOC tool output for the comparison
    if code == 'sic':
        true_col = 'industry_code'
        pred_col = 'SIC2007_prototype'
        error_codes_list = ['VVVV','XXXX','YYYY','ZZZZ']
    elif code == 'soc':
        true_col = 'SOC2020_code'
        pred_col = 'SOC2020_predicted'
        error_codes_list = None
    else:
        raise('Code should be either sic or soc')
    hdfs_filepath = '/dapsen/workspace_zone/sic_soc/dsc/benchmark/' + code
    
    if balanced:
        hdfs_filepath = hdfs_filepath + '_b'
        classifier = classifier + '_b'
        
    tool_sdf = spark_session.read.parquet(hdfs_filepath)
    
    #ml_sdf.mlid == tool_sdf.id .drop("mlid")
    compare_sdf = ml_sdf.join(tool_sdf, on=[id_col], how="inner")
      
    compare_sdf.cache()
    compare_sdf.count()
    
    # Defined on ML table columns for consistent formatting
    compare_sdf = compare_sdf.withColumn("ml_right", F.when(F.col("prediction_labels") == F.col("true_label"), True)\
                                                      .otherwise(False))
    
    # Defined on tool's columns for consistent formatting
    compare_sdf = compare_sdf.withColumn("tool_right", F.when(F.col(pred_col) == F.col(true_col), True)\
                                                        .otherwise(False))
    
    # Where both succeeded
    compare_sdf = compare_sdf.withColumn("both_right", F.when((F.col("ml_right") == True) & (F.col("tool_right") == True), True)\
                                                         .otherwise(False))
    
    # Where either succeeded
    compare_sdf = compare_sdf.withColumn("either_right", F.when((F.col("ml_right") == True) | (F.col("tool_right") == True), True)\
                                                         .otherwise(False))
    
    # Where just one succeeded (fun fact, pSpark has no XOR operator!)
    compare_sdf = compare_sdf.withColumn("one_right", F.when(F.col("ml_right") != F.col("tool_right"), True)\
                                                        .otherwise(False))
    
    compare_sdf.cache()
    compare_sdf.count()
    
    # ----- Produce overlap table ----- #
    df = compare_sdf[[id_col, true_col, 'ml_right', 'tool_right', 'both_right', 'either_right', 'one_right']]\
                    .toPandas()
    
    df['support'] = 1
    
    overlap_df = df.groupby(true_col)\
                   .agg({"ml_right":"sum",
                         "tool_right":"sum",
                         "both_right":"sum",
                         "either_right":"sum",
                         "support":"count"})\
                   .reset_index()
    
    overlap_df['overlap'] = 100.0 * overlap_df['both_right'] / overlap_df['either_right']
    
    overlap_df.sort_values('support', ascending=False)\
              .to_csv(path.join(output_path, classifier + "_tools_overlap.csv"), index=False)
    
    # ----- What did ML do better? ----- #
    examine_df = compare_sdf.filter(F.col("ml_right") == True)\
                            .filter(F.col("tool_right") == False)\
                            .filter(F.col(pred_col) != "-6")\
                            .toPandas()
    
    # Clean out intermediate columns
    examine_df.drop([col for col in df.columns if "right" in col], axis=1)\
              .to_csv(path.join(output_path, classifier + "_ml_got_right.csv"), index=False)
    
    # ----- What did the coding tool do better? ----- #
    examine_df = compare_sdf.filter(F.col("ml_right") == False)\
                            .filter(F.col("tool_right") == True)\
                            .filter(F.col(pred_col) != "-6")\
                            .toPandas()
    
    # Clean out intermediate columns
    examine_df.drop([col for col in df.columns if "right" in col], axis=1)\
              .to_csv(path.join(output_path, classifier + "_tool_got_right.csv"), index=False)
    
    return 0
  