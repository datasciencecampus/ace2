import os
import bz2

from pandas import read_pickle, read_excel, read_csv

from ace.utils.ace_exception import AceException


def get(doc_source_file_name):

    if not os.path.isfile(doc_source_file_name):
        raise AceException('file: ' + doc_source_file_name + ' does not exist in input folder')

    if doc_source_file_name.endswith('.pkl.bz2') or doc_source_file_name.endswith('.pkl'):
        return read_pickle(doc_source_file_name)
    elif doc_source_file_name.endswith('.xls'):
        return read_excel(doc_source_file_name)
    elif doc_source_file_name.endswith('.csv'):
        return read_csv(doc_source_file_name, engine='python', error_bad_lines=False, skipinitialspace=True)
    elif doc_source_file_name.endswith('.xlsx'):
        return read_excel(doc_source_file_name)
    else:
        raise AceException('Unsupported file: ' + doc_source_file_name)
           
def load_data_from_sql(sql_string, spark_session):
    return spark_session.sql(sql_string)

def load_data_from_file(file_path, spark_session):
    if file_path.endswith('.csv'):
        return spark_session.read.option("header", "true").csv(file_path)
    else:
        raise AceException('Unsupported file: ' + file_path)

def load_data(source_string, spark_session, data_source='file', limit=None, return_as_pandas=False):
    """
    Container function sending source string to either load_data_from_sql or load_data_from_file depending on data_source
    
    source_string: either sql string or file path
    spark_session
    data_source: {'file','sql'}
    """
    if data_source=='file':
        if limit is not None:
            df = load_data_from_file(source_string, spark_session).limit(limit)
        else:
            df = load_data_from_file(source_string, spark_session)
    elif data_source=='sql':
        if limit is not None:
            df = load_data_from_sql(source_string, spark_session).limit(limit)
        else:
            df = load_data_from_sql(source_string, spark_session)
    else:
        raise AceException(f'Unrecognised data source {data_source}')
    
    if return_as_pandas:
        return df.toPandas()
    else:
        return df
    
def save_data_to_hdfs(df, file_path, write_mode="error"):
    """
    Saves a pyspark dataframe from CDSW to HDFS area, or locally

    local: save the data locally or to HDFS area, bool
    file_path: write location
    write_mode: {'error','overwrite','append'}, mode of pyspark .write

    """
    df.write.mode(write_mode).csv(file_path, header = True)
    print(f"Data saved to {file_path}")    
    
    
def dump_pickle_file(object_to_dump, file_path):
    with bz2.BZ2File(file_path + '.pkl.bz2', 'wb') as pickle_file:
        pickle.dump(object_to_dump, pickle_file, protocol=4, fix_imports=False)        
    