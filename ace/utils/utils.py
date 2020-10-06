import os
import csv
import glob

from pyspark.sql import SparkSession
from datetime import datetime
from memory_profiler import profile, memory_usage
import matplotlib.pyplot as plt
import pandas as pd

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

def spark():        
    """Use this method to get the reference to the Spark session"""
    return (SparkSession.builder.appName("mltool")
         .config("spark.executor.memory", "40g")
         .config("spark.executor.cores", 3)
         .config("spark.dynamicAllocation.maxExecutors", 40)
         .config("spark.dynamicAllocation.enabled", "true")
         .config("spark.shuffle.service.enabled", "true")
         .config("spark.shuffle.io.maxRetries", 10)
         .config("spark.sql.execution.arrow.maxRecordsPerBatch", 5000)
         .config("spark.kryoserializer.buffer.max", "256m") 
         .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())


class MemoryMonitor:
    def __init__(self, func, args = None, alg_type = None, dirname = None, runtype = None, size = None, inv = 1):
        self.func = func
        self.args = args 
        self.dt = datetime.now().strftime("%d%m%Y_%H:%M:%S")
        self.directory = "./outputs/memory_profile/"
        self.inv = inv 
        self.alg_type = alg_type
        self.size = size
        self.runtype = runtype
        
        if self.args is not None: 
            try:
                self.folder = self.args.code
            except AttributeError: 
                self.folder = self.args.dirname
        else: 
            self.folder = dirname

    def create_folder(self): 
        """
            Create filepaths location for memory proilfer results 
        
        """

        filepaths = [self.directory + "failed_models/", self.directory + self.folder + "/"]  
        for f in filepaths: 
            if not os.path.exists(f):
                os.makedirs(f)
            
    def create_file(self):
        
        """
            Create files to record the models/scripts being tested. The script/models are recorded as failing first, 
            if the scripts run to completition, then the models/scripts are removed from the failed list.
        
        """
        
        if self.alg_type != None: 
            if not os.path.isfile(self.directory +"failed_models/failed_models_" + self.folder + ".csv"): 
                df = pd.DataFrame(columns = ["Datetime", "Model Description", "Size", "Runtype"])
                df.to_csv(self.directory + "failed_models/failed_models_" + self.folder + ".csv")
            with open(self.directory + "failed_models/failed_models_" + self.folder + ".csv", "a+", newline = "") as write_obj:
                csv_write = csv.writer(write_obj)
                csv_write.writerow([str(self.dt), str(self.alg_type), str(self.size), str(self.runtype)]) 
        else: 
            file = open(self.directory + self.folder + "/" + self.dt + "_" + self.args.filename + ".log", "w+")
            file.write(self.dt + "\n")
            file.write(str(vars(self.args)) + "\n")
            file.write("Process has not been completed.")
            file.close()
            return file
   
                
    def profile_memory(self, file = None): 
        """
            Stream memory consumed in function to file.

        """
        
        if not self.alg_type: 
            func = profile(stream=file)(self.func)
            return func
        else: 
            mem_usage = memory_usage(self.func, interval = self.inv)
            failure = False
            return mem_usage, failure
    
    def remove_lines_from_file(self, excluded_line_log, excluded_line_csv = None): 
        """
            This function removes specified lines from a file if a conidtion is met. 
        """
            
        latest_file = max(glob.glob(self.directory + self.folder + "/*") , key=os.path.getctime)    
        with open(latest_file, "r") as f: 
            lines = f.readlines()
            with open(latest_file, "w") as f: 
                for line in lines: 
                    if line.strip("\n") != excluded_line_log: 
                        f.write(line)
        
        if self.alg_type: 
            filename = self.directory + "/failed_models/failed_models_" + self.folder + ".csv"
            lines = list()
            with open(filename, 'r') as readFile: 
                reader = csv.reader(readFile, delimiter = "\t")
                for row in reader: 
                    lines.append(row)
                    for field in row: 
                        if field.startswith(excluded_line_csv):
                            lines.remove(row)
                            
            with open(filename, 'w') as writeFile: 
                writer = csv.writer(writeFile)
                writer.writerows(lines)
                                        
    def memory_graph(self, mem_use): 
        """ 
            Saves graph of memory useage for ml classification functions specifically

        """    
        fig, axs = plt.subplots(1, 1)
        axs.plot(list(range(len(mem_use))), mem_use)
        plt.xlabel("Time (s)")
        plt.ylabel("Memory usage (MB)")
        if type(self.alg_type) is list: 
            alg_type = self.alg_type[0]
        else: 
            alg_type = self.alg_type
        plt.title("Memory useage of " + str(alg_type) + " classifier with " + "\n" + str(self.size) + " rows of data. (" + self.runtype + ")")
        plt.savefig("./today_test.png")
        max_memory_useage = str(max(mem_use)) +  " MB"

        return max_memory_useage
    
        
    def save_memory_output(self):
        """
            Save a log of memory useage for funciton to file. 
            If the memory stream is completed, then the arguments are removed from the 
            failed models file".

        """
        if not self.alg_type:
            #Create output folder
            self.create_folder()
        
        #create log file - this file is to store a line by line memory record of a function 
        file = self.create_file()
        # If you are recording memory of a ml algorithm specfically
        if self.alg_type: 
            #create failed models - this keeps a record of the model that is being tested
            self.create_file()
            
            # record the memory usage during the process (the function actually runs at this point)
            # if the script is killed due to memory it will happen here
            # if the process is completed then it will return the variable "failure = False"
            mem_usage, failure = self.profile_memory()
            
            #create a graph with the memory usage (default every 1s)
            # returns the maximum memory used in the process
            max_memory = self.memory_graph(mem_usage)
            
            # If the prcoess has run succesfully,remove the model from the "failed_models" file,  
            # and the line in the log file specifing that the process has not completed 
            if failure == False:
                self.remove_lines_from_file("Process has not been completed.", str(self.dt))
                return max_memory 
        # If this is a general memory test
        # wrap profiler around function
        else:
            file = open(self.directory + self.folder + "/" + self.dt + "_" + self.args.filename + ".log", "a+")
            func = profile(stream=file)(self.func)
            return func



