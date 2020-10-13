# Copyright 2018-2019 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""

import logging
from typing import Dict, List

import warnings
warnings.simplefilter('ignore')

import pywt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sc

from collections import Counter

from kedro.io import (
    DataCatalog,
    CSVLocalDataSet,
    PickleLocalDataSet,
    HDFLocalDataSet
)
from analytics_new.io.xls_local import (
    ExcelLocalDataSet,    
) 
# import matplotlib.pyplot as plt
# from kedro.extras.datasets.matplotlib import MatplotlibWriter


# def nan_removal(data: pd.DataFrame) -> pd.DataFrame:
#     data = data.dropna(axis=0)
#     return data 


def preprocess_raw_data(parameters: Dict):
    import glob, os
    os.chdir(parameters["path_raw"])
    files = []
    for file in glob.glob("*.csv"):
        files.append(file)            
    filenames = []
    wells_data = []
    for file in files:
        filename, extension = file.split('.')
        filenames.append(filename)  
    for file, filename in zip(files, filenames):
        io = DataCatalog(
            {
                filename: CSVLocalDataSet(filepath=parameters["path_raw"]+"/"+file),
            }
        )
        well_data = io.load(filename)
        wells_data.append(well_data)
    Raw_Data_preprocessed = []
    Raw_Data_dated = []
    for well, file, filename in zip (wells_data, files, filenames): 
        well = well[['Date', 'Injector Bottom Hole Pressure', 'Steam Flow Rate - Outer', 
                                    'Bottom Hole Heel Temperature', 'Emulsion Pressure', 'Producer Bottom Hole Pressure', 
                                    'ESP Speed', 'Emulsion Flow Rate']]
        for i in range(1,len(well.columns)):
            well[well.columns[i]] = pd.to_numeric(well[well.columns[i]], errors='coerce') 

        well = well.iloc[:1399]
        well = well.fillna(well.rolling(30, min_periods=1).median())
        well = well.fillna(well.median())
        
        well_dated = well.copy()
        well_dated["Well"] = filename                                                                    # create a column for well name
        data_set = CSVLocalDataSet(filepath=parameters["path_intermediate"]+"/pre_processed_data_"+file)
        data_set.save(well_dated)
        Raw_Data_dated.append(well_dated)
        
        well['Date'] = pd.to_datetime(well['Date'])
        well = well.set_index('Date')
        Raw_Data_preprocessed.append(well)
       

    os.chdir(parameters["path_raw_static"])
    static_files = []
    for static_file in glob.glob("*.csv"):
        static_files.append(static_file)            
    static_filenames = []
    statics_data = []
    for static_file in static_files:
        static_filename, others = static_file.split('_')
        static_filenames.append(static_filename) 
    for static_file, static_filename in zip(static_files, static_filenames):
        io = DataCatalog(
            {
                static_filename: CSVLocalDataSet(filepath=parameters["path_raw_static"]+"/"+static_file),
            }
        )
        static_data = io.load(static_filename)
        statics_data.append(static_data)
    statics_data_new = []
    well_name_list = []
    for pad_static in statics_data:  
        well_name = pad_static['WELLPAIR_NAME'].values
        well_name_list.append(well_name)
        pad_static = pad_static.set_index('WELLPAIR_NAME')
        pad_static = pad_static.drop(columns = ['PLAN_NAME','HIGH_PRESSURE']) 
        statics_data_new.append(pad_static)
    properties = []
    probabilities = []
    asset_names = []
    for pad_static, names in zip(statics_data_new, well_name_list):
        for well in names:
            prob = pad_static.loc[well, 'Forecast_Prob']
            probabilities.append(prob)
            pad_code = pad_static.loc[well, 'PAD_CODE']
            asset_name, pad = pad_code.split('_')
            asset_names.append(asset_name)
            property_ = pad_static.loc[well, 'SAGD_PRESSURE':'BOTTOM_WATER_THICKNESS'].values
            properties.append(property_)
    properties = np.array(properties)
    return [Raw_Data_preprocessed, Raw_Data_dated, files, filenames, probabilities, asset_names, properties] 


def create_master_table(Raw_Data_dated: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    master_table = pd.concat(Raw_Data_dated, axis=1, sort=False) 
    data_set = CSVLocalDataSet(filepath=parameters["path_primary"]+"/master_table.csv")
    data_set.save(master_table)
    return master_table

    
def split(data: pd.DataFrame) -> pd.DataFrame:
    label_names = ['Steam Flow Rate - Outer', 'Emulsion Flow Rate'] 
    input_data = data.drop(columns = label_names)
    labels = data[label_names]
    return [input_data, labels]

  
def discrete_wavelet_transform(Raw_Data_preprocessed: pd.DataFrame, parameters: Dict, files: List):
    for well_data, file in zip(Raw_Data_preprocessed, files):
        list_input_DWT_Aprox_coeff = []
        input_data, labels =  split(well_data)
               
        input_columns = list(input_data.columns)
        for data_idx in input_columns:
            signal = well_data[data_idx].values
            thresh = parameters["thresh"]*np.nanmax(signal)
            coeff = pywt.wavedec(signal, wavelet = parameters["wavelet"], mode=parameters["mode1"], level = parameters["level"])
            coeff[1:] = (pywt.threshold(i, value=thresh, mode=str(parameters["mode2"])) for i in coeff[1:])
            list_input_DWT_Aprox_coeff.append(coeff[0])
        list_input_DWT_Aprox_coeff = pd.DataFrame(np.transpose(list_input_DWT_Aprox_coeff), columns = input_columns)
        data_set_input = CSVLocalDataSet(filepath=parameters["path_primary"]+"/input_DWT_coeffs_"+file)
        data_set_input.save(list_input_DWT_Aprox_coeff)
        data_set_labels = CSVLocalDataSet(filepath=parameters["path_primary"]+"/labels_"+file)
        data_set_labels.save(labels)
    dummy = labels
    return dummy
        

def resample(data: pd.DataFrame) -> pd.DataFrame:
    weekly = data.resample('W', convention='start').asfreq()
    weekly_no_NAN = weekly.dropna(axis=0) 
    return weekly_no_NAN


# def B_Splines(dummy, Raw_Data_preprocessed: pd.DataFrame, parameters: Dict, files: List):    
#     for well_data, file in zip(Raw_Data_preprocessed, files):
#         list_input_spline_coeff = []        
#         weekly_no_NAN =  resample(well_data)
#         input_data, labels =  split(weekly_no_NAN)
#         timesteps = np.arange(len(weekly_no_NAN)) + 1 
        
#         input_columns = list(input_data.columns)
#         for data_idx in input_columns:
#             knots, spline_coeff, degree = sc.interpolate.splrep(timesteps, input_data[data_idx].values, s=parameters["s"], k=parameters["spline_degree"])
#             list_input_spline_coeff.append(spline_coeff)
#         list_input_spline_coeff = pd.DataFrame(np.transpose(list_input_spline_coeff), columns = input_columns)
#         data_set_input = CSVLocalDataSet(filepath=parameters["path_primary"]+"/input_spline_coeffs_"+file)
#         data_set_input.save(list_input_spline_coeff)
#     dummy_Spline = labels
#     return dummy_Spline

        
# def principal_component_analysis(dummy_Spline, Raw_Data_preprocessed: pd.DataFrame, parameters: Dict, files: List, filenames: List):
#     from sklearn.decomposition import PCA
#     from sklearn.preprocessing import StandardScaler
    
#     all_wells_input = []
#     for file in files:
#         data_set_input = CSVLocalDataSet(filepath=parameters["path_primary"]+"/input_DWT_coeffs_"+file)
#         DWT_Aprox_coeff_input = data_set_input.load()
#         all_wells_input.append(DWT_Aprox_coeff_input.values)
    
#     principal_scores_all = []
#     variance_explained_percent_all = []
#     for input_data, file, filename in zip(all_wells_input, files, filenames):   
# #         well_data = well_data.drop(['Well'], axis=1, inplace=True)
#         std_data = input_data - np.nanmean(input_data, axis=0)
# #         std_data = StandardScaler().fit_transform(input_data)
#         pca = PCA(n_components=parameters["n_components"])
#         pca.fit(std_data)
#         transformed_data = pca.transform(std_data)
#         X_pca_data = pca.inverse_transform(transformed_data)
#         principal_scores = pd.DataFrame(transformed_data, columns = ["A", "B", "C", "D", "E"])
#         principal_scores_all.append(principal_scores)
#         data_set_scores = CSVLocalDataSet(filepath=parameters["path_intermediate"]+"/principal_scores_"+file)
#         data_set_scores.save(principal_scores)
#         variance_explained_percent = pca.explained_variance_ratio_ * 100
# #         print("varianced_explained_"+filename+" : ", variance_explained_percent)
#         variance_explained_percent_all.append(variance_explained_percent)
        
#         total_variance_exp = 0
#         total_variance_exp_vector = []
#         components_array = np.arange(len(variance_explained_percent)) + 1
#         for i in range(0, len(components_array)):
#             total_variance_exp += variance_explained_percent[i]
#             total_variance_exp_vector.append(total_variance_exp)
        
#         # Saving plots
#         fig, ax = plt.subplots(figsize=(10,8))
#         ax.plot(components_array, variance_explained_percent, 'r')
#         ax2 = ax.twinx()
#         ax2.plot(components_array, total_variance_exp_vector, 'b')
#         plot_writer = MatplotlibWriter(filepath=parameters["path_model_output"]+"/elbow_plot_"+file)
#         plot_writer.save(ax)
#         plot_writer.save(ax2)
#         plt.close()
