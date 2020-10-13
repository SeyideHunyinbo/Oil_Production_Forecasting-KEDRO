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
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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
    wells_life = []
    wells_data_ = []
    for well in wells_data:
#         well = well[['Date', 'Injector Bottom Hole Pressure',
#                    'Producer Bottom Hole Pressure', 'ESP Speed',
#                    'Steam Flow Rate - Outer',
#                    'Emulsion Flow Rate']]
        
        well = well[['Date', 'Speed [Hz]', 'Current [A]', 'IBHP', 'PBHP', 
                     'Co-Injection [E3m3/d]', 'Oil [bbl/d]', 'Steam [m3/d]', 
                     'Emulsion [m3/d]']]

        for i in range(1,len(well.columns)):
            well[well.columns[i]] = pd.to_numeric(well[well.columns[i]], errors='coerce') 
        well['Prod_Date'] = pd.to_datetime(well['Date'])
        well = well.set_index('Prod_Date')
#         well = well.resample('7D').mean()   # weekly data 
#         well = well.resample('30D').mean()   # monthly data
#         well = well.rolling(30, min_periods=1).mean()

        data = well['Oil [bbl/d]'] / 6.28981
        well.insert(4, 'Oil [m3/d]', data)   
        time_data = np.arange(len(well))
        well.insert(0, 'Timestep', time_data) 
        
        wells_life.append(len(well))
        wells_data_.append(well)
    min_well_length = np.min(np.array(wells_life))
    print((min_well_length, np.argmin(np.array(wells_life))))
    timesteps = min_well_length
#     timesteps = 371
    
    for well, file, filename in zip (wells_data_, files, filenames):
        well = well.iloc[:timesteps]   # daily, weekly, monthly data 
#         well = well.fillna(0)        
#         well = well.fillna(well.rolling(30, min_periods=1).median())
#         well = well.fillna(well.median())
        
        well["Well"] = filename                                              # create a column for well name
        well = well.reset_index(drop=True)                                   # remove date index
        data_set = CSVLocalDataSet(filepath=parameters["path_intermediate"]+"/pre_processed_data_"+file)
        data_set.save(well)
        Raw_Data_dated.append(well)
        well = well.drop(columns = ['Date', 'Well'])
        Raw_Data_preprocessed.append(well)
    
    stats_training = CSVLocalDataSet(filepath=parameters["path_raw_static"]+"/static_P50_data_training.csv")
    stats = stats_training.load()
    stats_ROIP = stats.loc[:, 'ROIP']
    stats = stats.loc[:, 'Effective_Length':'BottomWater_Oil_Saturation']

# #     using only rich geoostats and no bottom water properties
#     stats = stats.loc[:, 'Effective_Length':'Rich_Oil_Saturation']
    
# #     Using "Effective_Rich_Pay_Thickness" to account for standoff and rich thickness
#     data = stats['Rich_Pay_Thickness'] - stats['Stand_Off']
#     stats.insert(3, 'Effective_Rich_Pay_Thickness', data)
#     stats = stats.drop(columns = ['Rich_Pay_Thickness', 'Stand_Off'])
   
    property_names = list(stats.columns)
    properties = list(stats.values)

    properties = np.array(properties)
    return [timesteps, Raw_Data_preprocessed, Raw_Data_dated, files, filenames, properties, stats_ROIP, property_names] 



def create_master_table(Raw_Data_dated: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    master_table = pd.concat(Raw_Data_dated, axis=1, sort=False)
    data_set = CSVLocalDataSet(filepath=parameters["path_primary"]+"/master_table.csv")
    data_set.save(master_table)
    return master_table


def dynamic_time_warping(Raw_Data_preprocessed, parameters):
    reference_well = CSVLocalDataSet(filepath=parameters["path_raw"]+"/B03-1P.csv")
    well_ref = reference_well.load()   
    data = well_ref['Oil [bbl/d]'] / 6.28981
    well_ref.insert(4, 'Oil [m3/d]', data)
    well_ref_oil_data = well_ref['Oil [m3/d]'].values
    
    Raw_Data_preprocessed_ = []
    distance_array = []
    for well_data in Raw_Data_preprocessed:
        well_oil_data = well_data['Oil [m3/d]'].values
        
        distance, path = fastdtw(well_ref_oil_data, well_oil_data, dist=euclidean)
        distance_array.append(distance)
        path = np.array(path)
        index_well = path[...,1]
        index_ref_well = path[...,0]
        well = well_data.iloc[index_well]
        well.insert(0, 'index_ref', index_ref_well)
        well = well.groupby('index_ref').mean()
#         well = well.reset_index(drop=True) 
        Raw_Data_preprocessed_.append(well)
    
    distance_array = np.array(distance_array)
    return [distance_array, Raw_Data_preprocessed_]

              
def split(data: pd.DataFrame) -> pd.DataFrame:
#     steam_input_names = ['Timestep', 'Injector Bottom Hole Pressure']
#     emulsion_input_names = ['Timestep', 'Injector Bottom Hole Pressure', 'ESP Speed']
    steam_input_names = ['Timestep', 'IBHP']
    emulsion_input_names = ['Timestep', 'Speed [Hz]']
    steam_input_data = data[steam_input_names]
    emulsion_input_data = data[emulsion_input_names]
#     label_names = ['Steam Flow Rate - Outer', 'Emulsion Flow Rate'] 
    label_names = ['Steam [m3/d]', 'Oil [m3/d]']
    labels = data[label_names]
    return [steam_input_data, emulsion_input_data, labels]


def save_well_data(Raw_Data_preprocessed, parameters, files):
    for well_data, file in zip(Raw_Data_preprocessed, files):
# def save_well_data(Raw_Data_preprocessed_, parameters, files):
#     for well_data, file in zip(Raw_Data_preprocessed_, files):
        steam_input_data, emulsion_input_data, labels =  split(well_data)
        data_set_steam_input = CSVLocalDataSet(filepath=parameters["path_primary"]+"/steam_inputs_"+file)
        data_set_steam_input.save(steam_input_data)
        data_set_emulsion_input = CSVLocalDataSet(filepath=parameters["path_primary"]+"/emulsion_inputs_"+file)
        data_set_emulsion_input.save(emulsion_input_data)
        data_set_labels = CSVLocalDataSet(filepath=parameters["path_primary"]+"/labels_"+file)
        data_set_labels.save(labels)
    dummy = labels
    return dummy






    
    
    
    
        
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
