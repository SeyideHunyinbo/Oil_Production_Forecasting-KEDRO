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
# pylint: disable=invalid-name

import logging
from typing import Dict, List

import warnings
warnings.simplefilter('ignore')

import pywt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score 

from kedro.io import (
    DataCatalog,
    CSVLocalDataSet,
    PickleLocalDataSet,
    HDFLocalDataSet
)


def split(data: pd.DataFrame) -> pd.DataFrame:
    label_names = ['Steam Flow Rate - Outer', 'Emulsion Flow Rate'] 
    input_data = data.drop(columns = label_names)
    labels = data[label_names]
    return [input_data, labels]

def validate(parameters: Dict): 
# def validate(dummy2, parameters: Dict): 
    import glob, os
    os.chdir(parameters["path_model_input"])
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
                filename: CSVLocalDataSet(filepath=parameters["path_model_input"]+"/"+file),
            }
        )
        well_data = io.load(filename)
        wells_data.append(well_data)
    Raw_Data_preprocessed = []
    for well, file, filename in zip (wells_data, files, filenames): 
        well = well[['Date', 'Injector Bottom Hole Pressure', 'Steam Flow Rate - Outer', 
                                    'Bottom Hole Heel Temperature', 'Emulsion Pressure', 'Producer Bottom Hole Pressure', 
                                    'ESP Speed', 'Emulsion Flow Rate']]
        for i in range(1,len(well.columns)):
            well[well.columns[i]] = pd.to_numeric(well[well.columns[i]], errors='coerce') 

        well = well.iloc[:1399]
        well = well.fillna(well.rolling(30, min_periods=1).median())
        well = well.fillna(well.median())
        
        well['Date'] = pd.to_datetime(well['Date'])
        well = well.set_index('Date')
        Raw_Data_preprocessed.append(well)
       
    os.chdir(parameters["path_val_stats"])
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
                static_filename: CSVLocalDataSet(filepath=parameters["path_val_stats"]+"/"+static_file),
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
    
    all_wells_input = []
    all_wells_labels = []
    for well_data, file in zip(Raw_Data_preprocessed, files):
        DWT_Aprox_coeff_input = []
        input_data, labels =  split(well_data)
               
        input_columns = list(input_data.columns)
        for data_idx in input_columns:
            signal = well_data[data_idx].values
            thresh = parameters["thresh"]*np.nanmax(signal)
            coeff = pywt.wavedec(signal, wavelet = parameters["wavelet"], mode=parameters["mode1"], level = parameters["level"])
            coeff[1:] = (pywt.threshold(i, value=thresh, mode=str(parameters["mode2"])) for i in coeff[1:])
            DWT_Aprox_coeff_input.append(coeff[0])
        DWT_Aprox_coeff_input = pd.DataFrame(np.transpose(DWT_Aprox_coeff_input), columns = input_columns)
        data_set_input = CSVLocalDataSet(filepath=parameters["path_val_pre_processed"]+"/validation_input_DWT_coeffs_"+file)
        data_set_input.save(DWT_Aprox_coeff_input)
        all_wells_input.append(DWT_Aprox_coeff_input.values)
        data_set_labels = CSVLocalDataSet(filepath=parameters["path_val_pre_processed"]+"/validation_labels_"+file)
        data_set_labels.save(labels)
        all_wells_labels.append(labels.values)
        
    #     Standardize dynamic data coeffs
    data_set_scaler_coeffs = PickleLocalDataSet(filepath=parameters["path_models"]+"/scaler_coeffs.pickle")
    scaler_coeffs = data_set_scaler_coeffs.load()
    all_wells_standardized_input = []
    all_wells_standardized_input_flattened = []
    for well_coeffs in all_wells_input:
        std_coeffs = scaler_coeffs.transform(well_coeffs)  
        all_wells_standardized_input.append(std_coeffs)     
        transposed_std_coeffs = np.transpose(std_coeffs)
        flattened_std_coeffs = transposed_std_coeffs.flatten()     
        all_wells_standardized_input_flattened.append(flattened_std_coeffs)   
    all_wells_standardized_input = np.array(all_wells_standardized_input)
    all_wells_standardized_input_flattened = np.array(all_wells_standardized_input_flattened)
    input_columns = list(DWT_Aprox_coeff_input.columns)
    for std_coeffs, file in zip(all_wells_standardized_input, files):
        std_coeffs = pd.DataFrame(std_coeffs, columns = input_columns)
        data_set = CSVLocalDataSet(filepath=parameters["path_val_pre_processed"]+"/validation_std_DWT_input_coeffs_"+file)
        data_set.save(std_coeffs)
#     Standardize static data   
    
    data_set_scaler_static = PickleLocalDataSet(filepath=parameters["path_models"]+"/scaler_static.pickle")
    scaler_static = data_set_scaler_static.load()
    all_wells_standardized_properties = scaler_static.fit_transform(properties)  
    all_wells_coeffs_reservoir_data = []
    for flattened_std_coeffs, standardized_properties in zip(all_wells_standardized_input_flattened, all_wells_standardized_properties):
        flattened_std_coeffs = list(flattened_std_coeffs)
        standardized_properties = list(standardized_properties)
        for reservoir_property in standardized_properties:
            flattened_std_coeffs.append(reservoir_property)      # append reservoir data to dynamic data coeffs
        all_wells_coeffs_reservoir_data.append(flattened_std_coeffs)
    all_wells_coeffs_reservoir_data = np.array(all_wells_coeffs_reservoir_data) 
    
    
    well_count = np.arange(len(all_wells_coeffs_reservoir_data))   
    daily_timesteps = np.arange(len(all_wells_labels[0]))
    input_data = []
    for coeff_inputs in all_wells_coeffs_reservoir_data:
        for time_lapse in daily_timesteps: 
            well_inputs = [time_lapse] + list(coeff_inputs)            # append time lapse to input data
            input_data.append(well_inputs)
    input_data = np.array(input_data)       
    data_set_regressor_1 = PickleLocalDataSet(filepath=parameters["path_models"]+"/regressor_1.pickle")
    regressor_1 = data_set_regressor_1.load()
    number_of_wells = len(well_count)
    wells_steam_rate_predicted = regressor_1.predict(input_data)
    wells_steam_rate_predicted = wells_steam_rate_predicted.reshape((number_of_wells, 1399)).T
    

    # prediction inputs to model 2
    input_data_model_2 = []
    for coeff_inputs, well in zip(all_wells_coeffs_reservoir_data, well_count):
        for time_lapse in daily_timesteps: 
            well_inputs = [time_lapse] + list(coeff_inputs)            # append time lapse to input data
            well_inputs_model_2 = [wells_steam_rate_predicted[time_lapse, well]] + well_inputs
            input_data_model_2.append(well_inputs_model_2)
    input_data_model_2 = np.array(input_data_model_2)
    
    data_set_regressor_2 = PickleLocalDataSet(filepath=parameters["path_models"]+"/regressor_2.pickle")
    regressor_2 = data_set_regressor_2.load()
    wells_emulsion_rate_predicted = regressor_2.predict(input_data_model_2)
    wells_emulsion_rate_predicted = wells_emulsion_rate_predicted.reshape((number_of_wells, 1399)).T   
    
    # actual targets
    all_wells_steam_data = []
    all_wells_emulsion_data = []
    for ID in well_count:        
        well_steam_data = all_wells_labels[ID][:,0]   
        well_emulsion_data = all_wells_labels[ID][:,1]  
        all_wells_steam_data  = all_wells_steam_data + list(well_steam_data)
        all_wells_emulsion_data  = all_wells_emulsion_data + list(well_emulsion_data)
    all_wells_steam_data = np.array(all_wells_steam_data)
    all_wells_emulsion_data = np.array(all_wells_emulsion_data)
    wells_steam_rate_actual = all_wells_steam_data.reshape((number_of_wells, 1399)).T
    wells_emulsion_rate_actual = all_wells_emulsion_data.reshape((number_of_wells, 1399)).T 
    
    
    print("Prediction Performance:\n")
    print("Steam Flow Rate:")
    for well, file in zip(well_count, files): 
        steam_rate_predicted = wells_steam_rate_predicted[:,well]
        steam_rate_actual = wells_steam_rate_actual[:,well]
        steam_rate_actual_predicted = pd.DataFrame(np.vstack((steam_rate_actual, steam_rate_predicted)).T, 
                                                   columns = ["steam rate actual", "steam rate predicted"])
        data_set_steam_rate = CSVLocalDataSet(filepath=parameters["path_model_output"]+"/steam_rate_"+file)
        data_set_steam_rate.save(steam_rate_actual_predicted)
        print(file+" R_squared: {0:.4f}".format(r2_score(steam_rate_actual, steam_rate_predicted)))
    
    print("\n")
    print("Emulsion Flow Rate:")
    for well, file in zip(well_count, files): 
        emulsion_rate_predicted = wells_emulsion_rate_predicted[:,well]
        emulsion_rate_actual = wells_emulsion_rate_actual[:,well]
        emulsion_rate_actual_predicted  = pd.DataFrame(np.vstack((emulsion_rate_actual, emulsion_rate_predicted)).T, 
                                                   columns = ["emulsion rate actual", "emulsion rate predicted"])
        data_set_emulsion_rate = CSVLocalDataSet(filepath=parameters["path_model_output"]+"/emulsion_rate_"+file)
        data_set_emulsion_rate.save(emulsion_rate_actual_predicted)
        print(file+" R_squared: {0:.4f}".format(r2_score(emulsion_rate_actual, emulsion_rate_predicted)))
         
    dummy_validate = files 
    return dummy_validate


   # Cross_validation
