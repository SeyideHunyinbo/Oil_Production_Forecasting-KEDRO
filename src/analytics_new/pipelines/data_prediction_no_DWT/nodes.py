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
#     steam_input_names = ['Date', 'Timestep', 'Injector Bottom Hole Pressure']
#     emulsion_input_names = ['Timestep', 'Injector Bottom Hole Pressure', 'ESP Speed']
    steam_input_names = ['Date', 'Timestep', 'IBHP']
    emulsion_input_names = ['Timestep', 'Speed [Hz]']
    steam_input_data = data[steam_input_names]
    emulsion_input_data = data[emulsion_input_names]
#     label_names = ['Steam Flow Rate - Outer', 'Emulsion Flow Rate'] 
    label_names = ['Steam [m3/d]', 'Oil [m3/d]']
    labels = data[label_names]
    return [steam_input_data, emulsion_input_data, labels]


# def load_well_validation_data(parameters: Dict):
#     timesteps = 1008
def load_well_validation_data(dummy2, timesteps, parameters: Dict):
    import glob, os
#     os.chdir(parameters["path_model_input"])
    os.chdir(parameters["path_model_input_matlab"])
    files_val = []
    for file in glob.glob("*.csv"):
        files_val.append(file)            
    filenames_val = []
    wells_data = []
    for file in files_val:
        filename, extension = file.split('.')
        filenames_val.append(filename)  
    for file, filename in zip(files_val, filenames_val):
        io = DataCatalog(
            {
#                 filename: CSVLocalDataSet(filepath=parameters["path_model_input"]+"/"+file),
                filename: CSVLocalDataSet(filepath=parameters["path_model_input_matlab"]+"/"+file),
            }
        )
        well_data = io.load(filename)
        wells_data.append(well_data)
    Raw_Data_preprocessed_val = []
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
#         well = well.dropna(axis=0)   # may change
#         well = well.resample('7D').mean()   # weekly data 
#         well = well.resample('30D').mean()   # monthly data 
#         well = well.rolling(30, min_periods=1).mean()
#         well = well.rolling(30, min_periods=1).mean()

        data = well['Oil [bbl/d]'] / 6.28981
        well.insert(4, 'Oil [m3/d]', data)
        time_data = np.arange(len(well))
        well.insert(0, 'Timestep', time_data) 

        wells_life.append(len(well))
        wells_data_.append(well)
    min_well_length = np.min(np.array(wells_life))
    if min_well_length < timesteps:
        timesteps_validation = min_well_length
    else:         
        timesteps_validation = timesteps
    
    for well, file, filename in zip (wells_data_, files_val, filenames_val):
        well = well.iloc[:timesteps_validation]   # daily data 
#         well = well.fillna(0)       
#         well = well.fillna(well.rolling(30, min_periods=1).median())
#         well = well.fillna(well.median())
        Raw_Data_preprocessed_val.append(well)

#     stats_validation = CSVLocalDataSet(filepath=parameters["path_val_stats"]+"/static_P50_data_validation.csv")
    stats_validation = CSVLocalDataSet(filepath=parameters["path_val_stats_matlab"]+"/static_P50_data_validation.csv")    
    
    stats_val = stats_validation.load()
    stats_val_ROIP = stats_val.loc[:, 'ROIP']
    stats_val = stats_val.loc[:, 'Effective_Length':'BottomWater_Oil_Saturation']
    
# #     using only rich geoostats and no bottom water properties
#     stats_val = stats_val.loc[:, 'Effective_Length':'Rich_Oil_Saturation']
    
# #     Using "Effective_Rich_Pay_Thickness" to account for standoff and rich thickness
#     data = stats_val['Rich_Pay_Thickness'] - stats_val['Stand_Off']
#     stats_val.insert(3, 'Effective_Rich_Pay_Thickness', data)
#     stats_val = stats_val.drop(columns = ['Rich_Pay_Thickness', 'Stand_Off'])
   
    property_names_val = list(stats_val.columns)
    properties_val = list(stats_val.values)
        
#     properties_val = stats.loc[:, ['Effective_Length', 'Spacing', 'Effective_Rich_Pay_Thickness', 'Non_Rich_Pay_Thickness', 
#                               'Rich_Vertical_Permeability','Non_Rich_Vertical_Permeability', 'Rich_Porosity', 
#                                       'Non_Rich_Porosity', 'Rich_Oil_Saturation', 'Non_Rich_Oil_Saturation']].values
    properties_val = np.array(properties_val)
    dummy11 = files_val
    return [dummy11, timesteps_validation, Raw_Data_preprocessed_val, files_val, filenames_val, properties_val, stats_val_ROIP, property_names_val] 


def dynamic_time_warping_validation(dummy11, Raw_Data_preprocessed_val, parameters):
    reference_well = CSVLocalDataSet(filepath=parameters["path_raw"]+"/B03-1P.csv")
    well_ref = reference_well.load()   
    data = well_ref['Oil [bbl/d]'] / 6.28981
    well_ref.insert(4, 'Oil [m3/d]', data)
    well_ref_oil_data = well_ref['Oil [m3/d]'].values
    
    Raw_Data_preprocessed_val_ = []
    distance_array_val = []
    for well_data in Raw_Data_preprocessed_val:
        well_oil_data = well_data['Oil [m3/d]'].values
        
        distance, path = fastdtw(well_ref_oil_data, well_oil_data, dist=euclidean)
        distance_array_val.append(distance)
        path = np.array(path)
        index_well = path[...,1]
        index_ref_well = path[...,0]
        well = well_data.iloc[index_well]
        well.insert(0, 'index_ref', index_ref_well)
        well = well.groupby('index_ref').mean()
#         well = well.reset_index(drop=True) 
        Raw_Data_preprocessed_val_.append(well)
    
    distance_array_val = np.array(distance_array_val)
    return [dummy12, distance_array_val, Raw_Data_preprocessed_val_]
 
    
    
def save_well_validation_data(dummy11, Raw_Data_preprocessed_val, parameters, files_val):    
# def save_well_validation_data(dummy12, Raw_Data_preprocessed_val_, parameters, files_val):
    all_wells_dates_input = []
    all_wells_steam_input_val = []
    all_wells_emulsion_input_val = []    
    all_wells_labels_val = [] 
    for well_data, file in zip(Raw_Data_preprocessed_val, files_val):
#     for well_data, file in zip(Raw_Data_preprocessed_val_, files_val):
        steam_input_data, emulsion_input_data, labels =  split(well_data)
        data_set_steam_input = CSVLocalDataSet(filepath=parameters["path_val_pre_processed"]+"/vali_steam_inputs_"+file)
        data_set_steam_input.save(steam_input_data)
        all_wells_dates_input.append(steam_input_data['Date'].values)
        steam_input_data = steam_input_data.drop(columns = 'Date')
        all_wells_steam_input_val.append(steam_input_data.values)
        data_set_emulsion_input = CSVLocalDataSet(filepath=parameters["path_val_pre_processed"]+"/vali_emulsion_inputs_"+file)
        data_set_emulsion_input.save(emulsion_input_data)
        all_wells_emulsion_input_val.append(emulsion_input_data.values)
        data_set_labels = CSVLocalDataSet(filepath=parameters["path_val_pre_processed"]+"/validation_labels_"+file)
        data_set_labels.save(labels)
        all_wells_labels_val.append(labels.values)   
 
    steam_input_column = steam_input_data.columns
    emulsion_input_column = emulsion_input_data.columns    
    labels_column = list(labels.columns)
    all_wells_dates_input = np.array(all_wells_dates_input)
    all_wells_steam_input_val = np.array(all_wells_steam_input_val)
    all_wells_emulsion_input_val = np.array(all_wells_emulsion_input_val)
    
    dummy13 = files_val
    return [dummy13, steam_input_column, emulsion_input_column, labels_column, all_wells_dates_input, all_wells_steam_input_val, all_wells_emulsion_input_val, all_wells_labels_val] 
  
    
    
def targets_computation_val(dummy13, timesteps_validation, all_wells_labels_val):
    well_count = np.arange(len(all_wells_labels_val))   
    time_index = np.arange(len(all_wells_labels_val[0])) 
    number_of_wells = len(well_count)
    
    # targets
    all_wells_steam_data = []
    all_wells_emulsion_data_val = []
    for ID in well_count:        
        well_steam_data = all_wells_labels_val[ID][:,0]   
        well_emulsion_data = all_wells_labels_val[ID][:,1]  
        all_wells_steam_data  = all_wells_steam_data + list(well_steam_data)
        all_wells_emulsion_data_val  = all_wells_emulsion_data_val + list(well_emulsion_data)
    all_wells_steam_data = np.array(all_wells_steam_data)
    all_wells_emulsion_data_val = np.array(all_wells_emulsion_data_val)
    wells_steam_rate_actual = all_wells_steam_data.reshape((number_of_wells, timesteps_validation)).T
    wells_emulsion_rate_actual = all_wells_emulsion_data_val.reshape((number_of_wells, timesteps_validation)).T
    
    dummy14 = well_count
    return [dummy14, well_count, number_of_wells, time_index, wells_steam_rate_actual, wells_emulsion_rate_actual] 



def predict_steam_then_oil(dummy14, well_count, number_of_wells, time_index, timesteps_validation, all_wells_steam_input_val, all_wells_emulsion_input_val, properties_val, parameters, algorithm):
    
    input_data = [] 
    for well_predictors_steam, property_ in zip(all_wells_steam_input_val, properties_val):
         for time_lapse in time_index:
            well_inputs = list(well_predictors_steam[time_lapse]) + list(property_) 
            input_data.append(well_inputs)
    input_data = np.array(input_data) 
    if  algorithm == 1:
        data_set_regressor_1 = PickleLocalDataSet(filepath=parameters["path_models"]+"/regressor_1.pickle")
        regressor_1 = data_set_regressor_1.load()
        wells_steam_rate_predicted = regressor_1.predict(input_data)
    else: 
#         print(algorithm)
#         # standardization
#         dataset_scaler_input_1 = PickleLocalDataSet(filepath=parameters["path_models"]+"/scaler_input_1.pickle")
#         scaler_input_1 = dataset_scaler_input_1.load()
#         input_data = scaler_input_1.transform(input_data)

        from tensorflow import keras
        model_1 = keras.models.load_model(parameters["path_models"]+"/network_model_1.h5")
        wells_steam_rate_predicted = model_1.predict(input_data)
        
    wells_steam_rate_predicted = wells_steam_rate_predicted.reshape((number_of_wells, timesteps_validation)).T
   
    # prediction inputs to model 2
    input_data_model_2 = []
    for well_predictors_emulsion, property_, well in zip(all_wells_emulsion_input_val, properties_val, well_count):
        for time_lapse in time_index:
            well_inputs_model_2 = [wells_steam_rate_predicted[time_lapse, well]] + list(well_predictors_emulsion[time_lapse]) + list(property_)
#             well_inputs_model_2 = [wells_steam_rate_predicted[time_lapse, well]] + list(well_predictors_emulsion[time_lapse])
            input_data_model_2.append(well_inputs_model_2)
    input_data_model_2 = np.array(input_data_model_2)
    if  algorithm == 1:
        data_set_regressor_2 = PickleLocalDataSet(filepath=parameters["path_models"]+"/regressor_2.pickle")
        regressor_2 = data_set_regressor_2.load()
        wells_emulsion_rate_predicted = regressor_2.predict(input_data_model_2)
    else:
#         # standardization
#         dataset_scaler_input_2 = PickleLocalDataSet(filepath=parameters["path_models"]+"/scaler_input_2.pickle")
#         scaler_input_2 = dataset_scaler_input_2.load()
#         input_data_model_2 = scaler_input_2.transform(input_data_model_2)

        model_2 = keras.models.load_model(parameters["path_models"]+"/network_model_2.h5")
        wells_emulsion_rate_predicted = model_2.predict(input_data_model_2)
    
    scheme = 1
    wells_emulsion_rate_predicted = wells_emulsion_rate_predicted.reshape((number_of_wells, timesteps_validation)).T  
    
    dummy15 = scheme
    return [dummy15, wells_steam_rate_predicted, wells_emulsion_rate_predicted, scheme] 

    
    
def predict_oil_then_steam(dummy14, well_count, number_of_wells, time_index, timesteps_validation, all_wells_emulsion_input_val, all_wells_steam_input_val, properties_val, parameters, algorithm):
    
    input_data = [] 
    for well_predictors_emulsion, property_ in zip(all_wells_emulsion_input_val, properties_val):
         for time_lapse in time_index:
            well_inputs = list(well_predictors_emulsion[time_lapse]) + list(property_) 
            input_data.append(well_inputs)
    input_data = np.array(input_data)
    if  algorithm == 1:
        data_set_regressor_1 = PickleLocalDataSet(filepath=parameters["path_models"]+"/regressor_1.pickle")
        regressor_1 = data_set_regressor_1.load()
        wells_emulsion_rate_predicted = regressor_1.predict(input_data)
    else:
#         # standardization
#         dataset_scaler_input_1 = PickleLocalDataSet(filepath=parameters["path_models"]+"/scaler_input_1.pickle")
#         scaler_input_1 = dataset_scaler_input_1.load()
#         input_data_model_1 = scaler_input_1.transform(input_data_model_1)
    
        from tensorflow import keras
        model_1 = keras.models.load_model(parameters["path_models"]+"/network_model_1.h5")
        wells_emulsion_rate_predicted = model_1.predict(input_data)

    wells_emulsion_rate_predicted = wells_emulsion_rate_predicted.reshape((number_of_wells, timesteps_validation)).T  
   
    # prediction inputs to model 2
    input_data_model_2 = []
    for well_predictors_steam, property_, well in zip(all_wells_steam_input_val, properties_val, well_count):
        for time_lapse in time_index:
            well_inputs_model_2 = [wells_emulsion_rate_predicted[time_lapse, well]] + list(well_predictors_steam[time_lapse]) + list(property_)
#             well_inputs_model_2 = [wells_emulsion_rate_predicted[time_lapse, well]] + list(well_predictors_steam[time_lapse])
            input_data_model_2.append(well_inputs_model_2)
    input_data_model_2 = np.array(input_data_model_2)
    if  algorithm == 1:
        data_set_regressor_2 = PickleLocalDataSet(filepath=parameters["path_models"]+"/regressor_2.pickle")
        regressor_2 = data_set_regressor_2.load()
        wells_steam_rate_predicted = regressor_2.predict(input_data_model_2)
    else:
#         # standardization
#         dataset_scaler_input_2 = PickleLocalDataSet(filepath=parameters["path_models"]+"/scaler_input_2.pickle")
#         scaler_input_1 = dataset_scaler_input_2.load()
#         input_data_model_1 = scaler_input_1.transform(input_data_model_2)
        
        model_2 = keras.models.load_model(parameters["path_models"]+"/network_model_2.h5")
        wells_steam_rate_predicted = model_2.predict(input_data_model_2)
    
    scheme = 2
    wells_steam_rate_predicted = wells_steam_rate_predicted.reshape((number_of_wells, timesteps_validation)).T
    
    dummy15 = scheme
    return [dummy15, wells_steam_rate_predicted, wells_emulsion_rate_predicted, scheme]  

  
    
def save_predicted_data(dummy15, well_count, all_wells_dates_input, wells_steam_rate_actual, wells_steam_rate_predicted, wells_emulsion_rate_actual, wells_emulsion_rate_predicted, steam_input_column, all_wells_steam_input_val, emulsion_input_column, all_wells_emulsion_input_val, labels_column, parameters, files_val, scheme):    # to input wells_RF_array for RF case
    
    print("Prediction Performance:\n")
    print("Steam Flow Rate:")
    for well, file in zip(well_count, files_val):
        dates = all_wells_dates_input[well].T
        steam_input = all_wells_steam_input_val[well].T
        steam_rate_actual = wells_steam_rate_actual[:,well]
        
        steam_rate_predicted = wells_steam_rate_predicted[:,well]
        emulsion_rate_predicted = wells_emulsion_rate_predicted[:,well]
        
        if scheme == 1:
            steam_rate_actual_predicted = pd.DataFrame(np.vstack((dates, steam_input, steam_rate_actual, steam_rate_predicted)).T, columns = ["Date"] + list(steam_input_column) + [labels_column[0]+" actual", labels_column[0]+" predicted"])
            data_set_steam_rate = CSVLocalDataSet(filepath=parameters["path_model_output_No_DWT_scheme1"]+"/steam_rate_"+file)
        elif scheme == 2:
            steam_rate_actual_predicted  = pd.DataFrame(np.vstack((dates, steam_input, emulsion_rate_predicted, steam_rate_actual, steam_rate_predicted)).T, columns = ["Date"] + list(steam_input_column) + [labels_column[1]+" predicted", labels_column[0]+" actual", labels_column[0]+" predicted"])
            data_set_steam_rate = CSVLocalDataSet(filepath=parameters["path_model_output_No_DWT_scheme2"]+"/steam_rate_"+file)
        elif scheme == 3:
            RF_input = wells_RF_array[well].T
            steam_rate_actual_predicted = pd.DataFrame(np.vstack((dates, RF_input, steam_input, steam_rate_actual, steam_rate_predicted)).T, columns = ["Date", "RF"] + list(steam_input_column) + [labels_column[0]+" actual", labels_column[0]+" predicted"])
            data_set_steam_rate = CSVLocalDataSet(filepath=parameters["path_model_output_No_DWT_RF_scheme1"]+"/steam_rate_"+file)
        else:
            RF_input = wells_RF_array[well].T
            steam_rate_actual_predicted  = pd.DataFrame(np.vstack((dates, RF_input, steam_input, emulsion_rate_predicted, steam_rate_actual, steam_rate_predicted)).T, columns = ["Date", "RF"] + list(steam_input_column) + [labels_column[1]+" predicted", labels_column[0]+" actual", labels_column[0]+" predicted"])
            data_set_steam_rate = CSVLocalDataSet(filepath=parameters["path_model_output_No_DWT_RF_scheme2"]+"/steam_rate_"+file)
        data_set_steam_rate.save(steam_rate_actual_predicted)
        print(file+" R_squared: {0:.4f}".format(r2_score(steam_rate_actual, steam_rate_predicted)))

        
    print("\n")
    print("Oil Rate:")
    for well, file in zip(well_count, files_val):
        dates = all_wells_dates_input[well].T
        emulsion_input = all_wells_emulsion_input_val[well].T
        emulsion_rate_actual = wells_emulsion_rate_actual[:,well]
        
        steam_rate_predicted = wells_steam_rate_predicted[:,well]
        emulsion_rate_predicted = wells_emulsion_rate_predicted[:,well]
        
        if scheme == 1:
            emulsion_rate_actual_predicted  = pd.DataFrame(np.vstack((dates, emulsion_input, steam_rate_predicted, emulsion_rate_actual, emulsion_rate_predicted)).T, columns = ["Date"] + list(emulsion_input_column) + [labels_column[0]+" predicted", labels_column[1]+" actual", labels_column[1]+" predicted"])
            data_set_emulsion_rate = CSVLocalDataSet(filepath=parameters["path_model_output_No_DWT_scheme1"]+"/emulsion_rate_"+file)
        elif scheme == 2:
            emulsion_rate_actual_predicted = pd.DataFrame(np.vstack((dates, emulsion_input, emulsion_rate_actual, emulsion_rate_predicted)).T, columns = ["Date"] + list(emulsion_input_column) + [labels_column[1]+" actual", labels_column[1]+" predicted"])
            data_set_emulsion_rate = CSVLocalDataSet(filepath=parameters["path_model_output_No_DWT_scheme2"]+"/emulsion_rate_"+file) 
        elif scheme == 3:
#             cum_input = wells_cum_oil_array[well].T
            RF_input = wells_RF_array[well].T
            emulsion_rate_actual_predicted  = pd.DataFrame(np.vstack((dates, RF_input, emulsion_input, steam_rate_predicted, emulsion_rate_actual, emulsion_rate_predicted)).T, columns = ["Date", "RF"] + list(emulsion_input_column) + [labels_column[0]+" predicted", labels_column[1]+" actual", labels_column[1]+" predicted"])
            data_set_emulsion_rate = CSVLocalDataSet(filepath=parameters["path_model_output_No_DWT_RF_scheme1"]+"/emulsion_rate_"+file)
        else:
            RF_input = wells_RF_array[well].T
            emulsion_rate_actual_predicted = pd.DataFrame(np.vstack((dates, RF_input, emulsion_input, emulsion_rate_actual, emulsion_rate_predicted)).T, columns = ["Date", "RF"] + list(emulsion_input_column) + [labels_column[1]+" actual", labels_column[1]+" predicted"])
            data_set_emulsion_rate = CSVLocalDataSet(filepath=parameters["path_model_output_No_DWT_RF_scheme2"]+"/emulsion_rate_"+file)   
        data_set_emulsion_rate.save(emulsion_rate_actual_predicted)
        print(file+" R_squared: {0:.4f}".format(r2_score(emulsion_rate_actual, emulsion_rate_predicted)))
    print("\n")
    dummy_validate = files_val
    return dummy_validate








#     os.chdir(parameters["path_val_stats"])
#     static_files_val = []
#     for static_file in glob.glob("*.csv"):
#         static_files_val.append(static_file)            
#     static_filenames_val = []
#     statics_data = []
#     for static_file in static_files_val:
#         static_filename, others = static_file.split('_')
#         static_filenames_val.append(static_filename) 
#     for static_file, static_filename in zip(static_files_val, static_filenames_val):
#         io = DataCatalog(
#             {
#                 static_filename: CSVLocalDataSet(filepath=parameters["path_val_stats"]+"/"+static_file),
#             }
#         )
#         static_data = io.load(static_filename)
             
# #         data = static_data['Rich_Pay_Thickness'] - static_data['Stand_Off']
# #         static_data.insert(9, 'Effective_Rich_Pay_Thickness', data)
        
#         statics_data.append(static_data)
#     statics_data_new = []
#     well_name_list = []
#     for pad_static in statics_data:  
#         well_name = pad_static['WELLPAIR_NAME'].values
#         well_name_list.append(well_name)
#         pad_static = pad_static.set_index('WELLPAIR_NAME')
#         pad_static = pad_static.drop(columns = ['PLAN_NAME','HIGH_PRESSURE']) 
#         statics_data_new.append(pad_static)
#     properties_val = []
#     probabilities = []
#     asset_names = []
#     for pad_static, names in zip(statics_data_new, well_name_list):
#         for well in names:
#             prob = pad_static.loc[well, 'Forecast_Prob']
#             probabilities.append(prob)
#             pad_code = pad_static.loc[well, 'PAD_CODE']
#             asset_name, pad = pad_code.split('_')
#             asset_names.append(asset_name)
#             property_ = pad_static.loc[well, 'Effective_Length':'BottomWater_Oil_Saturation'].values
# #             property_ = pad_static.loc[well, ['Effective_Length', 'Spacing', 'Effective_Rich_Pay_Thickness', 'Non_Rich_Pay_Thickness', 
# #                                       'Rich_Vertical_Permeability','Non_Rich_Vertical_Permeability', 'Rich_Porosity', 
# #                                               'Non_Rich_Porosity', 'Rich_Oil_Saturation', 'Non_Rich_Oil_Saturation']].values
#             properties_val.append(property_)
#     properties_val = np.array(properties)





