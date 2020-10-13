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

import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sc
import matplotlib.pyplot as plt
from kedro.io import (
    DataCatalog,
    CSVLocalDataSet,
    PickleLocalDataSet,
    HDFLocalDataSet
) 

        
def standardisation(dummy, properties: np.ndarray, files: List, parameters: Dict):
    from sklearn.preprocessing import StandardScaler
    all_wells_input = []
    all_wells_labels = []
    
    for file in files:
        data_set_input = CSVLocalDataSet(filepath=parameters["path_primary"]+"/input_DWT_coeffs_"+file)
        DWT_Aprox_coeff_input = data_set_input.load()
        all_wells_input.append(DWT_Aprox_coeff_input.values)
        data_set_labels = CSVLocalDataSet(filepath=parameters["path_primary"]+"/labels_"+file)
        labels = data_set_labels.load()
        all_wells_labels.append(labels.values)

#     Standardize dynamic data coeffs
    scaler_coeffs = StandardScaler()
    scaler_coeffs.fit(all_wells_input[0])           # fit based on first well record          
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
    
    data_set_scaler_coeffs = PickleLocalDataSet(filepath=parameters["path_models"]+"/scaler_coeffs.pickle")
    data_set_scaler_coeffs.save(scaler_coeffs)
    
    input_columns = list(DWT_Aprox_coeff_input.columns)
    for std_coeffs, file in zip(all_wells_standardized_input, files):
        std_coeffs = pd.DataFrame(std_coeffs, columns = input_columns)
        data_set = CSVLocalDataSet(filepath=parameters["path_features"]+"/std_DWT_input_coeffs_"+file)
        data_set.save(std_coeffs)
    
#     Standardize static data      
    scaler_static = StandardScaler()
    all_wells_standardized_properties = scaler_static.fit_transform(properties)  
#     print(all_wells_standardized_input[0])
#     print(all_wells_standardized_properties[0])

    data_set_scaler_static = PickleLocalDataSet(filepath=parameters["path_models"]+"/scaler_static.pickle")
    data_set_scaler_static.save(scaler_static)
    
    return [all_wells_standardized_input_flattened, all_wells_standardized_properties, all_wells_labels]


def merge_dynamic_static_data(all_wells_standardized_input_flattened: np.ndarray, all_wells_standardized_properties: np.ndarray):     
    all_wells_coeffs_reservoir_data = []
    for flattened_std_coeffs, standardized_properties in zip(all_wells_standardized_input_flattened, all_wells_standardized_properties):
        flattened_std_coeffs = list(flattened_std_coeffs)
        standardized_properties = list(standardized_properties)
        for reservoir_property in standardized_properties:
            flattened_std_coeffs.append(reservoir_property)      # append reservoir data to dynamic data coeffs
        all_wells_coeffs_reservoir_data.append(flattened_std_coeffs)
    all_wells_coeffs_reservoir_data = np.array(all_wells_coeffs_reservoir_data)   
    return all_wells_coeffs_reservoir_data


def augment_data(all_wells_coeffs_reservoir_data, all_wells_labels): 
    well_count = np.arange(len(all_wells_coeffs_reservoir_data))   
    daily_timesteps = np.arange(len(all_wells_labels[0]))

    input_data = []
    input_data_model_2 = []
    for coeff_inputs, well_label in zip(all_wells_coeffs_reservoir_data, all_wells_labels):
        for time_lapse in daily_timesteps: 
            well_inputs = [time_lapse] + list(coeff_inputs)            # append time lapse to input data
            well_inputs_model_2 = [well_label[time_lapse][0]] + well_inputs
            input_data.append(well_inputs)
            input_data_model_2.append(well_inputs_model_2)
    input_data = np.array(input_data)
    input_data_model_2 = np.array(input_data_model_2)

    all_wells_steam_data = []
    all_wells_emulsion_data = []
    for ID in well_count:        
        well_steam_data = all_wells_labels[ID][:,0]   
        well_emulsion_data = all_wells_labels[ID][:,1]  
        all_wells_steam_data  = all_wells_steam_data + list(well_steam_data)
        all_wells_emulsion_data  = all_wells_emulsion_data + list(well_emulsion_data)
    all_wells_steam_data = np.array(all_wells_steam_data)
    all_wells_emulsion_data = np.array(all_wells_emulsion_data)
    return [input_data, input_data_model_2, all_wells_steam_data, all_wells_emulsion_data]


def train_test_split_data(input_data, input_data_model_2, all_wells_steam_data, all_wells_emulsion_data, parameters: Dict) -> np.ndarray:   
    from sklearn.model_selection import train_test_split     
    X_train, X_test, y_train, y_test = train_test_split(input_data, all_wells_steam_data, test_size=parameters["test_size"],
                                                        random_state=parameters["random_state"])
    X_train2, X_test2, y_train2, y_test2 = train_test_split(input_data_model_2, all_wells_emulsion_data, test_size=parameters["test_size"],
                                                        random_state=parameters["random_state"])
    print(X_train.shape)
    print(X_test.shape)
    print(input_data.shape)
    return [X_train, X_test, y_train, y_test, X_train2, X_test2, y_train2, y_test2]


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score    
def random_forest_model_1(X_train, y_train, X_test, y_test, parameters:Dict):    
    regressor_1  = RandomForestRegressor(random_state=0)
    regressor_1.fit(X_train, y_train)
    '''
    print the mean squared error and accuracy of regression model
    '''
    '''
    training performance
    '''
    print("Train Result:\n")
    print("mean squared error: {}\n".format(mean_squared_error(y_train, regressor_1.predict(X_train))))
    print("R_squared: {0:.4f}\n".format(r2_score(y_train, regressor_1.predict(X_train))))
#     res = cross_val_score(regr, X_train, y_train, cv=10, scoring='accuracy')
#     print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
#     print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
    '''
    test performance
    '''
    print("Test Result:\n")        
    print("mean squared error: {0:.4f}\n".format(mean_squared_error(y_test, regressor_1.predict(X_test))))
    print("R_squared: {0:.4f}\n".format(r2_score(y_test, regressor_1.predict(X_test))))
    
    data_set_regressor_1 = PickleLocalDataSet(filepath=parameters["path_models"]+"/regressor_1.pickle")
    data_set_regressor_1.save(regressor_1)
    dummy1 = X_test
    return dummy1


def random_forest_model_2(dummy1, X_train2, y_train2, X_test2, y_test2, parameters:Dict):    
    regressor_2  = RandomForestRegressor(random_state=0)
    regressor_2.fit(X_train2, y_train2)
    '''
    print the mean squared error and accuracy of regression model
    '''
    '''
    training performance
    '''
    print("Train Result:\n")
    print("mean squared error: {}\n".format(mean_squared_error(y_train2, regressor_2.predict(X_train2))))
    print("R_squared: {0:.4f}\n".format(r2_score(y_train2, regressor_2.predict(X_train2))))
    '''
    test performance
    '''
    print("Test Result:\n")        
    print("mean squared error: {0:.4f}\n".format(mean_squared_error(y_test2, regressor_2.predict(X_test2))))
    print("R_squared: {0:.4f}\n".format(r2_score(y_test2, regressor_2.predict(X_test2))))   
    data_set_regressor_2 = PickleLocalDataSet(filepath=parameters["path_models"]+"/regressor_2.pickle")
    data_set_regressor_2.save(regressor_2)
    dummy2 = X_test2
    return dummy2 


import tensorflow
from tensorflow import keras
from tensorflow.keras import layers  
def train_ANN_model_1(dummy2, X_train: np.ndarray, y_train: np.ndarray, parameters: Dict):
    tensorflow.set_random_seed(0)
    np.random.seed(0)    
    
    inputs = keras.Input(shape=(X_train.shape[1],), name='input_coeffs')
    hidden_layer_1 = layers.Dense(parameters["hidden_layer_nodes_model_1"], activation='relu', name='hidden_layer_1')(inputs)
    hidden_layer_2 = layers.Dense(parameters["hidden_layer_nodes_model_1"], activation='relu', name='hidden_layer_2')(hidden_layer_1)
    hidden_layer_3 = layers.Dense(parameters["hidden_layer_nodes_model_1"], activation='relu', name='hidden_layer_3')(hidden_layer_2)
    
    outputs = layers.Dense(1, name='target_coefficients')(hidden_layer_3)

    model_1 = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_predictive_model')
    model_1.summary() 
    model_1.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
    model_1.fit(X_train, y_train, epochs=parameters["epochs_model_1"])
    
    dummy3 = y_train                                               # dummy variable set to enable sequential run
    model_1.save(parameters["path_models"]+"/network_model_1.h5")
    return dummy3


def evaluate_model_1(dummy3, X_test: np.ndarray, y_test: np.ndarray, parameters: Dict):
    model_1 = keras.models.load_model(parameters["path_models"]+"/network_model_1.h5")
    
    metrics = model_1.evaluate(X_test, y_test)
    logger = logging.getLogger(__name__)
    logger.info("MSE: %.2f.", metrics[0])    
    logger.info("accuracy: %.2f.", metrics[1]) 
    dummy4 = X_test                                            
    return dummy4
    
    
def train_ANN_model_2(dummy4, X_train2: np.ndarray, y_train2: np.ndarray, parameters: Dict):
    tensorflow.set_random_seed(0)
    np.random.seed(0)    
    
    inputs = keras.Input(shape=(X_train2.shape[1],), name='input_coeffs')
    hidden_layer_1 = layers.Dense(parameters["hidden_layer_nodes_model_2"], activation='relu', name='hidden_layer_1')(inputs)
    hidden_layer_2 = layers.Dense(parameters["hidden_layer_nodes_model_2"], activation='relu', name='hidden_layer_2')(hidden_layer_1)
    hidden_layer_3 = layers.Dense(parameters["hidden_layer_nodes_model_2"], activation='relu', name='hidden_layer_3')(hidden_layer_2)
    
    outputs = layers.Dense(1, name='target_coefficients')(hidden_layer_3)

    model_2 = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_predictive_model')
    model_2.summary() 
    model_2.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
    model_2.fit(X_train2, y_train2, epochs=parameters["epochs_model_2"])
    
    dummy5 = y_train2                                               
    model_2.save(parameters["path_models"]+"/network_model_2.h5")
    return dummy5


def evaluate_model_2(dummy5, X_test2: np.ndarray, y_test2: np.ndarray, parameters: Dict):
    model_2 = keras.models.load_model(parameters["path_models"]+"/network_model_2.h5")
    
    metrics = model_2.evaluate(X_test2, y_test2)
    logger = logging.getLogger(__name__)
    logger.info("MSE: %.2f.", metrics[0])    
    logger.info("accuracy: %.2f.", metrics[1]) 
    dummy6 = X_test2                                            
    return dummy6
    

def validate(dummy6, regressor_1, regressor_2, input_data, parameters: Dict):  
    well_1_inputs = input_data[:1399]
    well_2_inputs = input_data[1399:2798]
    well_13_inputs = input_data[16788:18187]
    well_51_inputs = input_data[69950:71349]  
    
    wells_inputs = list(well_1_inputs) + list(well_2_inputs) + list(well_13_inputs) + list(well_51_inputs)
    wells_steam_rate = regressor_1.predict(wells_inputs)
    
    well_1_steam_rate = wells_steam_rate[:1399]
    well_2_steam_rate = wells_steam_rate[1399:2798]
    well_13_steam_rate = wells_steam_rate[16788:18187]
    well_51_steam_rate = wells_steam_rate[69950:71349] 
    
    well_1_steam_rate_dataframe = pd.DataFrame(np.transpose(np.array(well_1_steam_rate)), columns = ["well 1 steam rate"])
    well_2_steam_rate_dataframe = pd.DataFrame(np.transpose(np.array(well_2_steam_rate)), columns = ["well 2 steam rate"])
    well_13_steam_rate_dataframe = pd.DataFrame(np.transpose(np.array(well_13_steam_rate)), columns = ["well 13 steam rate"])
    well_51_steam_rate_dataframe = pd.DataFrame(np.transpose(np.array(well_51_steam_rate)), columns = ["well 51 steam rate"])
    
    data_set = CSVLocalDataSet(filepath=parameters["path_model_output"]+"/well_1_steam_rate_RF.csv")
    data_set.save(well_1_steam_rate_dataframe)   
    
    data_set = CSVLocalDataSet(filepath=parameters["path_model_output"]+"/well_2_steam_rate_RF.csv")
    data_set.save(well_2_steam_rate_dataframe) 
    
    data_set = CSVLocalDataSet(filepath=parameters["path_model_output"]+"/well_13_steam_rate_RF.csv")
    data_set.save(well_13_steam_rate_dataframe) 
    
    data_set = CSVLocalDataSet(filepath=parameters["path_model_output"]+"/well_51_steam_rate_RF.csv")
    data_set.save(well_51_steam_rate_dataframe) 

    dummy7 = well_1_steam_rate 
    return dummy7
    
    
    
# Cross_validation

    
# change array to image  
# def array_to_image(dummy2, X_train: np.ndarray, X_test: np.ndarray, parameters: Dict, filenames: List):
#     for training_image, test_image, filename in  zip(X_train, X_test, filenames):      
#         plt.figure(figsize=(40,10))
#         plt.imshow(training_image)
#         plt.imsave(parameters["path_features"]+"/training_image_"+filename+".png", training_image)
#         plt.imshow(test_image)
#         plt.imsave(parameters["path_features"]+"/test_image_"+filename+".png", test_image)
#         dummy3 = X_train
#     return dummy3

    
# #CNN Node
# def train_model_CNN(dummy2, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, parameters: Dict):
#     tensorflow.set_random_seed(0)
#     np.random.seed(0)
    
#     y_training = []
#     y_testing = []
#     for y in  y_train:
#         y_training.append(y[:,1])
#     for y in  y_test:
#         y_testing.append(y[:,1])
#     y_training = np.array(y_training)
#     y_testing = np.array(y_testing)
    
#     X_training = X_train.reshape(len(X_train), X_train.shape[1], X_train.shape[2], 1)
#     X_testing = X_test.reshape(len(X_test), X_test.shape[1], X_test.shape[2], 1)
       
#     inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[2], 1), name='input_image') #input is greyscale "image"
#     filters = (16, 32)
#     for (i, filter_) in enumerate(filters):
#         if i == 0:
#             x = inputs
#         x = keras.layers.Conv2D(filter_, (3, 3), padding='same')(x)  #confirm what padding means
#         x = keras.layers.Activation('relu')(x)
#         x = keras.layers.BatchNormalization(axis=-1)(x)  #confirm what this means
#         x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

#     x = keras.layers.Flatten()(x)
#     x = layers.Dense(parameters["fully_connected_layer_1"], activation='relu', name='fully_connected_layer_1')(x)
#     x = keras.layers.BatchNormalization(axis=-1)(x)
#     x = keras.layers.Dropout(0.5)(x)                          #confirm what this means
#     x = layers.Dense(parameters["fully_connected_layer_2"], activation='relu', name='fully_connected_layer_2')(x)
#     x = layers.Dense(X_train.shape[1], activation='linear', name='outputs')(x)
    
#     model = keras.Model(inputs=inputs, outputs=x, name='CNN_regression_model')
#     model.summary() 
#     model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
#     model.fit(X_training, y_training, epochs=parameters["epochs_CNN"], batch_size=parameters["batch_size"])
#     test_loss = model.evaluate(X_testing, y_testing)
  
#     dummy3 = X_train                                               # dummy variable set to enable sequential run
#     model.save(parameters["path_models"]+"/CNN_network_model.h5")
#     return dummy3


# def k_means_clustering(labels: pd.DataFrame, parameters: Dict) -> List:
#     from sklearn.cluster import KMeans
    
#     kmeans = KMeans(n_clusters=parameters["n_clusters"])
#     data = labels.values
#     kmeans.fit(data)
#     y_kmeans = pd.DataFrame(kmeans.predict(data), columns = ["y_kMeans"])
#     return [kmeans, y_kmeans]