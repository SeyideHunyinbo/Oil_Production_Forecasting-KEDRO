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

     

def load_data(dummy, files: List, parameters: Dict):
    all_wells_steam_input = []
    all_wells_emulsion_input = []
    all_wells_labels = []
    
    for file in files:
        data_set_steam_input = CSVLocalDataSet(filepath=parameters["path_primary"]+"/steam_inputs_"+file)
        steam_input_data = data_set_steam_input.load()
        all_wells_steam_input.append(steam_input_data.values)
        data_set_emulsion_input = CSVLocalDataSet(filepath=parameters["path_primary"]+"/emulsion_inputs_"+file)
        emulsion_input_data = data_set_emulsion_input.load()
        all_wells_emulsion_input.append(emulsion_input_data.values)
        data_set_labels = CSVLocalDataSet(filepath=parameters["path_primary"]+"/labels_"+file)
        labels = data_set_labels.load()
        all_wells_labels.append(labels.values)
    
    steam_input_names = steam_input_data.columns
    emulsion_input_names = emulsion_input_data.columns
    all_wells_steam_input = np.array(all_wells_steam_input)
    all_wells_emulsion_input = np.array(all_wells_emulsion_input)
    return [all_wells_steam_input, all_wells_emulsion_input, all_wells_labels, steam_input_names, emulsion_input_names]


def augment_data_steam_oil(all_wells_steam_input, all_wells_emulsion_input, properties, all_wells_labels, parameters):
    time_index = np.arange(len(all_wells_labels[0]))
    input_data = []
    input_data_model_2 = []
    
#     #     Standardization (option 1): standardize before merging all data 
#     #     Standardize static data 
#     from sklearn.preprocessing import StandardScaler     
#     scaler_static = StandardScaler()
#     properties = scaler_static.fit_transform(properties)  
#     scaler_static = PickleLocalDataSet(filepath=parameters["path_models"]+"/scaler_static.pickle")
#     scaler_static.save(scaler_static)    
#     #     Standardize dynamic data inputs
#     all_wells_steam_input_ = []
#     all_wells_emulsion_input_ = []
#     scaler_input_steam_input = StandardScaler()
#     scaler_input_emulsion_input = StandardScaler()
    
#     scaler_input_steam_input.fit(all_wells_steam_input[0])
#     scaler_steam_input = PickleLocalDataSet(filepath=parameters["path_models"]+"/scaler_steam_input.pickle")
#     scaler_steam_input.save(scaler_input_steam_input)
#     scaler_input_emulsion_input.fit(all_wells_emulsion_input[0])
#     scaler_emulsion_input = PickleLocalDataSet(filepath=parameters["path_models"]+"/scaler_emulsion_input.pickle")
#     scaler_emulsion_input.save(scaler_input_emulsion_input)
#     for steam_input, emulsion_input  in zip(all_wells_steam_input, all_wells_emulsion_input):
#         std_steam_input = scaler_input_steam_input.transform(steam_input)
#         all_wells_steam_input_.append(std_steam_input)
#         std_emulsion_input = scaler_input_emulsion_input.transform(emulsion_input)
#         all_wells_emulsion_input_.append(std_emulsion_input)
    
#     all_wells_steam_input_ = np.array(all_wells_steam_input_)
#     all_wells_steam_input = all_wells_steam_input_
#     all_wells_emulsion_input_ = np.array(all_wells_emulsion_input_)
#     all_wells_emulsion_input = all_wells_emulsion_input_    
    
# #     input_columns = list(inputs.columns)   # saves all standardized inputs per well; TO EDIT!!!
# #     for well_input_, file in zip(all_wells_standardized_input, files):
# #         well_input_ = pd.DataFrame(well_input_, columns = input_columns)
# #         data_set = CSVLocalDataSet(filepath=parameters["path_features"]+"/std_input_"+file)
# #         data_set.save(well_input_)
    
    
    for well_predictors_steam, well_predictors_emulsion, property_, well_label in zip(all_wells_steam_input, all_wells_emulsion_input, properties, all_wells_labels):
#         time_index = np.arange(len(well_label))    # time index based on variable well length may change
        for time_lapse in time_index:          
            well_inputs = list(well_predictors_steam[time_lapse]) + list(property_)  
            well_inputs_model_2 = [well_label[time_lapse][0]] + list(well_predictors_emulsion[time_lapse]) + list(property_) 
            input_data.append(well_inputs)
            input_data_model_2.append(well_inputs_model_2)
    input_data = np.array(input_data)
    input_data_model_2 = np.array(input_data_model_2)
    scheme_train = 1
    
#      #     Standardization (option 2): standardize after merging all data 
#     from sklearn.preprocessing import StandardScaler 
#     scaler_input_1 = StandardScaler()                 
#     input_data = scaler_input_1.fit_transform(input_data)
#     dataset_scaler_input_1 = PickleLocalDataSet(filepath=parameters["path_models"]+"/scaler_input_1.pickle")
#     dataset_scaler_input_1.save(scaler_input_1)
    
#     scaler_input_2 = StandardScaler()                 
#     input_data_model_2 = scaler_input_2.fit_transform(input_data_model_2)
#     dataset_scaler_input_2 = PickleLocalDataSet(filepath=parameters["path_models"]+"/scaler_input_2.pickle")
#     dataset_scaler_input_2.save(scaler_input_2)
    
#     input_data = pd.DataFrame(input_data)           # saves all standardized inputs; TO EDIT!!!
#     data_set = CSVLocalDataSet(filepath=parameters["path_features"]+"/std_input_"+file)
#     data_set.save(input_data) 

    return [input_data, input_data_model_2, scheme_train]


def augment_data_oil_steam(all_wells_steam_input, all_wells_emulsion_input, properties, all_wells_labels, parameters):
    time_index = np.arange(len(all_wells_labels[0]))
    input_data = []
    input_data_model_2 = []
    
    for well_predictors_emulsion, well_predictors_steam, property_, well_label in zip(all_wells_emulsion_input, all_wells_steam_input, properties, all_wells_labels):
#         time_index = np.arange(len(well_label))
        for time_lapse in time_index:          
            well_inputs = list(well_predictors_emulsion[time_lapse]) + list(property_)  
            well_inputs_model_2 = [well_label[time_lapse][1]] + list(well_predictors_steam[time_lapse]) + list(property_) 
            input_data.append(well_inputs)
            input_data_model_2.append(well_inputs_model_2)
    input_data = np.array(input_data)
    input_data_model_2 = np.array(input_data_model_2)
    scheme_train = 2
    return [input_data, input_data_model_2, scheme_train]


def targets_computation(all_wells_labels):
    well_count = np.arange(len(all_wells_labels))
    
    all_wells_steam_data = []
    all_wells_emulsion_data = []
    for ID in well_count:
        well_steam_data = all_wells_labels[ID][:,0]
        well_emulsion_data = all_wells_labels[ID][:,1]
        all_wells_steam_data  = all_wells_steam_data + list(well_steam_data)
        all_wells_emulsion_data  = all_wells_emulsion_data + list(well_emulsion_data)
    all_wells_steam_data = np.array(all_wells_steam_data)
    all_wells_emulsion_data = np.array(all_wells_emulsion_data)
    
    return [all_wells_steam_data, all_wells_emulsion_data]


def train_validate_split_data(input_data, input_data_model_2, all_wells_steam_data, all_wells_emulsion_data, parameters: Dict, scheme_train):   
    from sklearn.model_selection import train_test_split  
    if scheme_train == 1:
        X_train, X_validate, y_train, y_validate = train_test_split(input_data, all_wells_steam_data, test_size=0.4,
                                                                    random_state=parameters["random_state"])
        X_train2, X_validate2, y_train2, y_validate2 = train_test_split(input_data_model_2, all_wells_emulsion_data,
                                                                        test_size=0.4,random_state=parameters["random_state"])
    else:
        X_train, X_validate, y_train, y_validate = train_test_split(input_data, all_wells_emulsion_data, test_size=0.4,
                                                                    random_state=parameters["random_state"])
        X_train2, X_validate2, y_train2, y_validate2 = train_test_split(input_data_model_2, all_wells_steam_data,
                                                                  test_size=0.4,random_state=parameters["random_state"])
    dummy1 = X_train2
    return [dummy1, X_train, X_validate, y_train, y_validate, X_train2, X_validate2, y_train2, y_validate2]


from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score 
def cross_validation(dummy1, X_validate, y_validate, X_validate2, y_validate2) -> np.ndarray: 
    gsc1 = GridSearchCV(estimator=RandomForestRegressor(), param_grid={'max_depth': range(3,9),
                                                                      'n_estimators': (10, 50, 100, 1000),},
                       cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)   
    gsc2 = GridSearchCV(estimator=RandomForestRegressor(), param_grid={'max_depth': range(3,9),
                                                                      'n_estimators': (10, 50, 100, 1000),},
                       cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)  
    grid_result1 = gsc1.fit(X_validate, y_validate)
    grid_result2 = gsc2.fit(X_validate2, y_validate2)
    best_params1 = grid_result1.best_params_
    best_params2 = grid_result2.best_params_
    regressor_1 = RandomForestRegressor(max_depth=best_params1["max_depth"], n_estimators=best_params1["n_estimators"],                               random_state=False, verbose=False)
    regressor_2 = RandomForestRegressor(max_depth=best_params2["max_depth"], n_estimators=best_params2["n_estimators"],                               random_state=False, verbose=False)
    
    print(regressor_1)
    print(regressor_2)
    print(best_params1)
    print(best_params2)
    
# Perform K-Fold CV
    scores1 = cross_val_score(regressor_1, X_validate, y_validate, cv=10, scoring='neg_mean_absolute_error')
    scores2 = cross_val_score(regressor_2, X_validate2, y_validate2, cv=10, scoring='neg_mean_absolute_error')
    
    print(scores1)
# [-94.89231431 -95.4661502  -92.4290891  -93.67966719 -96.05050244
#  -94.14108202 -95.53134942 -94.86169842 -93.14639362 -93.75411634]
    print(scores2)
# [-126.47647956 -125.72581613 -126.52363841 -124.50573334 -126.31735573
#  -128.21630544 -126.87387774 -126.36797491 -124.86543231 -128.12539376]

    dummy2 = scores1
    return [dummy2, scores1, scores2]


def train_test_split_data(input_data, input_data_model_2, all_wells_steam_data, all_wells_emulsion_data, parameters, scheme_train):   
    from sklearn.model_selection import train_test_split 
    if scheme_train == 1:
        X_train, X_test, y_train, y_test = train_test_split(input_data, all_wells_steam_data, test_size=parameters["test_size"],
                                                            random_state=parameters["random_state"])
        X_train2, X_test2, y_train2, y_test2 = train_test_split(input_data_model_2, all_wells_emulsion_data,
                                                                test_size=parameters["test_size"], random_state=parameters["random_state"])
    else:
        X_train, X_test, y_train, y_test = train_test_split(input_data, all_wells_emulsion_data, test_size=parameters["test_size"],
                                                            random_state=parameters["random_state"])
        X_train2, X_test2, y_train2, y_test2 = train_test_split(input_data_model_2, all_wells_steam_data,
                                                                test_size=parameters["test_size"], random_state=parameters["random_state"])
    return [X_train, X_test, y_train, y_test, X_train2, X_test2, y_train2, y_test2]

   
def random_forest_model_1(X_train, y_train, X_test, y_test, parameters, property_names, steam_input_names, emulsion_input_names, scheme_train):    
    regressor_1  = RandomForestRegressor(n_estimators=200, random_state=0)
    regressor_1.fit(X_train, y_train)
    if scheme_train == 1:
        input_1_names = list(steam_input_names) + list(property_names) 
    else:
        input_1_names = list(emulsion_input_names) + list(property_names) 
    feat_idx = np.argsort(regressor_1.feature_importances_)[::-1]
    input_1_names = np.array(input_1_names)[feat_idx]
    input_1_names = list(input_1_names)
    print("Feature importance:\n")
    for name, importance in zip(input_1_names, regressor_1.feature_importances_[feat_idx]):
        print(name, ": {0:.3f}".format(importance))
        
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    pd.Series(regressor_1.feature_importances_[feat_idx][::-1], index=input_1_names[::-1]).plot(kind='barh', ax=ax)
    ax.set_title('Features importance')
    fig.savefig(parameters["path_model_output_No_DWT"]+"/regressor_1_feature_importance.png")
    
    print("\n")
    '''
    print the mean squared error and accuracy of regression model
    '''
    '''
    training performance
    '''
    print("Train Result:")
    print("mean squared error: {0:.4f}".format(mean_squared_error(y_train, regressor_1.predict(X_train))))
    print("R_squared: {0:.4f}\n".format(r2_score(y_train, regressor_1.predict(X_train))))
    '''
    test performance
    '''
    print("Test Result:")        
    print("mean squared error: {0:.4f}".format(mean_squared_error(y_test, regressor_1.predict(X_test))))
    print("R_squared: {0:.4f}\n".format(r2_score(y_test, regressor_1.predict(X_test))))   
    data_set_regressor_1 = PickleLocalDataSet(filepath=parameters["path_models"]+"/regressor_1.pickle")
    data_set_regressor_1.save(regressor_1)
    
#     fig, axes = plt.subplots(1, 1,figsize = (12,12), dpi=800)
#     tree.plot_tree(regressor_1.estimators_[0], feature_names = input_1_names, filled = True);
#     fig.savefig(parameters["path_model_output_No_DWT"]+"/regressor_1_tree.png")
#     tree.export_graphviz(regressor_1.estimators_[0], out_file="path_model_output_No_DWT"]+"/regressor_1_tree.dot",
#                             feature_names = input_1_names, filled = True)
                             
    dummy1 = X_test
    return dummy1


def random_forest_model_2(dummy1, X_train2, y_train2, X_test2, y_test2, parameters, property_names, steam_input_names, emulsion_input_names, scheme_train):
    regressor_2  = RandomForestRegressor(n_estimators=200, random_state=0)
    regressor_2.fit(X_train2, y_train2)
    if scheme_train == 1:
        input_2_names = ['Steam [m3/d]'] + list(emulsion_input_names) + list(property_names)  
    else:
        input_2_names = ['Oil [m3/d]'] + list(steam_input_names) + list(property_names)
    feat_idx = np.argsort(regressor_2.feature_importances_)[::-1]
    input_2_names = np.array(input_2_names)[feat_idx]
    input_2_names = list(input_2_names)
    print("Feature importance:\n")
    for name, importance in zip(input_2_names, regressor_2.feature_importances_[feat_idx]):
        print(name, ": {0:.3f}".format(importance))
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    pd.Series(regressor_2.feature_importances_[feat_idx][::-1], index=input_2_names[::-1]).plot(kind='barh', ax=ax)
    ax.set_title('Features importance')
    fig.savefig(parameters["path_model_output_No_DWT"]+"/regressor_2_feature_importance.png")
    
    print("\n")
    '''
    print the mean squared error and accuracy of regression model
    '''
    '''
    training performance
    '''
    print("Train Result:")
    print("mean squared error: {0:.4f}".format(mean_squared_error(y_train2, regressor_2.predict(X_train2))))
    print("R_squared: {0:.4f}\n".format(r2_score(y_train2, regressor_2.predict(X_train2))))
    '''
    test performance
    '''
    print("Test Result:")        
    print("mean squared error: {0:.4f}".format(mean_squared_error(y_test2, regressor_2.predict(X_test2))))
    print("R_squared: {0:.4f}\n".format(r2_score(y_test2, regressor_2.predict(X_test2))))   
    data_set_regressor_2 = PickleLocalDataSet(filepath=parameters["path_models"]+"/regressor_2.pickle")
    data_set_regressor_2.save(regressor_2)
    
#     fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (12,12), dpi=800)
#     tree.plot_tree(regressor_2.estimators_[0], feature_names = input_2_names, filled = True);
#     fig.savefig(parameters["path_model_output_No_DWT"]+"/regressor_2_tree.png")
#     tree.export_graphviz(regressor_2.estimators_[0], out_file="path_model_output_No_DWT"]+"/regressor_2_tree.dot",
#                             feature_names = input_2_names, filled = True) 
                             
    algorithm = 1                         
    dummy2 = X_test2
    return [dummy2, algorithm]


def DL_Model(activation= 'linear', neurons= 5, optimizer='Adam'):   
    inputs = keras.Input(shape=(16,))
    hidden_layer_1 = layers.Dense(neurons, activation=activation)(inputs)
    hidden_layer_2 = layers.Dense(neurons, activation=activation)(hidden_layer_1)
    x = keras.layers.Dropout(0.3)(hidden_layer_2)
    outputs = layers.Dense(1, name='target_coefficients')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics=['accuracy'])
    return model


def cross_validation_ANN(X_train, y_train):
    from keras.wrappers.scikit_learn import KerasRegressor
    activation = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
    neurons = [35, 55, 70, 80]
    optimizer = ['SGD', 'Adam', 'Adamax']
    param_grid = dict(activation = activation, neurons = neurons, optimizer = optimizer)
    clf = KerasRegressor(build_fn= DL_Model, epochs= 80, batch_size=X_train.shape[0], verbose= 0)
    model_ = GridSearchCV(estimator= clf, param_grid=param_grid, n_jobs=-1)
    model_.fit(X_train, y_train)
    print("Max Accuracy Registered: {} using {}".format(round(model_.best_score_,3), model_.best_params_))


# import keras.backend as K
# import tensorflow
# from tensorflow import keras
# from tensorflow.keras import layers
def train_ANN_model_1(X_train, y_train, X_test, y_test, parameters):
    tensorflow.set_random_seed(parameters["random_state"])
    np.random.seed(parameters["random_state"])  
    inputs = keras.Input(shape=(X_train.shape[1],), name='input_coeffs')
    x = layers.Dense(parameters["hidden_layer_nodes_model_1"], activation='relu', name='hidden_layer_1')(inputs)
    x = layers.Dense(parameters["hidden_layer_nodes_model_1"], activation='relu', name='hidden_layer_2')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, name='target_coefficients')(x)

    model_1 = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_predictive_model')
    model_1.summary()
    adam = keras.optimizers.Adam(learning_rate=0.001)
    model_1.compile(optimizer = adam, loss = 'mean_squared_error', metrics=['accuracy'])
    print(K.eval(model_1.optimizer.lr))
    model_1.fit(X_train, y_train, batch_size=len(X_train), epochs=parameters["epochs_model_1"])
    print("\n") 
    print("Test Result:") 
    metrics = model_1.evaluate(X_test, y_test)
    logger = logging.getLogger(__name__)
    logger.info("MSE: %.2f.", metrics[0])    
    logger.info("accuracy: %.2f.", metrics[1]) 
    print("\n") 
    dummy1 = y_train                                           
    model_1.save(parameters["path_models"]+"/network_model_1.h5")    
    return dummy1
    
    #     model_1.fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
    
    

def train_ANN_model_2(dummy1, X_train2, y_train2, X_test2, y_test2, parameters):
    tensorflow.set_random_seed(parameters["random_state"])
    np.random.seed(parameters["random_state"])      
    inputs = keras.Input(shape=(X_train2.shape[1],), name='input_coeffs')
    x = layers.Dense(parameters["hidden_layer_nodes_model_2"], activation='relu', name='hidden_layer_1')(inputs)
    x = layers.Dense(parameters["hidden_layer_nodes_model_2"], activation='relu', name='hidden_layer_2')(x)
    x = keras.layers.Dropout(0.2)(x) 
    outputs = layers.Dense(1, name='target_coefficients')(x)

    model_2 = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_predictive_model')
    model_2.summary()
    adam = keras.optimizers.Adam(learning_rate=0.001)
    model_2.compile(optimizer = adam, loss = 'mean_squared_error', metrics=['accuracy'])
    print(K.eval(model_2.optimizer.lr))
    model_2.fit(X_train2, y_train2, batch_size=len(X_train2), epochs=parameters["epochs_model_2"])
    print("\n") 
    print("Test Result:") 
    metrics = model_2.evaluate(X_test2, y_test2)
    logger = logging.getLogger(__name__)
    logger.info("MSE: %.2f.", metrics[0])
    logger.info("accuracy: %.2f.", metrics[1])
    print("\n") 
    dummy2 = y_train2                                              
    model_2.save(parameters["path_models"]+"/network_model_2.h5")
    algorithm = 2 
    return [dummy2, algorithm]


                      
                      
                      
                      
                      
# def evaluate_model_1(dummy1, X_test: np.ndarray, y_test: np.ndarray, parameters: Dict):
#     model_1 = keras.models.load_model(parameters["path_models"]+"/network_model_1.h5")    
#     metrics = model_1.evaluate(X_test, y_test)
#     logger = logging.getLogger(__name__)
#     logger.info("MSE: %.2f.", metrics[0])    
#     logger.info("accuracy: %.2f.", metrics[1]) 
#     dummy2 = X_test                                           
#     return dummy2  
    
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