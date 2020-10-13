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


from sklearn.model_selection import train_test_split
def train_validate_split_data(all_wells_steam_input, all_wells_emulsion_input, properties, all_wells_labels): 
    well_count_input = np.arange(len(all_wells_labels))
    well_count_output = well_count_input
    train_wells, validation_wells, y_train_wells, y_validation_wells = train_test_split(well_count_input, well_count_output, test_size=0.5, random_state=42)
    
    time_index = np.arange(len(all_wells_labels[0]))    
    X_validate = []
    X_validate2 = []
#     for well in validation_wells:
    for well in well_count_input:
        well_predictors_steam = all_wells_steam_input[well]
        well_predictors_emulsion = all_wells_emulsion_input[well]
        property_ = properties[well]
        well_label = all_wells_labels[well]
        for time_lapse in time_index:          
            well_inputs = list(well_predictors_steam[time_lapse]) + list(property_)  
#             well_inputs_model_2 = [well_label[time_lapse][0]] + list(well_predictors_emulsion[time_lapse]) + list(property_) 
            well_inputs_model_2 = [well_label[time_lapse][0]] + list(well_predictors_emulsion[time_lapse])
            X_validate.append(well_inputs)
            X_validate2.append(well_inputs_model_2)
    X_validate = np.array(X_validate)
    X_validate2 = np.array(X_validate2)
    
    y_validate = []
    y_validate2 = []
    all_wells_steam_data = []
    all_wells_emulsion_data = []
#     for well in validation_wells:
    for well in well_count_input:
        well_steam_data = all_wells_labels[well][:,0]
        all_wells_steam_data  = all_wells_steam_data + list(well_steam_data)
        well_emulsion_data = all_wells_labels[well][:,1]
        all_wells_emulsion_data  = all_wells_emulsion_data + list(well_emulsion_data)
    y_validate = np.array(all_wells_steam_data)
    y_validate2 = np.array(all_wells_emulsion_data)
    
    return [X_validate, y_validate, X_validate2, y_validate2]


from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score 
def grid_search_tuning(X_validate, y_validate, X_validate2, y_validate2) -> np.ndarray: 
#     gsc1 = GridSearchCV(estimator=RandomForestRegressor(), param_grid={'max_depth': range(4,10),
#                                                                       'n_estimators': (100, 200, 500),},
#                        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)   
    gsc1 = GridSearchCV(estimator=RandomForestRegressor(), param_grid={'max_depth': range(4,10),
                                                                      'n_estimators': (100, 200, 500),},
#                                                                       'min_samples_split': (10, 20, 30),
#                                                                       'min_samples_leaf': (10, 20, 30),},
                       cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)  
    grid_result1 = gsc.fit(X_validate, y_validate)
    grid_result2 = gsc.fit(X_validate2, y_validate2)
    best_params1 = grid_result1.best_params_
    best_params2 = grid_result2.best_params_
    regressor_1 = RandomForestRegressor(max_depth=best_params1["max_depth"], n_estimators=best_params1["n_estimators"],
                                        random_state=False, verbose=False)
    regressor_2 = RandomForestRegressor(max_depth=best_params2["max_depth"], n_estimators=best_params2["n_estimators"],
                                        random_state=False, verbose=False)
    
    print(regressor_1)
    print(regressor_2)
    print(best_params1)
    print(best_params2)
    
# Perform K-Fold CV
    scores1 = cross_val_score(regressor_1, X_validate, y_validate, cv=10, scoring='neg_mean_absolute_error')
    scores2 = cross_val_score(regressor_2, X_validate2, y_validate2, cv=10, scoring='neg_mean_absolute_error')
    
    print(scores1)
    print(scores2)

    dummy2 = scores1
    return dummy2


# def objective_1(space):
#     regressor_1 = RandomForestRegressor(criterion = space['criterion'], 
#                                    max_depth = space['max_depth'],
#                                  max_features = space['max_features'],
#                                  min_samples_leaf = space['min_samples_leaf'],
#                                  min_samples_split = space['min_samples_split'],
#                                  n_estimators = space['n_estimators'], 
#                                  )
#     scores1 = cross_val_score(regressor_1, X_validate, y_validate, cv = 4).mean()
#     # We aim to maximize accuracy, therefore we return it as a negative value
#     return {'loss': -scores1, 'status': STATUS_OK }
        
        
# from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
# def bayesian_optimization_tuning(X_validate, y_validate, X_validate2, y_validate2) -> np.ndarray: 
#     space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
#             'max_depth': hp.quniform('max_depth', 10, 1200, 10),
#             'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
#             'min_samples_leaf': hp.uniform ('min_samples_leaf', 0, 0.5),
#             'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
#             'n_estimators' : hp.choice('n_estimators', [10, 50, 300, 750, 1200])
#         }

#     trials = Trials()
#     best = fmin(fn= objective_1,
#                 space= space,
#                 algo= tpe.suggest,
#                 max_evals = 80,
#                 trials= trials)
#     print(best)
#     dummy2 = scores1
#     return dummy2



def train_test_split_data(all_wells_labels, parameters):  
    well_count_input = np.arange(len(all_wells_labels))
    well_count_output = well_count_input
    
    train_wells, test_wells, y_train_wells, y_test_wells = train_test_split(well_count_input, well_count_output, test_size=0.1, random_state=parameters["random_state"])
    print(len(train_wells))
    print(len(test_wells))
    return [train_wells, test_wells]


def augment_data_steam_oil(train_wells, test_wells, all_wells_steam_input, all_wells_emulsion_input, properties, all_wells_labels):
    time_index = np.arange(len(all_wells_labels[0]))
    X_train = []
    X_train2 = []
    for well in train_wells:
        well_predictors_steam = all_wells_steam_input[well]
        well_predictors_emulsion = all_wells_emulsion_input[well]
        property_ = properties[well]
        well_label = all_wells_labels[well]

        for time_lapse in time_index:          
            well_inputs = list(well_predictors_steam[time_lapse]) + list(property_)  
            well_inputs_model_2 = [well_label[time_lapse][0]] + list(well_predictors_emulsion[time_lapse]) + list(property_) 
#             well_inputs_model_2 = [well_label[time_lapse][0]] + list(well_predictors_emulsion[time_lapse])
            X_train.append(well_inputs)
            X_train2.append(well_inputs_model_2)
    
    X_test = []
    X_test2 = []
    for well in test_wells:
        well_predictors_steam = all_wells_steam_input[well]
        well_predictors_emulsion = all_wells_emulsion_input[well]
        property_ = properties[well]
        well_label = all_wells_labels[well]

        for time_lapse in time_index:          
            well_inputs = list(well_predictors_steam[time_lapse]) + list(property_)
            well_inputs_model_2 = [well_label[time_lapse][0]] + list(well_predictors_emulsion[time_lapse]) + list(property_) 
#             well_inputs_model_2 = [well_label[time_lapse][0]] + list(well_predictors_emulsion[time_lapse])
            X_test.append(well_inputs)
            X_test2.append(well_inputs_model_2)
    
    X_train = np.array(X_train)
    X_train2 = np.array(X_train2)
    X_test = np.array(X_test)
    X_test2 = np.array(X_test2)
    scheme_train = 1
    return [X_train, X_test, X_train2,  X_test2, scheme_train]



def augment_data_oil_steam(train_wells, test_wells, all_wells_steam_input, all_wells_emulsion_input, properties, all_wells_labels):
    time_index = np.arange(len(all_wells_labels[0]))
    X_train = []
    X_train2 = []
    for well in train_wells:
        well_predictors_emulsion = all_wells_emulsion_input[well]
        well_predictors_steam = all_wells_steam_input[well]
        property_ = properties[well]
        well_label = all_wells_labels[well]

        for time_lapse in time_index:          
            well_inputs = list(well_predictors_emulsion[time_lapse]) + list(property_)  
            well_inputs_model_2 = [well_label[time_lapse][1]] + list(well_predictors_steam[time_lapse]) + list(property_) 
#             well_inputs_model_2 = [well_label[time_lapse][1]] + list(well_predictors_steam[time_lapse])
            X_train.append(well_inputs)
            X_train2.append(well_inputs_model_2)
    
    X_test = []
    X_test2 = []
    for well in test_wells:
        well_predictors_emulsion = all_wells_emulsion_input[well]
        well_predictors_steam = all_wells_steam_input[well]
        property_ = properties[well]
        well_label = all_wells_labels[well]

        for time_lapse in time_index:          
            well_inputs = list(well_predictors_emulsion[time_lapse]) + list(property_)  
            well_inputs_model_2 = [well_label[time_lapse][1]] + list(well_predictors_steam[time_lapse]) + list(property_) 
#             well_inputs_model_2 = [well_label[time_lapse][1]] + list(well_predictors_steam[time_lapse])
            X_test.append(well_inputs)
            X_test2.append(well_inputs_model_2)
    
    X_train = np.array(X_train)
    X_train2 = np.array(X_train2)
    X_test = np.array(X_test)
    X_test2 = np.array(X_test2)
    scheme_train = 2
    return [X_train, X_test, X_train2,  X_test2, scheme_train]


def targets_computation(train_wells, test_wells, all_wells_labels, scheme_train):
    well_count = np.arange(len(all_wells_labels))
    
    y_train = []
    y_train2 = []
    all_wells_steam_data = []
    all_wells_emulsion_data = []
    for well in train_wells:
        well_steam_data = all_wells_labels[well][:,0]
        all_wells_steam_data  = all_wells_steam_data + list(well_steam_data)
        well_emulsion_data = all_wells_labels[well][:,1]
        all_wells_emulsion_data  = all_wells_emulsion_data + list(well_emulsion_data)
    if scheme_train == 1:
        y_train = np.array(all_wells_steam_data)
        y_train2 = np.array(all_wells_emulsion_data)
    else:
        y_train = np.array(all_wells_emulsion_data)
        y_train2 = np.array(all_wells_steam_data)   
        
    y_test = []
    y_test2 = []
    all_wells_steam_data = []
    all_wells_emulsion_data = []
    for well in test_wells:
        well_steam_data = all_wells_labels[well][:,0]
        all_wells_steam_data  = all_wells_steam_data + list(well_steam_data)
        well_emulsion_data = all_wells_labels[well][:,1]
        all_wells_emulsion_data  = all_wells_emulsion_data + list(well_emulsion_data)
    if scheme_train == 1:
        y_test = np.array(all_wells_steam_data)
        y_test2 = np.array(all_wells_emulsion_data)
    else:
        y_test = np.array(all_wells_emulsion_data)
        y_test2 = np.array(all_wells_steam_data)
    
    return [y_train, y_test, y_train2, y_test2]



from sklearn import metrics, datasets, ensemble
def print_decision_rules(rf):
    for tree_idx, est in enumerate(rf.estimators_):
        tree = est.tree_
        assert tree.value.shape[1] == 1                        # no support for multi-output
        print('TREE: {}'.format(tree_idx))
        iterator = enumerate(zip(tree.children_left, tree.children_right, tree.feature, tree.threshold, tree.value))
        for node_idx, data in iterator:
            left, right, feature, th, value = data
            '''
            left: index of left child (if any)
            right: index of right child (if any)
            feature: index of the feature to check
            th: the threshold to compare against
            value: values associated with classes            

            for classifier, value is 0 except the index of the class to return
            '''
            class_idx = np.argmax(value[0])
            if left == -1 and right == -1:
                print('{} LEAF: return class={}'.format(node_idx, class_idx))
            else:
                print('{} NODE: if feature[{}] < {} then next={} else next={}'.format(node_idx, feature, th, left, right))
    return None



def random_forest_model_1(X_train, y_train, X_test, y_test, parameters, property_names, steam_input_names, emulsion_input_names, scheme_train):    
#     regressor_1  = RandomForestRegressor(n_estimators=250, min_samples_split=10, min_samples_leaf=10, max_depth=7)
    regressor_1  = RandomForestRegressor(max_depth=4, n_estimators=500,  random_state=0)
    
#     AdaBoost and XGBoost Regressors
#     regressor_1  = AdaBoostRegressor(n_estimators = 500, random_state=0)
#     regressor_1 = XGBRegressor(objective='reg:squarederror')
    
    regressor_1.fit(X_train, y_train)
#     print_decision_rules(regressor_1)
    
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
    pd.Series(regressor_1.feature_importances_[feat_idx][::-1], index=input_1_names[::-1]).plot('barh', ax=ax)
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
    print("R_squared: {0:.4f}\n".format(regressor_1.score(X_train, y_train)))
    '''
    test performance
    '''
    print("Test Result:")        
    print("mean squared error: {0:.4f}".format(mean_squared_error(y_test, regressor_1.predict(X_test))))
    print("R_squared: {0:.4f}\n".format(regressor_1.score(X_test, y_test)))   
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
#     regressor_2  = RandomForestRegressor(n_estimators=250, min_samples_split=30, min_samples_leaf=30, max_depth=7)
    regressor_2  = RandomForestRegressor(max_depth=5, n_estimators=500, random_state=0)
    
#     AdaBoost and XGBoost Regressors
#     regressor_2  = AdaBoostRegressor(n_estimators = 500, random_state=0)
#     regressor_2 = XGBRegressor(objective='reg:squarederror')

    regressor_2.fit(X_train2, y_train2)
#     print_decision_rules(regressor_2)
    if scheme_train == 1:
        input_2_names = ['Steam [m3/d]'] + list(emulsion_input_names) + list(property_names)  
    else:
        input_2_names = ['Oil [m3/d]'] + list(steam_input_names) + list(property_names)
    feat_idx = np.argsort(regressor_2.feature_importances_)[::-1]
    
#     import eli5
#     from eli5.sklearn import PermutationImportance
#     perm = PermutationImportance(regressor_2).fit(X_test2, y_test2)
#     feat_idx = np.argsort(eli5.show_weights(perm))[::-1]
    
    input_2_names = np.array(input_2_names)[feat_idx]
    input_2_names = list(input_2_names)
    print("Feature importance:\n")
    for name, importance in zip(input_2_names, regressor_2.feature_importances_[feat_idx]):
        print(name, ": {0:.3f}".format(importance))
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    pd.Series(regressor_2.feature_importances_[feat_idx][::-1], index=input_2_names[::-1]).plot('barh', ax=ax)
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
    print("R_squared: {0:.4f}\n".format(regressor_2.score(X_train2, y_train2)))
    '''
    test performance
    '''
    print("Test Result:")        
    print("mean squared error: {0:.4f}".format(mean_squared_error(y_test2, regressor_2.predict(X_test2))))
    print("R_squared: {0:.4f}\n".format(regressor_2.score(X_test2, y_test2)))   
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


import keras.backend as K
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

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