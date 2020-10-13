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

from kedro.pipeline import node, Pipeline
from analytics_new.pipelines.data_science_no_DWT.nodes import (
    load_data,
    augment_data_steam_oil,
    augment_data_oil_steam,
    targets_computation,
    train_validate_split_data,
    cross_validation,
    train_test_split_data,
    random_forest_model_1,
    random_forest_model_2,
    cross_validation_ANN,
    train_ANN_model_1,
    train_ANN_model_2,
#     train_model_CNN,
#     array_to_image
#     k_means_clustering
)

def create_pipeline(**kwargs):
    return Pipeline(
        [ 
            node(
                func=load_data,
                inputs=["dummy", "files", "parameters"],
                outputs=["all_wells_steam_input", "all_wells_emulsion_input", "all_wells_labels", "steam_input_names", "emulsion_input_names"],
                name="data_loading"
            ),
            node(
                func=augment_data_steam_oil,
                inputs=["all_wells_steam_input", "all_wells_emulsion_input", "properties", "all_wells_labels", "parameters"],
                outputs=["input_data", "input_data_model_2", "scheme_train"],
                name="augment_data_steam_oil"
            ),
#             node(
#                 func=augment_data_oil_steam,
#                 inputs=["all_wells_steam_input", "all_wells_emulsion_input", "properties", "all_wells_labels", "parameters"],
#                 outputs=["input_data", "input_data_model_2", "scheme_train"],
#                 name="augment_data_oil_steam"
#             ),
            node(
                func=targets_computation,
                inputs="all_wells_labels",
                outputs=["all_wells_steam_data", "all_wells_emulsion_data"],
                name="targets_computation"
            ),
#             node(
#                 func=train_validate_split_data,
#                 inputs=["input_data", "input_data_model_2", "all_wells_steam_data", "all_wells_emulsion_data", "parameters","scheme_train"],
#                 outputs=["dummy1", "X_train", "X_validate", "y_train", "y_validate", "X_train2", "X_validate2", "y_train2", "y_validate2"],
#                 name="train_validation_split"
#             ),
#             node(
#                 func=cross_validation,
#                 inputs=["dummy1", "X_validate", "y_validate", "X_validate2", "y_validate2", "parameters"],
#                 outputs=["dummy2", "scores1", "scores2"],
#                 name="cross_validation"
#             ),
            node(
                func=train_test_split_data,
                inputs=["input_data", "input_data_model_2", "all_wells_steam_data", "all_wells_emulsion_data", "parameters", "scheme_train"],
                outputs=["X_train", "X_test", "y_train", "y_test", "X_train2", "X_test2", "y_train2", "y_test2"],
                name="train_test_split"
            ),
            node(
                func=random_forest_model_1,
                inputs=["X_train", "y_train", "X_test", "y_test", "parameters", "property_names", "steam_input_names", "emulsion_input_names", "scheme_train"],
                outputs="dummy1",
                name="random_forest_modeling_1"
            ),
            node(
                func=random_forest_model_2,
                inputs=["dummy1", "X_train2", "y_train2", "X_test2", "y_test2", "parameters", "property_names", "steam_input_names", "emulsion_input_names", "scheme_train"],
                outputs=["dummy2", "algorithm"],
                name="random_forest_modeling_2"
            ),
#             node(
#                 func=cross_validation_ANN,
#                 inputs=["X_train", "y_train"],
#                 outputs=None,
#                 name="cross_validation_ANN"
#             ),
#             node(
#                 func=train_ANN_model_1,
#                 inputs=["X_train", "y_train", "X_test", "y_test", "parameters"],
#                 outputs="dummy1",
#                 name="artificial_neural_network_modeling"
#             ),
#             node(
#                 func=train_ANN_model_2,
#                 inputs=["dummy1", "X_train2", "y_train2", "X_test2", "y_test2", "parameters"],
#                 outputs=["dummy2", "algorithm"],
#                 name="artificial_neural_network_modeling_2"
#             ),
#             node(
#                 func=array_to_image,
#                 inputs=["dummy2", "X_train", "X_test", "parameters", "filenames"],
#                 outputs="dummy3",
#                 name="array_to_image"
#             ),
#             node(
#                 func=train_model_CNN,
#                 inputs=["dummy2", "X_train", "y_train", "X_test", "y_test", "parameters"],
#                 outputs="dummy3",
#                 name="convolutional_neural_network_modeling"
#             ),
#             node(
#                 func=k_means_clustering,
#                 inputs=["labels", "parameters"],
#                 outputs=["kmeans", "y_kmeans"],
#                 name="KMeans_Clustering"
#             ),
        ]
    )
