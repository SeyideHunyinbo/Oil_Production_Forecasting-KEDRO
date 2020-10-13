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
from analytics_new.pipelines.data_science.nodes import (
    standardisation,
    merge_dynamic_static_data,
    augment_data,
    train_test_split_data,
    random_forest_model_1,
    random_forest_model_2,
#     train_ANN_model_1,
#     evaluate_model_1,
#     train_ANN_model_2,
#     evaluate_model_2,
#     train_model_CNN,
#     array_to_image
#     k_means_clustering
)

def create_pipeline(**kwargs):
    return Pipeline(
        [ 
            node(
                func=standardisation,
                inputs=["dummy", "properties", "files", "parameters"],
                outputs=["all_wells_standardized_input_flattened", "all_wells_standardized_properties", "all_wells_labels"],
                name="standardisation"
            ),
            node(
                func=merge_dynamic_static_data,
                inputs=["all_wells_standardized_input_flattened", "all_wells_standardized_properties"],
                outputs="all_wells_coeffs_reservoir_data",
                name="merge_dynamic_static_data"
            ),
            node(
                func=augment_data,
                inputs=["all_wells_coeffs_reservoir_data", "all_wells_labels"],
                outputs=["input_data", "input_data_model_2", "all_wells_steam_data", "all_wells_emulsion_data"],
                name="data_augmentation"
            ),
            node(
                func=train_test_split_data,
                inputs=["input_data", "input_data_model_2", "all_wells_steam_data", "all_wells_emulsion_data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test", "X_train2", "X_test2", "y_train2", "y_test2"],
                name="train_test_split"
            ),
            node(
                func=random_forest_model_1,
                inputs=["X_train", "y_train", "X_test", "y_test", "parameters"],
                outputs="dummy1",
                name="random_forest_modeling_1"
            ),
            node(
                func=random_forest_model_2,
                inputs=["dummy1", "X_train2", "y_train2", "X_test2", "y_test2", "parameters"],
                outputs="dummy2",
                name="random_forest_modeling_2"
            ),
#             node(
#                 func=train_ANN_model_1,
#                 inputs=["dummy2", "X_train", "y_train", "parameters"],
#                 outputs="dummy3",
#                 name="artificial_neural_network_modeling"
#             ),
#             node(
#                 func=evaluate_model_1,
#                 inputs=["dummy3", "X_test", "y_test", "parameters"],
#                 outputs="dummy4",
#                 name="model_1_evaluation"
#             ),
#             node(
#                 func=train_ANN_model_2,
#                 inputs=["dummy4", "X_train2", "y_train2", "parameters"],
#                 outputs="dummy5",
#                 name="artificial_neural_network_modeling_2"
#             ),
#             node(
#                 func=evaluate_model_2,
#                 inputs=["dummy5", "X_test2", "y_test2", "parameters"],
#                 outputs="dummy6",
#                 name="model_2_evaluation"
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
