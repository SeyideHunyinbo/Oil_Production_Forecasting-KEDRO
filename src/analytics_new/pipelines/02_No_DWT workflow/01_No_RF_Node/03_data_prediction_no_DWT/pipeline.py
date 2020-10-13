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
from analytics_new.pipelines.data_prediction_no_DWT.nodes import (
    load_well_validation_data,
    dynamic_time_warping_validation,
    save_well_validation_data,
    targets_computation_val,
    predict_steam_then_oil,
    predict_oil_then_steam,
    save_predicted_data,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [ 
            node(
                func=load_well_validation_data,
#                 inputs="parameters",
                inputs=["dummy2", "timesteps", "parameters"],
                outputs=["dummy11", "timesteps_validation", "Raw_Data_preprocessed_val", "files_val", "filenames_val", "properties_val", "stats_val_ROIP", "property_names_val"],
                name="load_well_validation_data"
            ),
#             node(
#                 func=dynamic_time_warping_validation,
#                 inputs=["dummy11", "Raw_Data_preprocessed_val", "parameters"],
#                 outputs=["dummy12", "distance_array_val", "Raw_Data_preprocessed_val_"],
#                 name="dynamic_time_warping_validation"
#             ),
            node(
                func=save_well_validation_data,
                inputs=["dummy11", "Raw_Data_preprocessed_val", "parameters", "files_val"],
#                 inputs=["dummy12", "Raw_Data_preprocessed_val_", "parameters", "files_val"],
                outputs=["dummy13", "steam_input_column", "emulsion_input_column", "labels_column", "all_wells_dates_input", "all_wells_steam_input_val", "all_wells_emulsion_input_val", "all_wells_labels_val"],
                name="save_well_validation_data"
            ),
            node(
                func=targets_computation_val,
                inputs=["dummy13", "timesteps_validation", "all_wells_labels_val"],
                outputs=["dummy14", "well_count", "number_of_wells", "time_index", "wells_steam_rate_actual", "wells_emulsion_rate_actual"],
                name="targets_computation_val"
            ),
            node(
                func=predict_steam_then_oil,
                inputs=["dummy14", "well_count", "number_of_wells", "time_index", "timesteps_validation", "all_wells_steam_input_val", "all_wells_emulsion_input_val", "properties_val", "parameters", "algorithm"],
                outputs=["dummy15", "wells_steam_rate_predicted", "wells_emulsion_rate_predicted", "scheme"], 
                name="predict_steam_then_oil"
            ),
#             node(
#                 func=predict_oil_then_steam,
#                 inputs=["dummy14", "well_count", "number_of_wells", "time_index", "timesteps_validation", "all_wells_emulsion_input_val", "all_wells_steam_input_val", "properties_val", "parameters", "algorithm"],
#                 outputs=["dummy15", "wells_steam_rate_predicted", "wells_emulsion_rate_predicted", "scheme"], 
#                 name="predict_oil_then_steam"
#             ),
            node(
                func=save_predicted_data,
                inputs=["dummy15", "well_count", "all_wells_dates_input", "wells_steam_rate_actual", "wells_steam_rate_predicted", "wells_emulsion_rate_actual", "wells_emulsion_rate_predicted", "steam_input_column", "all_wells_steam_input_val", "emulsion_input_column", "all_wells_emulsion_input_val", "labels_column", "parameters", "files_val", "scheme"],
                outputs="dummy_validate",
                name="save_predicted_data"
            ),
        ]
    )
