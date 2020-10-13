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
from analytics_new.pipelines.data_engineering.nodes import (
    preprocess_raw_data,
    create_master_table,
    discrete_wavelet_transform, 
#     B_Splines, 
#     principal_component_analysis
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
                node(
                func=preprocess_raw_data,
                inputs="parameters",
                outputs=["timesteps", "Raw_Data_preprocessed", "Raw_Data_dated", "files", "filenames", "probabilities", "asset_names", "properties"],
                name="preprocess_raw_data"
            ),
                node(
                func=create_master_table,
                inputs=["Raw_Data_dated", "parameters"],
                outputs="master_table",
                name="master_table"
            ),
                node(
                func=discrete_wavelet_transform,
                inputs=["Raw_Data_preprocessed", "parameters", "files"],
                outputs="dummy",
                name="discrete_wavelet_transformation"
            ),
#                 node(
#                 func=B_Splines,
#                 inputs=["dummy", "Raw_Data_preprocessed", "parameters", "files"],
#                 outputs="dummy_Spline",
#                 name="splines"
#             ),
#                 node(
#                 func=principal_component_analysis,
#                 inputs=["dummy_Spline", "Raw_Data_preprocessed", "parameters", "files", "filenames"],
#                 outputs=None,
#                 name="Principal_Component_Analysis"
#             ),
        ]
    )
