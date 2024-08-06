# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import datetime as dt
from typing import Optional

import pandas
import pandas.api.extensions
import pandas.testing
import pyarrow
import pytest

import packaging.version

import db_dtypes

is_supported_version = packaging.version.Version(pandas.__version__) >= packaging.version.Version("1.5.0")

@pytest.mark.skipif(not is_supported_version, reason="requires Pandas 1.5.0 and above")
def test_constructor_from_sequence():
    json_obj = [0, "str", {"a": 0, "b": 1}]
    data = db_dtypes.JSONArray._from_sequence(json_obj)
