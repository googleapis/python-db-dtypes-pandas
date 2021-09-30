# Copyright 2021 Google LLC
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

import pandas
import pytest

# To register the types.
import db_dtypes  # noqa


@pytest.mark.parametrize(
    "value, error",
    [
        ("thursday", "Bad date string: 'thursday'"),
        ("1-2-thursday", "Bad date string: '1-2-thursday'"),
        ("1-2-3-4", "Bad date string: '1-2-3-4'"),
        ("1-2-3.f", "Bad date string: '1-2-3.f'"),
        ("1-d-3", "Bad date string: '1-d-3'"),
        ("1-3", "Bad date string: '1-3'"),
        ("1", "Bad date string: '1'"),
        ("", "Bad date string: ''"),
        ("2021-2-99", "day is out of range for month"),
        ("2021-99-1", "month must be in 1[.][.]12"),
        ("10000-1-1", "year 10000 is out of range"),
    ],
)
def test_bad_date_parsing(value, error):
    with pytest.raises(ValueError, match=error):
        pandas.Series([value], dtype="date")
