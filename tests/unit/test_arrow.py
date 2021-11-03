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

import datetime as dt
from typing import Optional

import pandas
import pandas.api.extensions
import pandas.testing
import pyarrow
import pytest

import db_dtypes


def types_mapper(
    pyarrow_type: pyarrow.DataType,
) -> Optional[pandas.api.extensions.ExtensionDtype]:
    type_str = str(pyarrow_type)

    if type_str.startswith("date32") or type_str.startswith("date64"):
        return db_dtypes.DateDtype
    elif type_str.startswith("time32") or type_str.startswith("time64"):
        return db_dtypes.TimeDtype
    else:
        # Use default type mapping.
        return None


SERIES_ARRAYS_DEFAULT_TYPES = [
    (pandas.Series([], dtype="dbdate"), pyarrow.array([], type=pyarrow.date32())),
    (
        pandas.Series([None, None, None], dtype="dbdate"),
        pyarrow.array([None, None, None], type=pyarrow.date32()),
    ),
    (
        pandas.Series(
            [dt.date(2021, 9, 27), None, dt.date(2011, 9, 27)], dtype="dbdate"
        ),
        pyarrow.array(
            [dt.date(2021, 9, 27), None, dt.date(2011, 9, 27)], type=pyarrow.date32(),
        ),
    ),
    (
        pandas.Series(
            [dt.date(1677, 9, 22), dt.date(1970, 1, 1), dt.date(2262, 4, 11)],
            dtype="dbdate",
        ),
        pyarrow.array(
            [dt.date(1677, 9, 22), dt.date(1970, 1, 1), dt.date(2262, 4, 11)],
            type=pyarrow.date32(),
        ),
    ),
    (pandas.Series([], dtype="dbtime"), pyarrow.array([], type=pyarrow.time64("ns")),),
    (
        pandas.Series([None, None, None], dtype="dbtime"),
        pyarrow.array([None, None, None], type=pyarrow.time64("ns")),
    ),
    (
        pandas.Series(
            [dt.time(0, 0, 0, 0), None, dt.time(23, 59, 59, 999_999)], dtype="dbtime",
        ),
        pyarrow.array(
            [dt.time(0, 0, 0, 0), None, dt.time(23, 59, 59, 999_999)],
            type=pyarrow.time64("ns"),
        ),
    ),
    (
        pandas.Series(
            [
                dt.time(0, 0, 0, 0),
                dt.time(12, 30, 15, 125_000),
                dt.time(23, 59, 59, 999_999),
            ],
            dtype="dbtime",
        ),
        pyarrow.array(
            [
                dt.time(0, 0, 0, 0),
                dt.time(12, 30, 15, 125_000),
                dt.time(23, 59, 59, 999_999),
            ],
            type=pyarrow.time64("ns"),
        ),
    ),
]
SERIES_ARRAYS_CUSTOM_ARROW_TYPES = [
    (pandas.Series([], dtype="dbdate"), pyarrow.array([], type=pyarrow.date64())),
    (
        pandas.Series([None, None, None], dtype="dbdate"),
        pyarrow.array([None, None, None], type=pyarrow.date64()),
    ),
    (
        pandas.Series(
            [dt.date(2021, 9, 27), None, dt.date(2011, 9, 27)], dtype="dbdate"
        ),
        pyarrow.array(
            [dt.date(2021, 9, 27), None, dt.date(2011, 9, 27)], type=pyarrow.date64(),
        ),
    ),
    (
        pandas.Series(
            [dt.date(1677, 9, 22), dt.date(1970, 1, 1), dt.date(2262, 4, 11)],
            dtype="dbdate",
        ),
        pyarrow.array(
            [dt.date(1677, 9, 22), dt.date(1970, 1, 1), dt.date(2262, 4, 11)],
            type=pyarrow.date64(),
        ),
    ),
    (pandas.Series([], dtype="dbtime"), pyarrow.array([], type=pyarrow.time32("ms")),),
    (
        pandas.Series([None, None, None], dtype="dbtime"),
        pyarrow.array([None, None, None], type=pyarrow.time32("ms")),
    ),
    (
        pandas.Series(
            [dt.time(0, 0, 0, 0), None, dt.time(23, 59, 59, 999_000)], dtype="dbtime",
        ),
        pyarrow.array(
            [dt.time(0, 0, 0, 0), None, dt.time(23, 59, 59, 999_000)],
            type=pyarrow.time32("ms"),
        ),
    ),
    (
        pandas.Series(
            [dt.time(0, 0, 0, 0), None, dt.time(23, 59, 59, 999_999)], dtype="dbtime",
        ),
        pyarrow.array(
            [dt.time(0, 0, 0, 0), None, dt.time(23, 59, 59, 999_999)],
            type=pyarrow.time64("us"),
        ),
    ),
    (
        pandas.Series(
            [
                dt.time(0, 0, 0, 0),
                dt.time(12, 30, 15, 125_000),
                dt.time(23, 59, 59, 999_999),
            ],
            dtype="dbtime",
        ),
        pyarrow.array(
            [
                dt.time(0, 0, 0, 0),
                dt.time(12, 30, 15, 125_000),
                dt.time(23, 59, 59, 999_999),
            ],
            type=pyarrow.time64("us"),
        ),
    ),
]


@pytest.mark.parametrize(("series", "expected"), SERIES_ARRAYS_DEFAULT_TYPES)
def test_to_arrow(series, expected):
    array = pyarrow.array(series)
    assert array.equals(expected)


@pytest.mark.parametrize(("series", "expected"), SERIES_ARRAYS_CUSTOM_ARROW_TYPES)
def test_to_arrow_w_arrow_type(series, expected):
    array = pyarrow.array(series, type=expected.type)
    assert array.equals(expected)


@pytest.mark.parametrize(
    ["expected", "pyarrow_array"],
    SERIES_ARRAYS_DEFAULT_TYPES + SERIES_ARRAYS_CUSTOM_ARROW_TYPES,
)
def test_from_arrow(pyarrow_array: pyarrow.Array, expected: pandas.Series):
    # Convert to RecordBatch because types_mapper argument is ignored when
    # using a pyarrow.Array. https://issues.apache.org/jira/browse/ARROW-9664
    record_batch = pyarrow.RecordBatch.from_arrays([pyarrow_array], ["test_col"])
    dataframe = record_batch.to_pandas(date_as_object=False, types_mapper=types_mapper)
    series = dataframe["test_col"]
    pandas.testing.assert_series_equal(series, expected, check_names=False)
