# Copyright 2022 Google LLC
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

import datetime

import numpy
import pandas
import pytest

from db_dtypes import DateArray, DateDtype


_all_numeric_reductions = [
    "sum",
    "max",
    "min",
    "mean",
    "prod",
    "std",
    "var",
    "median",
    "kurt",
    "skew",
]


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names.

    See: https://github.com/pandas-dev/pandas/blob/main/pandas/conftest.py
    """
    return request.param


_all_boolean_reductions = ["all", "any"]


@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names.

    See: https://github.com/pandas-dev/pandas/blob/main/pandas/conftest.py
    """
    return request.param


@pytest.fixture
def data():
    return DateArray(
        numpy.arange(
            datetime.datetime(1900, 1, 1),
            datetime.datetime(2099, 12, 31),
            datetime.timedelta(days=731),
            dtype="datetime64[ns]",
        )
    )


@pytest.fixture
def data_missing():
    return DateArray([None, datetime.date(2022, 1, 27)])


@pytest.fixture
def dtype():
    return DateDtype()


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.

    See:
    https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/conftest.py
    """
    return request.param


@pytest.fixture
def na_value():
    return pandas.NaT


@pytest.fixture
def na_cmp():
    def cmp(a, b):
        return a is pandas.NaT and a is b

    return cmp
