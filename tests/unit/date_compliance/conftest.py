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


@pytest.fixture(params=[True, False])
def as_frame(request):
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.

    See:
    https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/conftest.py
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request):
    """
    Boolean fixture to support arr and Series(arr) comparison testing.

    See:
    https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/conftest.py
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
def data_for_grouping(dtype):
    """
    Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing

    See:
    https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/conftest.py
    """
    b = datetime.date(2022, 3, 9)
    a = datetime.date(1969, 12, 31)
    na = pandas.NaT
    return pandas.array([b, b, na, na, a, a, b], dtype=dtype)


@pytest.fixture
def data_for_sorting():
    """
    Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C

    See:
    https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/conftest.py
    """
    return DateArray(
        [
            datetime.date(2022, 1, 27),
            datetime.date(2022, 3, 9),
            datetime.date(1969, 12, 31),
        ]
    )


@pytest.fixture
def data_missing_for_sorting():
    """
    Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.

    See:
    https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/conftest.py
    """
    return DateArray(
        [datetime.date(2022, 1, 27), pandas.NaT, datetime.date(1969, 12, 31)]
    )


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]

    See:
    https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/conftest.py
    """
    return DateArray([None, datetime.date(2022, 1, 27)])


@pytest.fixture
def data_repeated(data):
    """
    Generate many datasets.

    See:
    https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/conftest.py
    """

    def gen(count):
        for _ in range(count):
            yield data

    return gen


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'

    See:
    https://github.com/pandas-dev/pandas/blob/main/pandas/tests/arrays/floating/conftest.py
    """
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


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


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """
    Simple fixture for testing keys in sorting methods.
    Tests None (no key) and the identity key.

    See: https://github.com/pandas-dev/pandas/blob/main/pandas/conftest.py
    """
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request):
    """
    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.

    See:
    https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/conftest.py
    """
    return request.param


@pytest.fixture
def invalid_scalar(data):
    """
    A scalar that *cannot* be held by this ExtensionArray.
    The default should work for most subclasses, but is not guaranteed.
    If the array can hold any item (i.e. object dtype), then use pytest.skip.

    See:
    https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/conftest.py
    """
    return object.__new__(object)
