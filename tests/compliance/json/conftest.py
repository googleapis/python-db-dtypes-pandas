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


import json

import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest

from db_dtypes import JSONArray, JSONDtype


def make_data():
    # Sample data with varied lengths.
    samples = [
        {"id": 1, "bool_value": True},  # Boolean
        {"id": 2, "float_num": 3.14159},  # Floating
        {"id": 3, "date": "2024-07-16"},  # Dates (as strings)
        {"id": 4, "null_field": None},  # Null
        {"list_data": [10, 20, 30]},  # Lists
        {"person": {"name": "Alice", "age": 35}},  # Nested objects
        {"address": {"street": "123 Main St", "city": "Anytown"}},
        {"order": {"items": ["book", "pen"], "total": 15.99}},
    ]
    return np.random.default_rng(2).choice(samples, size=100)


@pytest.fixture
def dtype():
    return JSONDtype()


@pytest.fixture
def data():
    """Length-100 PeriodArray for semantics test."""
    data = make_data()

    # Why the while loop? NumPy is unable to construct an ndarray from
    # equal-length ndarrays. Many of our operations involve coercing the
    # EA to an ndarray of objects. To avoid random test failures, we ensure
    # that our data is coercible to an ndarray. Several tests deal with only
    # the first two elements, so that's what we'll check.

    while len(data[0]) == len(data[1]):
        print(data)
        data = make_data()

    return JSONArray._from_sequence(data)


@pytest.fixture
def data_for_twos(dtype):
    """
    Length-100 array in which all the elements are two.

    Call pytest.skip in your fixture if the dtype does not support divmod.
    """
    pytest.skip(f"{dtype} is not a numeric dtype")


@pytest.fixture
def data_missing():
    """Length 2 array with [NA, Valid]"""
    return JSONArray._from_sequence([None, {"a": 10}])


@pytest.fixture
def data_for_sorting():
    return JSONArray._from_sequence(
        [json.dumps({"b": 1}), json.dumps({"c": 4}), json.dumps({"a": 2, "c": 3})]
    )


@pytest.fixture
def data_missing_for_sorting():
    return JSONArray._from_sequence([json.dumps({"b": 1}), None, json.dumps({"a": 4})])


@pytest.fixture
def na_cmp():
    """
    Binary operator for comparing NA values.

    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.

    By default, uses ``operator.is_``
    """

    def cmp(a, b):
        return lambda left, right: pd.isna(left) and pd.isna(right)

    return cmp


@pytest.fixture
def data_for_grouping():
    return JSONArray._from_sequence(
        [
            json.dumps({"b": 1}),
            json.dumps({"b": 1}),
            None,
            None,
            json.dumps({"a": 0, "c": 2}),
            json.dumps({"a": 0, "c": 2}),
            json.dumps({"b": 1}),
            json.dumps({"c": 2}),
        ]
    )


@pytest.fixture
def data_repeated(data):
    """
    Generate many datasets.

    Parameters
    ----------
    data : fixture implementing `data`

    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """

    def gen(count):
        for _ in range(count):
            yield data

    return gen


_all_numeric_accumulations = ["cumsum", "cumprod", "cummin", "cummax"]


@pytest.fixture(params=_all_numeric_accumulations)
def all_numeric_accumulations(request):
    """
    Fixture for numeric accumulation names
    """
    return request.param


_all_boolean_reductions = ["all", "any"]


@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names.
    """
    return request.param


_all_numeric_reductions = [
    "count",
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
    "sem",
]


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names.
    """
    return request.param


@pytest.fixture(params=tm.arithmetic_dunder_methods)
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations.
    """
    return request.param


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture returning 'data' or 'data_missing' integer arrays.

    Used to test dtype conversion with and without missing values.
    """
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing
