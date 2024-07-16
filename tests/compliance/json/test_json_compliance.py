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
"""
Tests for extension interface compliance, inherited from pandas.

See:
https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/decimal/test_decimal.py
and
https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/test_period.py
"""

import typing

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
import pyarrow as pa
import pytest

from db_dtypes import JSONArray

# We intentionally don't run base.BaseSetitemTests because pandas'
# internals has trouble setting sequences of values into scalar positions.
unhashable = pytest.mark.xfail(reason="Unhashable")


class TestJSONArray(base.ExtensionTests):
    @pytest.mark.parametrize(
        "limit_area, input_ilocs, expected_ilocs",
        [
            ("outside", [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]),
            ("outside", [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]),
            ("outside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]),
            ("outside", [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]),
            ("inside", [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]),
            ("inside", [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]),
            ("inside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]),
            ("inside", [0, 1, 0, 1, 0], [0, 1, 1, 1, 0]),
        ],
    )
    def test_ffill_limit_area(
        self, data_missing, limit_area, input_ilocs, expected_ilocs
    ):
        # GH#56616
        msg = "JSONArray does not implement limit_area"
        with pytest.raises(NotImplementedError, match=msg):
            super().test_ffill_limit_area(
                data_missing, limit_area, input_ilocs, expected_ilocs
            )

    @unhashable
    def test_value_counts_with_normalize(self, data):
        super().test_value_counts_with_normalize(data)

    @pytest.mark.xfail(reason="combine for JSONArray not supported")
    def test_combine_le(self, data_repeated):
        super().test_combine_le(data_repeated)

    @pytest.mark.xfail(
        reason="combine for JSONArray not supported - "
        "may pass depending on random data",
        strict=False,
        raises=AssertionError,
    )
    def test_combine_first(self, data):
        super().test_combine_first(data)

    @pytest.mark.skip(reason="2D support not implemented for JSONArray")
    def test_view(self, data):
        super().test_view(data)

    @pytest.mark.skip(reason="2D support not implemented for JSONArray")
    def test_setitem_preserves_views(self, data):
        super().test_setitem_preserves_views(data)

    @pytest.mark.skip(reason="2D support not implemented for JSONArray")
    def test_transpose(self, data):
        super().test_transpose(data)

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        if len(data[0]) != 1:
            mark = pytest.mark.xfail(reason="raises in coercing to Series")
            request.applymarker(mark)
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators, request):
        if len(data[0]) != 1:
            mark = pytest.mark.xfail(reason="raises in coercing to Series")
            request.applymarker(mark)
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        return op_name in ["min", "max"]

    def _get_expected_exception(
        self, op_name: str, obj, other
    ) -> type[Exception] | None:
        if op_name in ["__divmod__", "__rdivmod__"]:
            if isinstance(obj, pd.Series) or isinstance(other, pd.Series):
                return NotImplementedError
            return TypeError
        elif op_name in ["__mod__", "__rmod__", "__pow__", "__rpow__"]:
            return NotImplementedError
        elif op_name in ["__mul__", "__rmul__"]:
            # Can only multiply strings by integers
            return TypeError
        elif op_name in [
            "__truediv__",
            "__rtruediv__",
            "__floordiv__",
            "__rfloordiv__",
            "__sub__",
            "__rsub__",
        ]:
            return pa.ArrowNotImplementedError

        return None

    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        dtype = typing.cast(pd.StringDtype, tm.get_dtype(obj))
        if op_name in ["__add__", "__radd__"]:
            cast_to = dtype
        else:
            cast_to = "boolean[pyarrow]"  # type: ignore[assignment]
        return pointwise_result.astype(cast_to)


def custom_assert_frame_equal(left, right, *args, **kwargs):
    obj_type = kwargs.get("obj", "DataFrame")
    tm.assert_index_equal(
        left.columns,
        right.columns,
        exact=kwargs.get("check_column_type", "equiv"),
        check_names=kwargs.get("check_names", True),
        check_exact=kwargs.get("check_exact", False),
        check_categorical=kwargs.get("check_categorical", True),
        obj=f"{obj_type}.columns",
    )

    jsons = (left.dtypes == "json").index

    for col in jsons:
        tm.assert_series_equal(left[col], right[col], *args, **kwargs)

    left = left.drop(columns=jsons)
    right = right.drop(columns=jsons)
    tm.assert_frame_equal(left, right, *args, **kwargs)


def test_custom_asserts():
    data = JSONArray._from_sequence(
        [
            {"a": 1},
            {"b": 2},
            {"c": 3},
        ]
    )
    a = pd.Series(data)
    tm.assert_series_equal(a, a)
    custom_assert_frame_equal(a.to_frame(), a.to_frame())

    b = pd.Series(data.take([0, 0, 1]))
    with pytest.raises(AssertionError):
        tm.assert_series_equal(a, b)

    with pytest.raises(AssertionError):
        custom_assert_frame_equal(a.to_frame(), b.to_frame())
