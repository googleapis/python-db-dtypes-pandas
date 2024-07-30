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
import typing

import numpy as np
import pandas as pd
import pandas._testing as tm
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.tests.extension import base
import pytest


class TestJSONArray(base.ExtensionTests):
    @pytest.mark.xfail(reason="Unhashable")
    def test_value_counts_with_normalize(self, data):
        super().test_value_counts_with_normalize(data)

    @pytest.mark.xfail(reason="Unhashable")
    def test_groupby_extension_transform(self):
        """
        This currently fails in Series.name.setter, since the
        name must be hashable, but the value is a dictionary.
        I think this is what we want, i.e. `.name` should be the original
        values, and not the values for factorization.
        """
        super().test_groupby_extension_transform()

    @pytest.mark.xfail(reason="Unhashable")
    def test_groupby_extension_apply(self):
        """
        This fails in Index._do_unique_check with
        >   hash(val)
        E   TypeError: unhashable type: 'dict' with
        I suspect that once we support Index[ExtensionArray],
        we'll be able to dispatch unique.
        """
        super().test_groupby_extension_apply()

    @pytest.mark.xfail(reason="Unhashable")
    def test_sort_values_frame(self):
        super().test_sort_values_frame()

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

    @pytest.mark.xfail(reason="Arithmetic functions is not supported for json")
    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators):
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    @pytest.mark.xfail(reason="Arithmetic functions is not supported for json")
    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    @pytest.mark.xfail(reason="Arithmetic functions is not supported for json")
    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        super().test_arith_series_with_array(data, all_arithmetic_operators)

    @pytest.mark.xfail(reason="Arithmetic functions is not supported for json")
    def test_add_series_with_extension_array(self, data):
        super().test_add_series_with_extension_array(data, data)

    @pytest.mark.xfail(reason="Arithmetic functions is not supported for json")
    def test_divmod(self, data):
        super().test_divmod(data, data)

    def test_compare_array(self, data, comparison_op, request):
        if comparison_op.__name__ not in ["eq", "ne"]:
            mark = pytest.mark.xfail(reason="Comparison methods not implemented")
            request.applymarker(mark)
        super().test_compare_array(data, comparison_op)

    def test_compare_scalar(self, data, comparison_op, request):
        if comparison_op.__name__ not in ["eq", "ne"]:
            mark = pytest.mark.xfail(reason="Comparison methods not implemented")
            request.applymarker(mark)
        super().test_compare_scalar(data, comparison_op)

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        return False

    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        dtype = typing.cast(pd.StringDtype, tm.get_dtype(obj))
        if op_name in ["__add__", "__radd__"]:
            cast_to = dtype
        else:
            cast_to = "boolean[pyarrow]"  # type: ignore[assignment]
        return pointwise_result.astype(cast_to)

    @pytest.mark.skip(reason="'<' not supported between instances of 'dict' and 'dict'")
    def test_searchsorted(self, data_for_sorting, as_series):
        super().test_searchsorted(self, data_for_sorting, as_series)

    def test_astype_str(self, data):
        # Use `json.dumps(str)` instead of passing `str(obj)` directly to the super method.
        result = pd.Series(data[:5]).astype(str)
        expected = pd.Series(
            [json.dumps(x, sort_keys=True) for x in data[:5]], dtype=str
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "nullable_string_dtype",
        [
            "string[python]",
            "string[pyarrow]",
        ],
    )
    def test_astype_string(self, data, nullable_string_dtype):
        # Use `json.dumps(str)` instead of passing `str(obj)` directly to the super method.
        result = pd.Series(data[:5]).astype(nullable_string_dtype)
        expected = pd.Series(
            [json.dumps(x, sort_keys=True) for x in data[:5]],
            dtype=nullable_string_dtype,
        )
        tm.assert_series_equal(result, expected)

    def test_array_interface(self, data):
        result = np.array(data)
        # Use `json.dumps(data[0])` instead of passing `data[0]` directly to the super method.
        assert result[0] == json.dumps(data[0])

        result = np.array(data, dtype=object)
        # Use `json.dumps(x)` instead of passing `x` directly to the super method.
        expected = np.array([json.dumps(x) for x in data], dtype=object)
        if expected.ndim > 1:
            # nested data, explicitly construct as 1D
            expected = construct_1d_object_array_from_listlike(list(data))
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.xfail(reason="Setting a dict as a scalar")
    def test_fillna_series(self):
        """We treat dictionaries as a mapping in fillna, not a scalar."""
        super().test_fillna_series()

    @pytest.mark.xfail(reason="Setting a dict as a scalar")
    def test_fillna_frame(self):
        """We treat dictionaries as a mapping in fillna, not a scalar."""
        super().test_fillna_frame()

    @pytest.mark.skip("fill-value is interpreted as a dict of values")
    def test_fillna_copy_frame(self, data_missing):
        super().test_fillna_copy_frame(data_missing)

    def test_from_dtype(self, data):
        # construct from our dtype & string dtype
        dtype = data.dtype

        expected = pd.Series(data)
        result = pd.Series(list(data), dtype=dtype)
        tm.assert_series_equal(result, expected)

        result = pd.Series(list(data), dtype=str(dtype))
        tm.assert_series_equal(result, expected)

        # Use `{"col1": data}` instead of passing `data` directly to the super method.
        # This prevents the DataFrame constructor from attempting to interpret the
        # dictionary as column headers.

        # gh-30280
        expected = pd.DataFrame({"col1": data}).astype(dtype)
        result = pd.DataFrame({"col1": list(data)}, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        result = pd.DataFrame({"col1": list(data)}, dtype=str(dtype))
        tm.assert_frame_equal(result, expected)

    def test_series_constructor_scalar_with_index(self, data, dtype):
        # Use json.dumps(data[0]) instead of passing data[0] directly to the super method.
        # This prevents the Series constructor from attempting to interpret the dictionary
        # as column headers.
        scalar = json.dumps(data[0])
        result = pd.Series(scalar, index=[1, 2, 3], dtype=dtype)
        expected = pd.Series([scalar] * 3, index=[1, 2, 3], dtype=dtype)
        tm.assert_series_equal(result, expected)

        result = pd.Series(scalar, index=["foo"], dtype=dtype)
        expected = pd.Series([scalar], index=["foo"], dtype=dtype)
        tm.assert_series_equal(result, expected)

    # Patching `[....] * len()` to base.BaseSetitemTests because pandas' internals
    # has trouble setting sequences of values into scalar positions.

    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_setitem_integer_array(self, data, idx, box_in_series):
        arr = data[:5].copy()
        expected = data.take([0, 0, 0, 3, 4])

        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)

        # Use `[arr[0]] * len()` instead of passing `arr[0]` directly to the super method.
        arr[idx] = [arr[0]] * len(arr[idx])
        tm.assert_equal(arr, expected)

    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_broadcast(self, data, setter):
        ser = pd.Series(data)
        mask = np.zeros(len(data), dtype=bool)
        mask[:2] = True

        if setter:  # loc
            target = getattr(ser, setter)
        else:  # __setitem__
            target = ser

        # Use `[data[10]] * len()` instead of passing `data[10]` directly to the super method.
        target[mask] = [data[10]] * len(target[mask])
        assert ser[0] == data[10]
        assert ser[1] == data[10]

    def test_setitem_loc_scalar_mixed(self, data):
        df = pd.DataFrame({"A": np.arange(len(data)), "B": data})
        # Use `[data[1]]` instead of passing `data[1]` directly to the super method.
        df.loc[0, "B"] = [data[1]]
        assert df.loc[0, "B"] == data[1]

    @pytest.mark.xfail(reason="TODO: open an issue for ArrowExtentionArray")
    def test_setitem_loc_scalar_single(self, data):
        super().test_setitem_loc_scalar_single(data)

    def test_setitem_loc_iloc_slice(self, data):
        arr = data[:5].copy()
        s = pd.Series(arr, index=["a", "b", "c", "d", "e"])
        expected = pd.Series(data.take([0, 0, 0, 3, 4]), index=s.index)

        result = s.copy()
        # Use `[data[0]] * len()` instead of passing `data[0]` directly to the super method.
        result.iloc[:3] = [data[0]] * len(result.iloc[:3])
        tm.assert_equal(result, expected)

        result = s.copy()
        result.loc[:"c"] = [data[0]] * len(result.loc[:"c"])
        tm.assert_equal(result, expected)

    @pytest.mark.xfail(reason="TODO: open an issue for ArrowExtentionArray")
    def test_setitem_iloc_scalar_single(self, data):
        super().test_setitem_iloc_scalar_single(data)

    def test_setitem_iloc_scalar_mixed(self, data):
        df = pd.DataFrame({"A": np.arange(len(data)), "B": data})
        # Use `[data[1]] * len()` instead of passing `data[1]` directly to the super method.
        df.iloc[0, 1] = [data[1]] * len(df.iloc[0, 1])
        assert df.loc[0, "B"] == data[1]

    @pytest.mark.xfail(reason="eq not implemented for <class 'dict'>")
    def test_setitem_mask_boolean_array_with_na(self, data, box_in_series):
        super().test_setitem_mask_boolean_array_with_na(data, box_in_series)

    @pytest.mark.parametrize("setter", ["loc", "iloc"])
    @pytest.mark.xfail(reason="TODO: open an issue for ArrowExtentionArray")
    def test_setitem_scalar(self, data, setter):
        super().test_setitem_scalar(data, setter)

    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype="boolean"),
            pd.array([True, True, True, pd.NA, pd.NA], dtype="boolean"),
        ],
        ids=["numpy-array", "boolean-array", "boolean-array-na"],
    )
    def test_setitem_mask(self, data, mask, box_in_series):
        arr = data[:5].copy()
        expected = arr.take([0, 0, 0, 3, 4])
        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)
        # Use `[data[0]] * len()` instead of passing `data[0]` directly to the super method.
        arr[mask] = [data[0]] * len(arr[mask])
        tm.assert_equal(expected, arr)

    @pytest.mark.xfail(reasons="Setting a `dict` to an expansion row is not supported")
    def test_setitem_with_expansion_row(self, data, na_value):
        super().test_setitem_with_expansion_row(data, na_value)

    def test_setitem_iloc_scalar_multiple_homogoneous(self, data):
        df = pd.DataFrame({"A": data, "B": data})
        # Use `[data[1]]` instead of passing `data[1]` directly to the super method.
        df.iloc[10, 1] = [data[1]]
        assert df.loc[10, "B"] == data[1]

    def test_setitem_loc_scalar_multiple_homogoneous(self, data):
        df = pd.DataFrame({"A": data, "B": data})
        # Use `[data[1]]` instead of passing `data[1]` directly to the super method.
        df.loc[10, "B"] = [data[1]]
        assert df.loc[10, "B"] == data[1]

    def test_setitem_slice(self, data, box_in_series):
        arr = data[:5].copy()
        expected = data.take([0, 0, 0, 3, 4])
        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)

        # Use `[data[0]] * 3` instead of passing `data[0]` directly to the super method.
        arr[:3] = [data[0]] * 3
        tm.assert_equal(arr, expected)

    @pytest.mark.xfail(reason="only integer scalar arrays can be converted")
    def test_setitem_2d_values(self, data):
        super().test_setitem_2d_values(data)

    @pytest.mark.xfail(reason="data type 'json' not understood")
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data, request):
        super().test_EA_types(engine, data, request)

    @pytest.mark.xfail(
        reason="`to_numpy` returns serialized JSON, "
        + "while `__getitem__` returns JSON objects."
    )
    def test_setitem_frame_2d_values(self, data):
        super().test_setitem_frame_2d_values(data)

    @pytest.mark.xfail(
        reason="`to_numpy` returns serialized JSON, "
        + "while `__getitem__` returns JSON objects."
    )
    def test_transpose_frame(self, data):
        # `DataFrame.T` calls `to_numpy` to get results.
        super().test_transpose_frame(data)

    @pytest.mark.xfail(
        reason="`to_numpy` returns serialized JSON, "
        + "while `__getitem__` returns JSON objects."
    )
    def test_where_series(self, data, na_value, as_frame):
        # `Series.where` calls `to_numpy` to get results.
        super().test_where_series(data, na_value, as_frame)
