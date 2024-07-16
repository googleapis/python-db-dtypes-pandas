# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
from pandas._libs import lib
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.numeric import NumericDtype
from pandas.core.dtypes.common import is_integer, is_scalar, pandas_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.indexers import check_array_indexer, unpack_tuple_and_ellipses
import pyarrow as pa
import pyarrow.compute as pc


@pd.api.extensions.register_extension_dtype
class JSONDtype(pd.api.extensions.ExtensionDtype):
    """Extension dtype for JSON data."""

    name = "dbjson"

    @property
    def na_value(self) -> pd.NA:
        return pd.NA

    @property
    def type(self) -> type[str]:
        return str

    @property
    def _is_numeric(self) -> bool:
        return False

    @property
    def _is_boolean(self) -> bool:
        return False

    @classmethod
    def construct_array_type(cls):
        """Return the array type associated with this dtype."""
        return JSONArray

    @staticmethod
    def __from_arrow__(array: typing.Union[pa.Array, pa.ChunkedArray]) -> JSONArray:
        """Convert to JSONArray from an Arrow array."""
        return JSONArray(array)


class JSONArray(ArrowExtensionArray):
    """Extension array containing JSON data."""

    _dtype = JSONDtype()

    def __init__(self, values, dtype=None, copy=False) -> None:
        if isinstance(values, (pa.Array, pa.ChunkedArray)) and pa.types.is_string(
            values.type
        ):
            values = pc.cast(values, pa.large_string())

        super().__init__(values)
        self._dtype = JSONDtype()

        if not pa.types.is_large_string(self._pa_array.type) and not (
            pa.types.is_dictionary(self._pa_array.type)
            and pa.types.is_large_string(self._pa_array.type.value_type)
        ):
            raise ValueError(
                "ArrowStringArray requires a PyArrow (chunked) array of "
                "large_string type"
            )

    @classmethod
    def _box_pa_scalar(cls, value, pa_type: pa.DataType | None = None) -> pa.Scalar:
        pa_scalar = super()._box_pa_scalar(value, pa_type)
        if pa.types.is_string(pa_scalar.type) and pa_type is None:
            pa_scalar = pc.cast(pa_scalar, pa.large_string())
        return pa_scalar

    @classmethod
    def _box_pa_array(
        cls, value, pa_type: pa.DataType | None = None, copy: bool = False
    ) -> pa.Array | pa.ChunkedArray:
        pa_array = super()._box_pa_array(value, pa_type)
        if pa.types.is_string(pa_array.type) and pa_type is None:
            pa_array = pc.cast(pa_array, pa.large_string())
        return pa_array

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        from pandas.core.arrays.masked import BaseMaskedArray

        if isinstance(scalars, BaseMaskedArray):
            # avoid costly conversion to object dtype in ensure_string_array and
            # numerical issues with Float32Dtype
            na_values = scalars._mask
            result = scalars._data
            result = lib.ensure_string_array(result, copy=copy, convert_na_value=False)
            return cls(pa.array(result, mask=na_values, type=pa.large_string()))
        elif isinstance(scalars, (pa.Array, pa.ChunkedArray)):
            return cls(pc.cast(scalars, pa.large_string()))

        # convert non-na-likes to str
        result = lib.ensure_string_array(scalars, copy=copy)
        return cls(pa.array(result, type=pa.large_string(), from_pandas=True))

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: ExtensionDtype, copy: bool = False
    ) -> JSONArray:
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @property
    def dtype(self) -> JSONDtype:
        """An instance of JSONDtype"""
        return self._dtype

    def insert(self, loc: int, item) -> JSONArray:
        if not isinstance(item, str) and not pd.isna(item):
            raise TypeError("Scalar must be NA or str")
        return super().insert(loc, item)

    def astype(self, dtype, copy: bool = True):
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self
        elif isinstance(dtype, NumericDtype):
            data = self._pa_array.cast(pa.from_numpy_dtype(dtype.numpy_dtype))
            return dtype.__from_arrow__(data)
        elif isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.floating):
            return self.to_numpy(dtype=dtype, na_value=np.nan)

        return super().astype(dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls._from_sequence(values, dtype=original.dtype)

    def __getitem__(self, item):
        """Select a subset of self.

        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.
            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None
            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'

        Returns
        -------
        item : scalar or ExtensionArray

        Notes
        -----
        For scalar ``item``, return a scalar value suitable for the array's
        type. This should be an instance of ``self.dtype.type``.
        For slice ``key``, return an instance of ``ExtensionArray``, even
        if the slice is length 0 or 1.
        For a boolean mask, return an instance of ``ExtensionArray``, filtered
        to the values where ``item`` is True.
        """
        item = check_array_indexer(self, item)

        if isinstance(item, np.ndarray):
            if not len(item):
                return type(self)(pa.chunked_array([], type=pa.string()))
            elif item.dtype.kind in "iu":
                return self.take(item)
            elif item.dtype.kind == "b":
                return type(self)(self._pa_array.filter(item))
            else:
                raise IndexError(
                    "Only integers, slices and integer or "
                    "boolean arrays are valid indices."
                )
        elif isinstance(item, tuple):
            item = unpack_tuple_and_ellipses(item)

        if is_scalar(item) and not is_integer(item):
            # e.g. "foo" or 2.5
            # exception message copied from numpy
            raise IndexError(
                r"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                r"(`None`) and integer or boolean arrays are valid indices"
            )
        # We are not an array indexer, so maybe e.g. a slice or integer
        # indexer. We dispatch to pyarrow.
        if isinstance(item, slice):
            # Arrow bug https://github.com/apache/arrow/issues/38768
            if item.start == item.stop:
                pass
            elif (
                item.stop is not None
                and item.stop < -len(self)
                and item.step is not None
                and item.step < 0
            ):
                item = slice(item.start, None, item.step)

        value = self._pa_array[item]
        if isinstance(value, pa.ChunkedArray):
            return type(self)(value)
        else:
            scalar = value.as_py()
            if scalar is None:
                return self._dtype.na_value
            else:
                return scalar

    @classmethod
    def _result_converter(cls, values, na=None):
        return pd.BooleanDtype().__from_arrow__(values)

    @classmethod
    def _concat_same_type(cls, to_concat) -> JSONArray:
        """
        Concatenate multiple JSONArray.

        Parameters
        ----------
        to_concat : sequence of JSONArray

        Returns
        -------
        JSONArray
        """
        chunks = [array for ea in to_concat for array in ea._pa_array.iterchunks()]
        arr = pa.chunked_array(chunks, type=pa.large_string())
        return cls(arr)

    def _pad_or_backfill(self, *, method, limit=None, copy=True):
        # GH#56616 - test EA method without limit_area argument
        return super()._pad_or_backfill(method=method, limit=limit, copy=copy)
