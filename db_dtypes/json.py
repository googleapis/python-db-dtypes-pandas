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

import json
import typing

import numpy as np
import pandas as pd
import pandas.arrays as arrays
import pandas.core.dtypes.common as common
import pandas.core.indexers as indexers
import pyarrow as pa
import pyarrow.compute

ARROW_CMP_FUNCS = {
    "eq": pyarrow.compute.equal,
    "ne": pyarrow.compute.not_equal,
    "lt": pyarrow.compute.less,
    "gt": pyarrow.compute.greater,
    "le": pyarrow.compute.less_equal,
    "ge": pyarrow.compute.greater_equal,
}

@pd.api.extensions.register_extension_dtype
class JSONDtype(pd.api.extensions.ExtensionDtype):
    """Extension dtype for BigQuery JSON data."""

    name = "dbjson"

    @property
    def na_value(self) -> pd.NA:
        """Default NA value to use for this type."""
        return pd.NA

    @property
    def type(self) -> type[str]:
        """Return the scalar type for the array, e.g. int."""
        return dict

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


class JSONArray(arrays.ArrowExtensionArray):
    """Extension array that handles BigQuery JSON data, leveraging a string-based
    pyarrow array for storage. It enables seamless conversion to JSON objects when
    accessing individual elements."""

    _dtype = JSONDtype()

    def __init__(self, values, dtype=None, copy=False) -> None:
        self._dtype = JSONDtype()
        if isinstance(values, pa.Array):
            self._pa_array = pa.chunked_array([values])
        elif isinstance(values, pa.ChunkedArray):
            self._pa_array = values
        else:
            raise ValueError(f"Unsupported type '{type(values)}' for JSONArray")

    @classmethod
    def _box_pa(
        cls, value, pa_type: pa.DataType | None = None
    ) -> pa.Array | pa.ChunkedArray | pa.Scalar:
        """Box value into a pyarrow Array, ChunkedArray or Scalar."""
        if isinstance(value, pa.Scalar) or not (
            common.is_list_like(value) and not common.is_dict_like(value)
        ):
            return cls._box_pa_scalar(value, pa_type)
        return cls._box_pa_array(value, pa_type)

    @classmethod
    def _box_pa_scalar(cls, value, pa_type: pa.DataType | None = None) -> pa.Scalar:
        """Box value into a pyarrow Scalar."""
        if isinstance(value, pa.Scalar):
            pa_scalar = value
        if pd.isna(value):
            pa_scalar = pa.scalar(None, type=pa_type)
        else:
            value = JSONArray._serialize_json(value)
            pa_scalar = pa.scalar(value, type=pa_type, from_pandas=True)

        if pa_type is not None and pa_scalar.type != pa_type:
            pa_scalar = pa_scalar.cast(pa_type)
        return pa_scalar

    @classmethod
    def _box_pa_array(
        cls, value, pa_type: pa.DataType | None = None, copy: bool = False
    ) -> pa.Array | pa.ChunkedArray:
        """Box value into a pyarrow Array or ChunkedArray."""
        if isinstance(value, cls):
            pa_array = value._pa_array
        elif isinstance(value, (pa.Array, pa.ChunkedArray)):
            pa_array = value
        else:
            try:
                value = [JSONArray._serialize_json(x) for x in value]
                pa_array = pa.array(value, type=pa_type, from_pandas=True)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                # GH50430: let pyarrow infer type, then cast
                pa_array = pa.array(value, from_pandas=True)

        if pa_type is not None and pa_array.type != pa_type:
            pa_array = pa_array.cast(pa_type)

        return pa_array

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        """Construct a new ExtensionArray from a sequence of scalars."""
        result = []
        for scalar in scalars:
            result.append(JSONArray._serialize_json(scalar))
        return cls(pa.array(result, type=pa.string(), from_pandas=True))

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype, copy: bool = False
    ) -> JSONArray:
        """Construct a new ExtensionArray from a sequence of strings."""
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _concat_same_type(cls, to_concat) -> JSONArray:
        """Concatenate multiple JSONArray."""
        chunks = [array for ea in to_concat for array in ea._pa_array.iterchunks()]
        arr = pa.chunked_array(chunks, type=pa.string())
        return cls(arr)

    @classmethod
    def _from_factorized(cls, values, original):
        """Reconstruct an ExtensionArray after factorization."""
        return cls._from_sequence(values, dtype=original.dtype)

    @staticmethod
    def _serialize_json(value):
        """A static method that converts a JSON value into a string representation."""
        if pd.isna(value):
            return value
        else:
            # `sort_keys=True` sorts dictionary keys before serialization, making
            # JSON comparisons deterministic.
            return json.dumps(value, sort_keys=True)

    @staticmethod
    def _deserialize_json(value):
        """A static method that converts a JSON string back into its original value."""
        if not pd.isna(value):
            return json.loads(value)
        else:
            return value

    @property
    def dtype(self) -> JSONDtype:
        """An instance of JSONDtype"""
        return self._dtype

    def _cmp_method(self, other, op):
        pc_func = ARROW_CMP_FUNCS[op.__name__]
        result = pc_func(self._pa_array, self._box_pa(other))
        return arrays.ArrowExtensionArray(result)

    def __getitem__(self, item):
        """Select a subset of self."""
        item = indexers.check_array_indexer(self, item)

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
            item = indexers.unpack_tuple_and_ellipses(item)

        if common.is_scalar(item) and not common.is_integer(item):
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
            scalar = JSONArray._deserialize_json(value.as_py())
            if scalar is None:
                return self._dtype.na_value
            else:
                return scalar

    def __iter__(self):
        """Iterate over elements of the array."""
        for value in self._pa_array:
            val = JSONArray._deserialize_json(value.as_py())
            if val is None:
                yield self._dtype.na_value
            else:
                yield val

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        """Return a scalar result of performing the reduction operation."""
        if name in ["min", "max"]:
            raise TypeError("JSONArray does not support min/max reducntion.")
        super()._reduce(name, skipna=skipna, keepdims=keepdims, **kwargs)

    def __array__(
        self, dtype = None, copy = None
    ) -> np.ndarray:
        """Correctly construct numpy arrays when passed to `np.asarray()`."""
        return self.to_numpy(dtype=dtype)

    def to_numpy(self, dtype = None, copy = False, na_value = pd.NA) -> np.ndarray:
        dtype, na_value = self._to_numpy_dtype_inference(dtype, na_value, self._hasna)
        pa_type = self._pa_array.type
        if not self._hasna or pd.isna(na_value) or pa.types.is_null(pa_type):
            data = self
        else:
            data = self.fillna(na_value)
        result = np.array(list(data), dtype=dtype)
        
        if data._hasna:
            result[data.isna()] = na_value
        return result

    def _to_numpy_dtype_inference(
        self, dtype, na_value, hasna
    ):
        if dtype is not None:
            dtype = np.dtype(dtype)

        if dtype is None or not hasna:
            na_value = self.dtype.na_value
        elif dtype.kind == "f":  # type: ignore[union-attr]
            na_value = np.nan
        elif dtype.kind == "M":  # type: ignore[union-attr]
            na_value = np.datetime64("nat")
        elif dtype.kind == "m":  # type: ignore[union-attr]
            na_value = np.timedelta64("nat")
        else:
            na_value = self.dtype.na_value
        return dtype, na_value