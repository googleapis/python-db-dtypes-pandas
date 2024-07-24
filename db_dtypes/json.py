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

import operator
import json
import typing

import numpy as np
import pandas as pd
import pandas.arrays as arrays
import pandas.core.dtypes.common as common
import pandas.core.indexers as indexers
import db_dtypes.pandas_backports as pandas_backports
import pyarrow as pa
import pyarrow.compute as pc


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


# class JSONArray(pandas_backports.ArrowExtensionArray):
# class JSONArray(arrays.ArrowExtensionArray):
class JSONArray(pd.api.extensions.ExtensionArray):
    """Extension array that handles BigQuery JSON data, leveraging a string-based
    pyarrow array for storage. It enables seamless conversion to JSON objects when
    accessing individual elements."""

    _dtype = JSONDtype()

    def __init__(self, values, dtype=None, copy=False) -> None:
        self._dtype = JSONDtype()
        # super().__init__()
        if isinstance(values, pa.Array):
            self._pa_array = pa.chunked_array([values])
        elif isinstance(values, pa.ChunkedArray):
            self._pa_array = values
        else:
            raise ValueError(
                f"Unsupported type '{type(values)}' for JSONArray"
            )

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
            value = JSONArray._seralizate_json(value)
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
                value = [JSONArray._seralizate_json(x) for x in value]
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
        return cls(cls._box_pa(scalars, pa_type=pa.large_string()))

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype, copy = False
    ) -> JSONArray:
        """Construct a new ExtensionArray from a sequence of strings."""
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _concat_same_type(cls, to_concat) -> JSONArray:
        """Concatenate multiple JSONArray."""
        chunks = [array for ea in to_concat for array in ea._pa_array.iterchunks()]
        arr = pa.chunked_array(chunks, type=pa.large_string())
        return cls(arr)

    @classmethod
    def _from_factorized(cls, values, original):
        """Reconstruct an ExtensionArray after factorization."""
        return cls._from_sequence(values, dtype=original.dtype)

    @staticmethod
    def _seralizate_json(value):
        """A static method that converts a JSON value into a string representation."""
        if isinstance(value, str) or pd.isna(value):
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

    def __contains__(self, key) -> bool:
        """Return for `item in self`."""
        # TODO: check which super
        return bool(super().__contains__(JSONArray._seralizate_json(key)))

    def insert(self, loc: int, item) -> JSONArray:
        """
        Make new ExtensionArray inserting new item at location. Follows Python
        list.append semantics for negative values.
        """
        # TODO: check which super
        return super().insert(loc, JSONArray._seralizate_json(item))

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
            item = self._unpack_tuple_and_ellipses(item)

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

    def __len__(self) -> int:
        """Length of this array."""
        return len(self._pa_array)

    def _unpack_tuple_and_ellipses(self, item: tuple):
        """Possibly unpack arr[..., n] to arr[n]. Adapted from pandas.core.indexers."""
        if len(item) > 1:
            # Note: we are assuming this indexing is being done on a 1D arraylike
            if item[0] is Ellipsis:
                item = item[1:]
            elif item[-1] is Ellipsis:
                item = item[:-1]

        if len(item) > 1:
            raise IndexError("too many indices for array.")

        item = item[0]
        return item
    
    def _validate_indices(self, indices: np.ndarray, n: int) -> None:
        """Perform bounds-checking for an indexer. Adapted from pandas.core.indexers."""
        if len(indices):
            min_idx = indices.min()
            if min_idx < -1:
                msg = f"'indices' contains values less than allowed ({min_idx} < -1)"
                raise ValueError(msg)

            max_idx = indices.max()
            if max_idx >= n:
                raise IndexError("indices are out-of-bounds")

    def isna(self):
        """Boolean NumPy array indicating if each value is missing."""
        return self._pa_array.is_null().to_numpy()
    
    def copy(self):
        """
        Return a shallow copy of the array.

        Underlying ChunkedArray is immutable, so a deep copy is unnecessary.
        """
        return type(self)(self._pa_array)

    def take(self, indices, allow_fill = False, fill_value = None):
        """Take elements from an array."""
        indices_array = np.asanyarray(indices)

        if len(self._pa_array) == 0 and (indices_array >= 0).any():
            raise IndexError("cannot do a non-empty take")
        if indices_array.size > 0 and indices_array.max() >= len(self._pa_array):
            raise IndexError("out of bounds value in 'indices'.")

        if allow_fill:
            fill_mask = indices_array < 0
            if fill_mask.any():
                self._validate_indices(indices_array, len(self._pa_array))
                # TODO(ARROW-9433): Treat negative indices as NULL
                indices_array = pa.array(indices_array, mask=fill_mask)
                result = self._pa_array.take(indices_array)
                if pd.isna(fill_value):
                    return type(self)(result)
                # TODO: ArrowNotImplementedError: Function fill_null has no
                # kernel matching input types (array[string], scalar[string])
                result = type(self)(result)
                result[fill_mask] = fill_value
                return result
            else:
                # Nothing to fill
                return type(self)(self._pa_array.take(indices))
        else:  # allow_fill=False
            # TODO(ARROW-9432): Treat negative indices as indices from the right.
            if (indices_array < 0).any():
                # Don't modify in-place
                indices_array = np.copy(indices_array)
                indices_array[indices_array < 0] += len(self._pa_array)
            return type(self)(self._pa_array.take(indices_array))

    def __setitem__(self, key, value) -> None:
        """Set one or more values inplace."""
        key = indexers.check_array_indexer(self, key)
        value = self._box_pa(value, self._pa_array.type)

        if common.is_integer(key):
            # fast path
            key = typing.cast(int, key)
            n = len(self)
            if key < 0:
                key += n
            if not 0 <= key < n:
                raise IndexError(
                    f"index {key} is out of bounds for axis 0 with size {n}"
                )
            if isinstance(value, pa.Scalar):
                value = value.as_py()
            elif common.is_list_like(value):
                raise ValueError("Length of indexer and values mismatch")
            chunks = [
                *self._pa_array[:key].chunks,
                pa.array([value], type=self._pa_array.type, from_pandas=True),
                *self._pa_array[key + 1 :].chunks,
            ]
            data = pa.chunked_array(chunks).combine_chunks()

        elif common.is_bool_dtype(key):
            key = np.asarray(key, dtype=np.bool_)
            data = self._replace_with_mask(self._pa_array, key, value)

        elif common.is_scalar(value) or isinstance(value, pa.Scalar):
            mask = np.zeros(len(self), dtype=np.bool_)
            mask[key] = True
            data = self._if_else(mask, value, self._pa_array)

        else:
            indices = np.arange(len(self))[key]
            if len(indices) != len(value):
                raise ValueError("Length of indexer and values mismatch")
            if len(indices) == 0:
                return
            argsort = np.argsort(indices)
            indices = indices[argsort]
            value = value.take(argsort)
            mask = np.zeros(len(self), dtype=np.bool_)
            mask[indices] = True
            data = self._replace_with_mask(self._pa_array, mask, value)

        if isinstance(data, pa.Array):
            data = pa.chunked_array([data])
        self._pa_array = data

    def equals(self, other) -> bool:
        if not isinstance(other, JSONArray):
            return False
        return self._pa_array == other._pa_array

    @classmethod
    def _if_else(
        cls,
        cond: npt.NDArray[np.bool_] | bool,
        left: ArrayLike | Scalar,
        right: ArrayLike | Scalar,
    ):
        """
        Choose values based on a condition.

        Analogous to pyarrow.compute.if_else, with logic
        to fallback to numpy for unsupported types.

        Parameters
        ----------
        cond : npt.NDArray[np.bool_] or bool
        left : ArrayLike | Scalar
        right : ArrayLike | Scalar

        Returns
        -------
        pa.Array
        """
        try:
            return pc.if_else(cond, left, right)
        except pa.ArrowNotImplementedError:
            pass

        def _to_numpy_and_type(value) -> tuple[np.ndarray, pa.DataType | None]:
            if isinstance(value, (pa.Array, pa.ChunkedArray)):
                pa_type = value.type
            elif isinstance(value, pa.Scalar):
                pa_type = value.type
                value = value.as_py()
            else:
                pa_type = None
            return np.array(value, dtype=object), pa_type

        left, left_type = _to_numpy_and_type(left)
        right, right_type = _to_numpy_and_type(right)
        pa_type = left_type or right_type
        result = np.where(cond, left, right)
        return pa.array(result, type=pa_type, from_pandas=True)

    @classmethod
    def _replace_with_mask(
        cls,
        values,
        mask,
        replacements,
    ):
        """
        Replace items selected with a mask.

        Analogous to pyarrow.compute.replace_with_mask, with logic
        to fallback to numpy for unsupported types.

        Parameters
        ----------
        values : pa.Array or pa.ChunkedArray
        mask : npt.NDArray[np.bool_] or bool
        replacements : ArrayLike or Scalar
            Replacement value(s)

        Returns
        -------
        pa.Array or pa.ChunkedArray
        """
        if isinstance(replacements, pa.ChunkedArray):
            # replacements must be array or scalar, not ChunkedArray
            replacements = replacements.combine_chunks()
        if isinstance(values, pa.ChunkedArray) and pa.types.is_boolean(values.type):
            # GH#52059 replace_with_mask segfaults for chunked array
            # https://github.com/apache/arrow/issues/34634
            values = values.combine_chunks()
        try:
            return pc.replace_with_mask(values, mask, replacements)
        except pa.ArrowNotImplementedError:
            pass
        if isinstance(replacements, pa.Array):
            replacements = np.array(replacements, dtype=object)
        elif isinstance(replacements, pa.Scalar):
            replacements = replacements.as_py()
        result = np.array(values, dtype=object)
        result[mask] = replacements
        return pa.array(result, type=values.type, from_pandas=True)

    def __eq__(self, other):
        return self._cmp_method(other, operator.eq)

    