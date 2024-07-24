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

"""
Utilities to support older pandas versions.

These backported versions are simpler and, in some cases, less featureful than
the versions in the later versions of pandas.
"""

import operator
import typing

import numpy as np
import packaging.version
import pandas as pd
from pandas.api.types import is_integer
import pandas.compat.numpy.function
import pandas.core.nanops
import pyarrow as pa
from pandas.core.dtypes.dtypes import ArrowDtype
import pandas.core.dtypes.common as common
import pandas.core.indexers as indexers

pandas_release = packaging.version.parse(pandas.__version__).release

# Create aliases for private methods in case they move in a future version.
nanall = pandas.core.nanops.nanall
nanany = pandas.core.nanops.nanany
nanmax = pandas.core.nanops.nanmax
nanmin = pandas.core.nanops.nanmin
numpy_validate_all = pandas.compat.numpy.function.validate_all
numpy_validate_any = pandas.compat.numpy.function.validate_any
numpy_validate_max = pandas.compat.numpy.function.validate_max
numpy_validate_min = pandas.compat.numpy.function.validate_min

if pandas_release >= (1, 3):
    nanmedian = pandas.core.nanops.nanmedian
    numpy_validate_median = pandas.compat.numpy.function.validate_median


def import_default(module_name, force=False, default=None):
    """
    Provide an implementation for a class or function when it can't be imported

    or when force is True.

    This is used to replicate Pandas APIs that are missing or insufficient
    (thus the force option) in early pandas versions.
    """

    if default is None:
        return lambda func_or_class: import_default(module_name, force, func_or_class)

    if force:
        return default

    name = default.__name__
    try:
        module = __import__(module_name, {}, {}, [name])
    except ModuleNotFoundError:
        return default

    return getattr(module, name, default)


# pandas.core.arraylike.OpsMixin is private, but the related public API
# "ExtensionScalarOpsMixin" is not sufficient for adding dates to times.
# It results in unsupported operand type(s) for +: 'datetime.time' and
# 'datetime.date'
@import_default("pandas.core.arraylike")
class OpsMixin:
    def _cmp_method(self, other, op):  # pragma: NO COVER
        return NotImplemented

    def __eq__(self, other):
        return self._cmp_method(other, operator.eq)

    def __ne__(self, other):
        return self._cmp_method(other, operator.ne)

    def __lt__(self, other):
        return self._cmp_method(other, operator.lt)

    def __le__(self, other):
        return self._cmp_method(other, operator.le)

    def __gt__(self, other):
        return self._cmp_method(other, operator.gt)

    def __ge__(self, other):
        return self._cmp_method(other, operator.ge)

    __add__ = __radd__ = __sub__ = lambda self, other: NotImplemented


# TODO: use public API once pandas 1.5 / 2.x is released.
# See: https://github.com/pandas-dev/pandas/pull/45544
@import_default("pandas.core.arrays._mixins", pandas_release < (1, 3))
class NDArrayBackedExtensionArray(pandas.core.arrays.base.ExtensionArray):
    def __init__(self, values, dtype):
        assert isinstance(values, np.ndarray)
        self._ndarray = values
        self._dtype = dtype

    @classmethod
    def _from_backing_data(cls, data):
        return cls(data, data.dtype)

    def __getitem__(self, index):
        value = self._ndarray[index]
        if is_integer(index):
            return self._box_func(value)
        return self.__class__(value, self._dtype)

    def __setitem__(self, index, value):
        self._ndarray[index] = self._validate_setitem_value(value)

    def __len__(self):
        return len(self._ndarray)

    @property
    def shape(self):
        return self._ndarray.shape

    @property
    def ndim(self) -> int:
        return self._ndarray.ndim

    @property
    def size(self) -> int:
        return self._ndarray.size

    @property
    def nbytes(self) -> int:
        return self._ndarray.nbytes

    def copy(self):
        return self[:]

    def repeat(self, n):
        return self.__class__(self._ndarray.repeat(n), self._dtype)

    def take(
        self,
        indices,
        *,
        allow_fill: bool = False,
        fill_value: typing.Any = None,
        axis: int = 0,
    ):
        from pandas.core.algorithms import take

        if allow_fill:
            fill_value = self._validate_scalar(fill_value)

        new_data = take(
            self._ndarray,
            indices,
            allow_fill=allow_fill,
            fill_value=fill_value,
            axis=axis,
        )
        return self._from_backing_data(new_data)

    @classmethod
    def _concat_same_type(cls, to_concat, axis=0):
        dtypes = {str(x.dtype) for x in to_concat}
        if len(dtypes) != 1:
            raise ValueError("to_concat must have the same dtype (tz)", dtypes)

        new_values = [x._ndarray for x in to_concat]
        new_values = np.concatenate(new_values, axis=axis)
        return to_concat[0]._from_backing_data(new_values)  # type: ignore[arg-type]


class ArrowExtensionArray(pandas.core.arrays.base.ExtensionArray):
    """
    Pandas ExtensionArray backed by a PyArrow ChunkedArray.
    """
    _pa_array: pa.ChunkedArray
    _dtype: ArrowDtype

    def __init__(self, values: pa.Array | pa.ChunkedArray) -> None:
        if isinstance(values, pa.Array):
            self._pa_array = pa.chunked_array([values])
        elif isinstance(values, pa.ChunkedArray):
            self._pa_array = values
        else:
            raise ValueError(
                f"Unsupported type '{type(values)}' for ArrowExtensionArray"
            )
    
    @classmethod
    def _box_pa_scalar(cls, value, pa_type: pa.DataType | None = None) -> pa.Scalar:
        """Box value into a pyarrow Scalar."""
        if isinstance(value, pa.Scalar):
            pa_scalar = value
        elif pd.isna(value):
            pa_scalar = pa.scalar(None, type=pa_type)
        else:
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
                pa_array = pa.array(value, type=pa_type, from_pandas=True)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                # GH50430: let pyarrow infer type, then cast
                pa_array = pa.array(value, from_pandas=True)
        
        if pa_type is not None and pa_array.type != pa_type:
            try:
                pa_array = pa_array.cast(pa_type)
            except (
                pa.ArrowInvalid,
                pa.ArrowTypeError,
                pa.ArrowNotImplementedError,
            ):
                if pa.types.is_string(pa_array.type) or pa.types.is_large_string(
                    pa_array.type
                ):
                    # TODO: Move logic in _from_sequence_of_strings into
                    # _box_pa_array
                    return cls._from_sequence_of_strings(
                        value, dtype=pa_type
                    )._pa_array
                else:
                    raise

    def __len__(self) -> int:
        """Length of this array."""
        return len(self._pa_array)

    def __setitem__(self, key, value) -> None:
        """Set one or more values inplace."""
        # GH50085: unwrap 1D indexers
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]

        key = indexers.check_array_indexer(self, key)
        value = self._maybe_convert_setitem_value(value)

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
    
    def _maybe_convert_setitem_value(self, value):
        """Maybe convert value to be pyarrow compatible."""
        try:
            value = self._box_pa(value, self._pa_array.type)
        except pa.ArrowTypeError as err:
            msg = f"Invalid value '{str(value)}' for dtype {self.dtype}"
            raise TypeError(msg) from err
        return value

    def copy(self):
        """
        Return a shallow copy of the array.

        Underlying ChunkedArray is immutable, so a deep copy is unnecessary.
        """
        return type(self)(self._pa_array)

    def isna(self):
        """
        Boolean NumPy array indicating if each value is missing.

        This should return a 1-D array the same length as 'self'.
        """
        # GH51630: fast paths
        null_count = self._pa_array.null_count
        if null_count == 0:
            return np.zeros(len(self), dtype=np.bool_)
        elif null_count == len(self):
            return np.ones(len(self), dtype=np.bool_)

        return self._pa_array.is_null().to_numpy()

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

    def take(
        self,
        indices,
        allow_fill: bool = False,
        fill_value: typing.Any = None,
    ):
        """
        Take elements from an array.
        """
        indices_array = np.asanyarray(indices)

        if len(self._pa_array) == 0 and (indices_array >= 0).any():
            raise IndexError("cannot do a non-empty take")
        if indices_array.size > 0 and indices_array.max() >= len(self._pa_array):
            raise IndexError("out of bounds value in 'indices'.")

        if allow_fill:
            fill_mask = indices_array < 0
            if fill_mask.any():
                indexers.validate_indices(indices_array, len(self._pa_array))
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
                # return type(self)(pc.fill_null(result, pa.scalar(fill_value)))
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

    def _cmp_method(self, other, op):  # pragma: NO COVER
        return NotImplemented

    def __eq__(self, other):
        return self._cmp_method(other, operator.eq)

    def __ne__(self, other):
        return self._cmp_method(other, operator.ne)

    def __lt__(self, other):
        return self._cmp_method(other, operator.lt)

    def __le__(self, other):
        return self._cmp_method(other, operator.le)

    def __gt__(self, other):
        return self._cmp_method(other, operator.gt)

    def __ge__(self, other):
        return self._cmp_method(other, operator.ge)

    __add__ = __radd__ = __sub__ = lambda self, other: NotImplemented