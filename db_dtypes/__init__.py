# Copyright 2021 Google LLC
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
"""
Pandas Data Types for SQL systems (BigQuery, Spanner)
"""

import datetime
import re

import numpy
import packaging.version
import pandas
import pandas.compat.numpy.function
import pandas.core.algorithms
import pandas.core.arrays
import pandas.core.dtypes.base
import pandas.core.dtypes.dtypes
import pandas.core.dtypes.generic
import pandas.core.nanops
import pyarrow

from db_dtypes.version import __version__
from db_dtypes import core


date_dtype_name = "date"
time_dtype_name = "time"

pandas_release = packaging.version.parse(pandas.__version__).release


@pandas.core.dtypes.dtypes.register_extension_dtype
class TimeDtype(core.BaseDatetimeDtype):
    """
    Extension dtype for time data.
    """

    name = time_dtype_name
    type = datetime.time

    def construct_array_type(self):
        return TimeArray


class TimeArray(core.BaseDatetimeArray):
    """
    Pandas array type containing time data
    """

    # Data are stored as datetime64 values with a date of Jan 1, 1970

    dtype = TimeDtype()
    _epoch = datetime.datetime(1970, 1, 1)
    _npepoch = numpy.datetime64(_epoch)

    @classmethod
    def _datetime(
        cls,
        scalar,
        match_fn=re.compile(
            r"\s*(?P<hour>\d+)(?::(?P<minute>\d+)(?::(?P<second>\d+(?:[.]\d+)?)?)?)?\s*$"
        ).match,
    ):
        if isinstance(scalar, datetime.time):
            return datetime.datetime.combine(cls._epoch, scalar)
        elif isinstance(scalar, str):
            # iso string
            match = match_fn(scalar)
            if not match:
                raise ValueError(f"Bad time string: {repr(scalar)}")

            hour = match.group("hour")
            minute = match.group("minute")
            second = match.group("second")
            second, microsecond = divmod(float(second if second else 0), 1)
            return datetime.datetime(
                1970,
                1,
                1,
                int(hour),
                int(minute if minute else 0),
                int(second),
                int(microsecond * 1_000_000),
            )
        else:
            raise TypeError("Invalid value type", scalar)

    def _box_func(self, x):
        if pandas.isnull(x):
            return None

        try:
            return x.astype("<M8[us]").astype(datetime.datetime).time()
        except AttributeError:
            x = numpy.datetime64(x)
            return x.astype("<M8[us]").astype(datetime.datetime).time()

    __return_deltas = {"timedelta", "timedelta64", "timedelta64[ns]", "<m8", "<m8[ns]"}

    def astype(self, dtype, copy=True):
        deltas = self._ndarray - self._npepoch
        stype = str(dtype)
        if stype in self.__return_deltas:
            return deltas
        elif stype.startswith("timedelta64[") or stype.startswith("<m8["):
            return deltas.astype(dtype, copy=False)
        else:
            return super().astype(dtype, copy=copy)

    if pandas_release < (1,):

        def to_numpy(self, dtype="object"):
            return self.astype(dtype)

    def __arrow_array__(self, type=None):
        return pyarrow.array(
            self.to_numpy(dtype="object"),
            type=type if type is not None else pyarrow.time64("ns"),
        )


@pandas.core.dtypes.dtypes.register_extension_dtype
class DateDtype(core.BaseDatetimeDtype):
    """
    Extension dtype for time data.
    """

    name = date_dtype_name
    type = datetime.date

    def construct_array_type(self):
        return DateArray


class DateArray(core.BaseDatetimeArray):
    """
    Pandas array type containing date data
    """

    # Data are stored as datetime64 values with a date of Jan 1, 1970

    dtype = DateDtype()

    @staticmethod
    def _datetime(
        scalar,
        match_fn=re.compile(r"\s*(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+)\s*$").match,
    ):
        if isinstance(scalar, datetime.date):
            return datetime.datetime(scalar.year, scalar.month, scalar.day)
        elif isinstance(scalar, str):
            match = match_fn(scalar)
            if not match:
                raise ValueError(f"Bad date string: {repr(scalar)}")
            year = int(match.group("year"))
            month = int(match.group("month"))
            day = int(match.group("day"))
            return datetime.datetime(year, month, day)
        else:
            raise TypeError("Invalid value type", scalar)

    def _box_func(self, x):
        if pandas.isnull(x):
            return None
        try:
            return x.astype("<M8[us]").astype(datetime.datetime).date()
        except AttributeError:
            x = numpy.datetime64(x)
            return x.astype("<M8[us]").astype(datetime.datetime).date()

    def astype(self, dtype, copy=True):
        stype = str(dtype)
        if stype.startswith("datetime"):
            if stype == "datetime" or stype == "datetime64":
                dtype = self._ndarray.dtype
            return self._ndarray.astype(dtype, copy=copy)
        elif stype.startswith("<M8"):
            if stype == "<M8":
                dtype = self._ndarray.dtype
            return self._ndarray.astype(dtype, copy=copy)

        return super().astype(dtype, copy=copy)

    def __arrow_array__(self, type=None):
        return pyarrow.array(
            self._ndarray, type=type if type is not None else pyarrow.date32(),
        )

    def __add__(self, other):
        if isinstance(other, pandas.DateOffset):
            return self.astype("object") + other

        if isinstance(other, TimeArray):
            return (other._ndarray - other._npepoch) + self._ndarray

        return super().__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, pandas.DateOffset):
            return self.astype("object") - other

        if isinstance(other, self.__class__):
            return self._ndarray - other._ndarray

        return super().__sub__(other)


__all__ = [
    "__version__",
    "DateArray",
    "DateDtype",
    "TimeArray",
    "TimeDtype",
]
