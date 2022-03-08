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

import datetime

from pandas.tests.extension import base

import db_dtypes


class TestDtype(base.BaseDtypeTests):
    pass


class TestInterface(base.BaseInterfaceTests):
    pass


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestReshaping(base.BaseReshapingTests):
    pass


class TestGetitem(base.BaseGetitemTests):
    def test_take_na_value_other_date(self):
        arr = db_dtypes.DateArray(
            [datetime.date(2022, 3, 8), datetime.date(2022, 3, 9)]
        )
        result = arr.take(
            [0, -1], allow_fill=True, fill_value=datetime.date(1969, 12, 31)
        )
        expected = db_dtypes.DateArray(
            [datetime.date(2022, 3, 8), datetime.date(1969, 12, 31)]
        )
        self.assert_extension_array_equal(result, expected)


class TestMissing(base.BaseMissingTests):
    pass


class TestMethods(base.BaseMethodsTests):
    pass
