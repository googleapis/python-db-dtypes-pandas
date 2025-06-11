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

import pandas
import pandas._testing as tm
from pandas.tests.extension import base
import pytest

import db_dtypes

# TODO(https://github.com/googleapis/python-db-dtypes-pandas/issues/87): Add
# compliance tests for arithmetic operations.

# TODO(https://github.com/googleapis/python-db-dtypes-pandas/issues/78): Add
# compliance tests for reduction operations.


# NDArrayBacked2DTests suite added in https://github.com/pandas-dev/pandas/pull/44974
class Test2DCompat(base.NDArrayBacked2DTests):
    pass


class TestComparisonOps(base.BaseComparisonOpsTests):
    pass


class TestCasting(base.BaseCastingTests):
    pass


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestDtype(base.BaseDtypeTests):
    pass


class TestGetitem(base.BaseGetitemTests):
    def test_take_pandas_style_negative_raises(self, data, na_value):
        # This test was failing compliance checks because it attempted to match
        # a pytest regex match using an empty string (""), which pytest version
        # 8.4.0 stopped allowing.
        # The test has been updated in pandas main so that it will
        # no longer fail, but the fix is not expected to be released until
        # at least pandas version 3.0 (current version is 2.3).
        with pytest.raises(ValueError):
            data.take([0, -2], fill_value=na_value, allow_fill=True)


class TestGroupby(base.BaseGroupbyTests):
    pass


class TestIndex(base.BaseIndexTests):
    pass


class TestInterface(base.BaseInterfaceTests):
    def test_array_interface_copy(self, data):
        import numpy as np
        import warnings
        from pandas.compat.numpy import np_version_gt2

        result_copy1 = np.array(data, copy=True)
        result_copy2 = np.array(data, copy=True)
        assert not np.may_share_memory(result_copy1, result_copy2)

        if not np_version_gt2:
            # copy=False semantics are only supported in NumPy>=2.
            return

        with pytest.raises(ValueError):
            result_nocopy1 = np.array(data, copy=False)
            result_nocopy2 = np.array(data, copy=False)
            assert np.may_share_memory(result_nocopy1, result_nocopy2)


class TestMissing(base.BaseMissingTests):
    pass


class TestMethods(base.BaseMethodsTests):
    def test_combine_add(self):
        pytest.skip("Cannot add dates.")

    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna):
        all_data = all_data[:10]
        if dropna:
            # Overridden from
            # https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/base/methods.py
            # to avoid difference in dtypes.
            other = db_dtypes.TimeArray(all_data[~all_data.isna()])
        else:
            other = all_data

        result = pandas.Series(all_data).value_counts(dropna=dropna).sort_index()
        expected = pandas.Series(other).value_counts(dropna=dropna).sort_index()

        tm.assert_series_equal(result, expected)

    def test_argmax_argmin_no_skipna_notimplemented(self, data_missing_for_sorting):
        # This test was failing compliance checks because it attempted to match
        # a pytest regex match using an empty string (""), which pytest version
        # 8.4.0 stopped allowing.
        # The test has been updated in pandas main so that it will
        # no longer fail, but the fix is not expected to be released until
        # at least pandas version 3.0 (current version is 2.3)
        data = data_missing_for_sorting

        with pytest.raises(NotImplementedError):
            data.argmin(skipna=False)

        with pytest.raises(NotImplementedError):
            data.argmax(skipna=False)


class TestParsing(base.BaseParsingTests):
    pass


class TestPrinting(base.BasePrintingTests):
    pass


class TestReshaping(base.BaseReshapingTests):
    pass


class TestSetitem(base.BaseSetitemTests):
    def test_setitem_invalid(self, data, invalid_scalar):
        # This test was failing compliance checks because it attempted to match
        # a pytest regex match using an empty string (""), which pytest version
        # 8.4.0 stopped allowing.
        # The test has been updated in pandas main so that it will
        # no longer fail, but the fix is not expected to be released until
        # at least pandas version 3.0 (current version is 2.3)
        with pytest.raises((ValueError, TypeError)):
            data[0] = invalid_scalar

        with pytest.raises((ValueError, TypeError)):
            data[:] = invalid_scalar
