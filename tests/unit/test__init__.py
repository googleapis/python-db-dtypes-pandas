import sys
import pytest
import types
import warnings
from unittest import mock
import pyarrow as pa

# Module paths used for mocking
MODULE_PATH = "db_dtypes"
HELPER_MODULE_PATH = f"{MODULE_PATH}._versions_helpers"
MOCK_EXTRACT_VERSION = f"{HELPER_MODULE_PATH}.extract_runtime_version"
MOCK_WARN = "warnings.warn" # Target the standard warnings module

@pytest.mark.parametrize(
    "mock_version_tuple, version_str",
    [
        ((3, 7, 10), "3.7.10"),
        ((3, 7, 0), "3.7.0"),
        ((3, 8, 5), "3.8.5"),
        ((3, 8, 12), "3.8.12"),
    ]
)
def test_check_python_version_warns_on_unsupported(mock_version_tuple, version_str):
    """
    Test that _check_python_version issues a FutureWarning for Python 3.7/3.8.
    """
    # Import the function under test directly
    from db_dtypes import _check_python_version

    # Mock the helper function it calls and the warnings.warn function
    with mock.patch(MOCK_EXTRACT_VERSION, return_value=mock_version_tuple), \
         mock.patch(MOCK_WARN) as mock_warn_call:

        _check_python_version() # Call the function

        # Assert that warnings.warn was called exactly once
        mock_warn_call.assert_called_once()

        # Check the arguments passed to warnings.warn
        args, kwargs = mock_warn_call.call_args
        assert len(args) >= 1 # Should have at least the message
        warning_message = args[0]
        warning_category = args[1] if len(args) > 1 else kwargs.get('category')

        # Verify message content and category
        assert "longer supports Python 3.7 and Python 3.8" in warning_message
        assert f"Your Python version is {version_str}" in warning_message
        assert "https://cloud.google.com/python/docs/supported-python-versions" in warning_message
        assert warning_category == FutureWarning
        # Optionally check stacklevel if important
        assert kwargs.get('stacklevel') == 2


@pytest.mark.parametrize(
    "mock_version_tuple",
    [
        (3, 9, 1),
        (3, 10, 0),
        (3, 11, 2),
        (3, 12, 0),
        (4, 0, 0), # Future version
        (3, 6, 0), # Older unsupported, but not 3.7/3.8
    ]
)
def test_check_python_version_does_not_warn_on_supported(mock_version_tuple):
    """
    Test that _check_python_version does NOT issue a warning for other versions.
    """
    # Import the function under test directly
    from db_dtypes import _check_python_version

    # Mock the helper function it calls and the warnings.warn function
    with mock.patch(MOCK_EXTRACT_VERSION, return_value=mock_version_tuple), \
         mock.patch(MOCK_WARN) as mock_warn_call:

        _check_python_version() # Call the function

        # Assert that warnings.warn was NOT called
        mock_warn_call.assert_not_called()


def test_determine_all_includes_json_when_available():
    """
    Test that _determine_all includes JSON types when both are truthy.
    """
    # Import the function directly for testing
    from db_dtypes import _determine_all

    # Simulate available types (can be any truthy object)
    mock_json_array = object()
    mock_json_dtype = object()

    result = _determine_all(mock_json_array, mock_json_dtype)

    expected_all = [
        "__version__",
        "DateArray",
        "DateDtype",
        "TimeArray",
        "TimeDtype",
        "JSONDtype",
        "JSONArray",
        "JSONArrowType",
    ]
    assert set(result) == set(expected_all)
    assert "JSONDtype" in result
    assert "JSONArray" in result
    assert "JSONArrowType" in result

@pytest.mark.parametrize(
    "mock_array, mock_dtype",
    [
        (None, object()), # JSONArray is None
        (object(), None), # JSONDtype is None
        (None, None),     # Both are None
    ]
)
def test_determine_all_excludes_json_when_unavailable(mock_array, mock_dtype):
    """
    Test that _determine_all excludes JSON types if either is falsy.
    """
    # Import the function directly for testing
    from db_dtypes import _determine_all

    result = _determine_all(mock_array, mock_dtype)

    expected_all = [
        "__version__",
        "DateArray",
        "DateDtype",
        "TimeArray",
        "TimeDtype",
    ]
    assert set(result) == set(expected_all)
    assert "JSONDtype" not in result
    assert "JSONArray" not in result
    assert "JSONArrowType" not in result