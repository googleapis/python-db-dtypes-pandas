import sys
import pytest
import types
import warnings
from unittest import mock
import pyarrow as pa

# The module where the version check code resides
MODULE_PATH = "db_dtypes"
HELPER_MODULE_PATH = f"{MODULE_PATH}._versions_helpers"

@pytest.fixture
def cleanup_imports():
    """Ensures the target module and its helper are removed from sys.modules
    before each test, allowing for clean imports with patching.
    """

    # Store original modules that might exist
    original_modules = {}
    modules_to_clear = [MODULE_PATH, HELPER_MODULE_PATH]
    for mod_name in modules_to_clear:
        if mod_name in sys.modules:
            original_modules[mod_name] = sys.modules[mod_name]
            del sys.modules[mod_name]

    yield # Run the test

    # Clean up again and restore originals if they existed
    for mod_name in modules_to_clear:
        if mod_name in sys.modules:
            del sys.modules[mod_name] # Remove if test imported it
    # Restore original modules
    for mod_name, original_mod in original_modules.items():
        if original_mod:
            sys.modules[mod_name] = original_mod

@pytest.mark.parametrize(
    "mock_version_tuple, version_str, expect_warning",
    [
        # Cases expected to warn
        ((3, 7, 10), "3.7.10", True),
        ((3, 7, 0), "3.7.0", True),
        ((3, 8, 5), "3.8.5", True),
        ((3, 8, 12), "3.8.12", True),
        # Cases NOT expected to warn
        ((3, 9, 1), "3.9.1", False),
        ((3, 10, 0), "3.10.0", False),
        ((3, 11, 2), "3.11.2", False),
        ((3, 12, 0), "3.12.0", False),
    ]
)
def test_python_version_warning_on_import(mock_version_tuple, version_str, expect_warning, cleanup_imports):
    """Test that a FutureWarning is raised ONLY for Python 3.7 or 3.8 during import.
    """
    
    # Create a mock function that returns the desired version tuple
    mock_extract_func = mock.Mock(return_value=mock_version_tuple)

    # Create a mock module object for _versions_helpers
    mock_helpers_module = types.ModuleType(HELPER_MODULE_PATH)
    mock_helpers_module.extract_runtime_version = mock_extract_func

    # Use mock.patch.dict to temporarily replace the module in sys.modules
    # This ensures that when db_dtypes.__init__ does `from . import _versions_helpers`,
    # it gets our mock module.
    with mock.patch.dict(sys.modules, {HELPER_MODULE_PATH: mock_helpers_module}):
        if expect_warning:
            with pytest.warns(FutureWarning) as record:
                # The import will now use the mocked _versions_helpers module
                import db_dtypes

            assert len(record) == 1
            warning_message = str(record[0].message)
            assert "longer supports Python 3.7 and Python 3.8" in warning_message
        else:
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                # The import will now use the mocked _versions_helpers module
                import db_dtypes

            found_warning = False
            for w in record:
                if (issubclass(w.category, FutureWarning) and
                        "longer supports Python 3.7 and Python 3.8" in str(w.message)):
                    found_warning = True
                    break
            assert not found_warning, (
                f"Unexpected FutureWarning raised for Python version {version_str}"
            )

# --- Test Case 1: JSON types available ---

@pytest.fixture
def cleanup_imports_for_all(request):
    """
    Ensures the target module and its dependencies potentially affecting
    __all__ are removed from sys.modules before and after each test,
    allowing for clean imports with patching. Also handles PyArrow extension type registration.
    """

    # Modules that might be checked or imported in __init__
    modules_to_clear = [
        MODULE_PATH,
        f"{MODULE_PATH}.core",
        f"{MODULE_PATH}.json",
        f"{MODULE_PATH}.version",
        f"{MODULE_PATH}._versions_helpers",
    ]
    original_modules = {}

    # Store original modules and remove them
    for mod_name in modules_to_clear:
        original_modules[mod_name] = sys.modules.get(mod_name)
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    yield # Run the test

    # Restore original modules after test
    for mod_name, original_mod in original_modules.items():
        if original_mod:
            sys.modules[mod_name] = original_mod
        elif mod_name in sys.modules:
             # If it wasn't there before but is now, remove it
             del sys.modules[mod_name]

def test_all_includes_json_when_available(cleanup_imports_for_all):
    """
    Test that __all__ includes JSON types when JSONArray and JSONDtype are available.
    """

    # No patching needed for the 'else' block, assume normal import works
    # and JSONArray/JSONDtype are truthy.
    import db_dtypes

    expected_all = [
        "__version__",
        "DateArray",
        "DateDtype",
        "JSONDtype",
        "JSONArray",
        "JSONArrowType",
        "TimeArray",
        "TimeDtype",
    ]
    # Use set comparison for order independence, as __all__ order isn't critical
    assert set(db_dtypes.__all__) == set(expected_all)
    # Explicitly check presence of JSON types
    assert "JSONDtype" in db_dtypes.__all__
    assert "JSONArray" in db_dtypes.__all__
    assert "JSONArrowType" in db_dtypes.__all__
