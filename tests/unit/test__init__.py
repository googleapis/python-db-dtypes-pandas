import sys
import pytest
import warnings
from unittest import mock

# The module where the version check code resides
MODULE_PATH = "db_dtypes"
HELPER_MODULE_PATH = f"{MODULE_PATH}._versions_helpers"

@pytest.fixture(autouse=True)
def cleanup_imports():
    """
    Ensures the target module and its helper are removed from sys.modules
    before and after each test, allowing for clean imports with patching.
    """
    # Store original sys.version_info if it's not already stored
    if not hasattr(cleanup_imports, 'original_version_info'):
        cleanup_imports.original_version_info = sys.version_info

    # Remove modules before test
    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]
    if HELPER_MODULE_PATH in sys.modules:
         del sys.modules[HELPER_MODULE_PATH]

    yield # Run the test

    # Restore original sys.version_info after test
    sys.version_info = cleanup_imports.original_version_info

    # Remove modules after test
    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]
    if HELPER_MODULE_PATH in sys.modules:
         del sys.modules[HELPER_MODULE_PATH]


@pytest.mark.parametrize(
    "mock_version_tuple, version_str",
    [
        ((3, 7, 10), "3.7.10"),
        ((3, 7, 0), "3.7.0"),
        ((3, 8, 5), "3.8.5"),
        ((3, 8, 12), "3.8.12"),
    ]
)
def test_python_3_7_or_3_8_warning_on_import(mock_version_tuple, version_str):
    """Test that a FutureWarning is raised for Python 3.7 during import."""
    # Create a mock object mimicking sys.version_info attributes
    # Use spec=sys.version_info to ensure it has the right attributes if needed,
    # though just setting major/minor/micro is usually sufficient here.
    mock_version_info = mock.Mock(spec=sys.version_info,
                                  major=mock_version_tuple[0],
                                  minor=mock_version_tuple[1],
                                  micro=mock_version_tuple[2])

    # Patch sys.version_info *before* importing db_dtypes
    with mock.patch('sys.version_info', mock_version_info):
        # Use pytest.warns to catch the expected warning during import
        with pytest.warns(FutureWarning) as record:
            # This import triggers __init__.py, which calls
            # _versions_helpers.extract_runtime_version, which reads
            # the *mocked* sys.version_info
            import db_dtypes

        # Assert that exactly one warning was recorded
        assert len(record) == 1
        warning_message = str(record[0].message)
        # Assert the warning message content is correct
        assert "longer supports Python 3.7 and Python 3.8" in warning_message

@pytest.mark.parametrize(
    "mock_version_tuple",
    [
        (3, 9, 1),    # Supported
        (3, 10, 0),   # Supported
        (3, 11, 2),   # Supported
        (3, 12, 0),   # Supported
    ]
)
def test_no_warning_for_other_versions_on_import(mock_version_tuple):
    """Test that no FutureWarning is raised for other Python versions during import."""
    with mock.patch(f"{MODULE_PATH}._versions_helpers.extract_runtime_version", return_value=mock_version_tuple):
        # Use warnings.catch_warnings to check that NO relevant warning is raised
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always") # Ensure warnings aren't filtered out by default config
            import db_dtypes # Import triggers the code

        # Assert that no FutureWarning matching the specific message was recorded
        found_warning = False
        for w in record:
            # Check for the specific warning we want to ensure is NOT present
            if (issubclass(w.category, FutureWarning) and
                    "longer supports Python 3.7 and Python 3.8" in str(w.message)):
                found_warning = True
                break
        assert not found_warning, f"Unexpected FutureWarning raised for Python version {mock_version_tuple}"


@pytest.fixture
def cleanup_imports_for_all(request):
    """
    Ensures the target module and its dependencies potentially affecting
    __all__ are removed from sys.modules before and after each test,
    allowing for clean imports with patching.
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


# --- Test Case 1: JSON types available ---

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


# --- Test Case 2: JSON types unavailable ---

@pytest.mark.parametrize(
    "patch_target_name",
    [
        "JSONArray",
        "JSONDtype",
        # Add both if needed, though one is sufficient to trigger the 'if'
        # ("JSONArray", "JSONDtype"),
    ]
)
def test_all_excludes_json_when_unavailable(cleanup_imports_for_all, patch_target_name):
    """
    Test that __all__ excludes JSON types when JSONArray or JSONDtype is unavailable (falsy).
    """
    patch_path = f"{MODULE_PATH}.{patch_target_name}"

    # Patch one of the JSON types to be None *before* importing db_dtypes.
    # This simulates the condition `if not JSONArray or not JSONDtype:` being true.
    with mock.patch(patch_path, None):
        # Need to ensure the json submodule itself is loaded if patching its contents
        # If the patch target is directly in __init__, this isn't needed.
        # Assuming JSONArray/JSONDtype are imported *into* __init__ from .json:
        try:
            import db_dtypes.json
        except ImportError:
             # Handle cases where the json module might genuinely be missing
             pass

        # Now import the main module, which will evaluate __all__
        import db_dtypes

        expected_all = [
            "__version__",
            "DateArray",
            "DateDtype",
            "TimeArray",
            "TimeDtype",
        ]
        # Use set comparison for order independence
        assert set(db_dtypes.__all__) == set(expected_all)
        # Explicitly check absence of JSON types
        assert "JSONDtype" not in db_dtypes.__all__
        assert "JSONArray" not in db_dtypes.__all__
        assert "JSONArrowType" not in db_dtypes.__all__