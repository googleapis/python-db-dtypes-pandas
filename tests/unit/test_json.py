# Copyright 2024 Google LLC
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

import json
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import db_dtypes
import db_dtypes.json


# Check for minimum Pandas version.
pytest.importorskip("pandas", minversion="1.5.0")


# Python data types mirroring all standard JSON types:
# https://json-schema.org/understanding-json-schema/reference/type
JSON_DATA = {
    "boolean": True,
    "int": 100,
    "float": 0.98,
    "string": "hello world",
    "array": [0.1, 0.2],
    "dict": {
        "null_field": None,
        "order": {
            "items": ["book", "pen", "computer"],
            "total": 15,
            "address": {"street": "123 Main St", "city": "Anytown"},
        },
    },
    "null": None,
}


def test_construct_w_unspported_types():
    with pytest.raises(ValueError):
        db_dtypes.JSONArray(100)


def test_getitems_return_json_objects():
    data = db_dtypes.JSONArray._from_sequence(JSON_DATA.values())
    for id, key in enumerate(JSON_DATA.keys()):
        if key == "null":
            assert pd.isna(data[id])
        else:
            assert data[id] == JSON_DATA[key]


def test_getitems_w_unboxed_dict():
    data = db_dtypes.JSONArray._from_sequence([JSON_DATA["dict"]])
    assert len(data[0]) == 2

    assert data[0]["null_field"] is None
    assert data[0]["order"]["address"]["city"] == "Anytown"
    assert len(data[0]["order"]["items"]) == 3
    assert data[0]["order"]["items"][0] == "book"

    with pytest.raises(KeyError):
        data[0]["unknown"]


def test_getitems_when_iter_with_null():
    data = db_dtypes.JSONArray._from_sequence([JSON_DATA["null"]])
    s = pd.Series(data)
    result = s[:1].item()
    assert pd.isna(result)


def test_deterministic_json_serialization():
    x = {"a": 0, "b": 1}
    y = {"b": 1, "a": 0}
    data = db_dtypes.JSONArray._from_sequence([y])
    assert data[0] == x


def test_to_numpy():
    """
    Verifies that JSONArray can be cast to a NumPy array.
    This test ensures compatibility with Python 3.9 and replicates the behavior
    of the `test_to_numpy` test from `test_json_compliance.py::TestJSONArrayCasting`,
    which is run with Python 3.12 environments only.
    """
    data = db_dtypes.JSONArray._from_sequence(JSON_DATA.values())
    expected = np.asarray(data)

    result = data.to_numpy()
    pd._testing.assert_equal(result, expected)

    result = pd.Series(data).to_numpy()
    pd._testing.assert_equal(result, expected)


def test_as_numpy_array():
    data = db_dtypes.JSONArray._from_sequence(JSON_DATA.values())
    result = np.asarray(data)
    expected = np.asarray(
        [
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            if value is not None
            else pd.NA
            for value in JSON_DATA.values()
        ]
    )
    pd._testing.assert_equal(result, expected)


def test_json_arrow_array():
    data = db_dtypes.JSONArray._from_sequence(JSON_DATA.values())
    assert isinstance(data.__arrow_array__(), pa.ExtensionArray)


def test_json_arrow_storage_type():
    arrow_json_type = db_dtypes.JSONArrowType()
    assert arrow_json_type.extension_name == "dbjson"
    assert pa.types.is_string(arrow_json_type.storage_type)


def test_json_arrow_hash():
    arr = pa.array([], type=db_dtypes.JSONArrowType())
    assert hash(arr.type) == hash(db_dtypes.JSONArrowType())


def test_json_arrow_constructors():
    data = [
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in JSON_DATA.values()
    ]
    storage_array = pa.array(data, type=pa.string())

    arr_1 = db_dtypes.JSONArrowType().wrap_array(storage_array)
    assert isinstance(arr_1, pa.ExtensionArray)

    arr_2 = pa.ExtensionArray.from_storage(db_dtypes.JSONArrowType(), storage_array)
    assert isinstance(arr_2, pa.ExtensionArray)

    assert arr_1 == arr_2


def test_json_arrow_to_pandas():
    data = [
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in JSON_DATA.values()
    ]
    arr = pa.array(data, type=db_dtypes.JSONArrowType())

    s = arr.to_pandas()
    assert isinstance(s.dtypes, db_dtypes.JSONDtype)
    assert s[0]
    assert s[1] == "100"
    assert s[2] == "0.98"
    assert s[3] == '"hello world"'
    assert s[4] == "[0.1,0.2]"
    assert (
        s[5]
        == '{"null_field":null,"order":{"address":{"city":"Anytown","street":"123 Main St"},"items":["book","pen","computer"],"total":15}}'
    )
    assert s[6] == "null"


def test_json_arrow_to_pylist():
    data = [
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in JSON_DATA.values()
    ]
    arr = pa.array(data, type=db_dtypes.JSONArrowType())

    s = arr.to_pylist()
    assert isinstance(s, list)
    assert s[0]
    assert s[1] == "100"
    assert s[2] == "0.98"
    assert s[3] == '"hello world"'
    assert s[4] == "[0.1,0.2]"
    assert (
        s[5]
        == '{"null_field":null,"order":{"address":{"city":"Anytown","street":"123 Main St"},"items":["book","pen","computer"],"total":15}}'
    )
    assert s[6] == "null"


def test_json_arrow_record_batch():
    data = [
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in JSON_DATA.values()
    ]
    arr = pa.array(data, type=db_dtypes.JSONArrowType())
    batch = pa.RecordBatch.from_arrays([arr], ["json_col"])
    sink = pa.BufferOutputStream()

    with pa.RecordBatchStreamWriter(sink, batch.schema) as writer:
        writer.write_batch(batch)

    buf = sink.getvalue()

    with pa.ipc.open_stream(buf) as reader:
        result = reader.read_all()

    json_col = result.column("json_col")
    assert isinstance(json_col.type, db_dtypes.JSONArrowType)

    s = json_col.to_pylist()

    assert isinstance(s, list)
    assert s[0]
    assert s[1] == "100"
    assert s[2] == "0.98"
    assert s[3] == '"hello world"'
    assert s[4] == "[0.1,0.2]"
    assert (
        s[5]
        == '{"null_field":null,"order":{"address":{"city":"Anytown","street":"123 Main St"},"items":["book","pen","computer"],"total":15}}'
    )
    assert s[6] == "null"


@pytest.fixture
def cleanup_json_module_for_reload():
    """
    Fixture to ensure db_dtypes.json is registered and then removed
    from sys.modules to allow testing the registration except block via reload.
    """

    json_module_name = "db_dtypes.json"
    original_module = sys.modules.get(json_module_name)

    # Ensure the type is registered initially (usually by the first import)
    try:
        # Make sure the module is loaded so the type exists
        import db_dtypes.json

        # Explicitly register just in case it wasn't, or was cleaned up elsewhere.
        # This might raise ArrowKeyError itself if already registered, which is fine here.
        pa.register_extension_type(db_dtypes.json.JSONArrowType())

    except pa.ArrowKeyError:
        pass  # Already registered is the state we want before the test runs

    # Remove the module from sys.modules so importlib.reload which will happen
    # after the yield statement will re-execute it
    if json_module_name in sys.modules:  # pragma: NO COVER
        del sys.modules[json_module_name]

    yield  # Run the test that uses this fixture

    # Cleanup: Put the original module back if it existed
    # This helps isolate from other tests that might import db_dtypes.json
    if original_module:
        sys.modules[json_module_name] = original_module
    elif json_module_name in sys.modules:  # pragma: NO COVER
        # If the test re-imported it but it wasn't there originally, remove it
        del sys.modules[json_module_name]

    # Note: PyArrow doesn't have a public API to unregister types easily,
    # thus we are using the testing pattern of module isolation/reloading.


# Test specifically for the fixture's pre-yield removal logic
def test_fixture_removes_module_if_present(cleanup_json_module_for_reload):
    """
    Tests that the cleanup_json_module_for_reload fixture removes
    db_dtypes.json from sys.modules before yielding to the test.
    This specifically targets the 'if json_module_name in sys.modules:' block.
    """
    # This test runs *after* the fixture's `yield`.
    # The fixture should have removed the module if it was present.

    json_module_name = "db_dtypes.json"

    assert (
        json_module_name not in sys.modules
    ), f"The fixture cleanup_json_module_for_reload should have removed {json_module_name}"


def test_json_arrow_type_reregistration_is_handled(cleanup_json_module_for_reload):
    """
    Verify that attempting to re-register JSONArrowType via module reload
    is caught by the except block and does not raise an error.
    """

    # Re-importing the module after the fixture removed it from sys.modules
    # forces Python to execute the module's top-level code again.
    # This includes the pa.register_extension_type call.

    import db_dtypes.json  # noqa: F401

    assert True, "Module re-import completed without error, except block likely worked."