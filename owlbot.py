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

"""This script is used to synthesize generated parts of this library."""

import pathlib

import synthtool as s
from synthtool import gcp
from synthtool.languages import python

REPO_ROOT = pathlib.Path(__file__).parent.absolute()

common = gcp.CommonTemplates()

# ----------------------------------------------------------------------------
# Add templated files
# ----------------------------------------------------------------------------
templated_files = common.py_library(
    unit_test_python_versions=["3.6", "3.7", "3.8", "3.9", "3.10"],
    system_test_python_versions=["3.8"],
    cov_level=100,
    intersphinx_dependencies={
        "pandas": "https://pandas.pydata.org/pandas-docs/stable/"
    },
)
s.move(templated_files, excludes=["docs/multiprocessing.rst"])

# ----------------------------------------------------------------------------
# Fixup files
# ----------------------------------------------------------------------------

s.replace(
    [".coveragerc"], "google/cloud/__init__.py", "db_dtypes/requirements.py",
)

s.replace(
    ["noxfile.py"], r"[\"']google[\"']", '"db_dtypes"',
)

s.replace(
    ["noxfile.py"], "--cov=google", "--cov=db_dtypes",
)

# There are no system tests for this package.
old_sessions = """
    "unit",
    "system",
    "cover",
    "lint",
"""

new_sessions = """
    "lint",
    "unit",
    "compliance",
    "cover",
"""

s.replace(["noxfile.py"], old_sessions, new_sessions)

# Add compliance tests.
s.replace(
    ["noxfile.py"], r"def default\(session\):", "def default(session, tests_path):"
)
s.replace(["noxfile.py"], r'os.path.join\("tests", "unit"\),', "tests_path,")
s.replace(
    ["noxfile.py"],
    r"""
@nox.session\(python=UNIT_TEST_PYTHON_VERSIONS\)
def unit\(session\):
""",
    '''
@nox.session(python=UNIT_TEST_PYTHON_VERSIONS[-1])
def compliance(session):
    """Run the compliance test suite."""
    default(session, os.path.join("tests", "compliance"))


@nox.session(python=UNIT_TEST_PYTHON_VERSIONS)
def unit(session):
''',
)


old_unittest = """
  cover:
    runs-on: ubuntu-latest
"""

new_unittest = """
  compliance:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ['3.10']
    steps:
    - name: Checkout
      uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python }}
    - name: Install nox
      run: |
        python -m pip install --upgrade setuptools pip wheel
        python -m pip install nox
    - name: Run compliance tests
      env:
        COVERAGE_FILE: .coverage-compliance-${{ matrix.python }}
      run: |
        nox -s compliance
    - name: Upload coverage results
      uses: actions/upload-artifact@v3
      with:
        name: coverage-artifacts
        path: .coverage-compliance-${{ matrix.python }}

  cover:
    runs-on: ubuntu-latest
"""

s.replace([".github/workflows/unittest.yml"], old_unittest, new_unittest)

# ----------------------------------------------------------------------------
# Samples templates
# ----------------------------------------------------------------------------

python.py_samples(skip_readmes=True)

# ----------------------------------------------------------------------------
# Final cleanup
# ----------------------------------------------------------------------------

s.shell.run(["nox", "-s", "blacken"], hide_output=False)
for noxfile in REPO_ROOT.glob("samples/**/noxfile.py"):
    s.shell.run(["nox", "-s", "blacken"], cwd=noxfile.parent, hide_output=False)
