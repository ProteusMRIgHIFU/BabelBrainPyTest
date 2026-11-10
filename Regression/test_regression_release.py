"""
Major release regression tests (BabelBrain_Test_Plan.md, Table 10).

Each regression test comes in two parts:

  * test_generate_<name>  runs the pipeline for every case and saves the outputs in
                          <gen_output_folder>/<suite>/<case_id> (marked complete only
                          once the run succeeded). Pytest runs all of these first.
  * test_<name>           compares the saved outputs with the reference outputs in
                          <ref_output_folder_1>/<suite>/<case_id> (they must match exactly).
                          It generates the case itself if needed, so it can also be
                          selected on its own.

Saved outputs from the same BabelBrain commit are reused across tests and sessions
(set BABEL_PYTEST_REGENERATE=1 to force a rerun). To create reference outputs, check
out the reference BabelBrain version (see Table 1 in BabelBrain_Test_Plan.md),
point gen_output_folder at the
reference location and run only the generate tests:

    pytest Tests/Regression -m generate_outputs
"""
import logging

import pytest

from Tests.helpers.parameters import REGRESSION_CASES
from Tests.helpers.pipeline_helpers import cases_from_spec, find_reference_dir

pytestmark = pytest.mark.major_release

# ================================================================================================================================
# CASES (parameters in parameters.REGRESSION_CASES)
# ================================================================================================================================
FULL_PIPELINE_SUITE = 'regression_full_pipeline'
FULL_PIPELINE_INTENSIVE_SUITE = 'regression_full_pipeline_intensive'
FULL_PIPELINE_SOFT_TISSUE_SUITE = 'regression_full_pipeline_soft_tissue'
SKULL_MASK_SUITE = 'regression_GetSkullMaskFromSimbNIBSSTL'


def regression_cases(name, marks):
    """pytest params for every combination of a REGRESSION_CASES entry."""
    return cases_from_spec(REGRESSION_CASES[name], marks)


FULL_PIPELINE_CASES = regression_cases('regression_test_full_pipeline', [pytest.mark.slow])
FULL_PIPELINE_INTENSIVE_CASES = regression_cases('regression_test_full_pipeline_intensive', [pytest.mark.slow])
FULL_PIPELINE_SOFT_TISSUE_CASES = regression_cases('regression_test_full_pipeline_soft_tissue', [pytest.mark.slow])
SKULL_MASK_CASES = regression_cases('regression_test_GetSkullMaskFromSimbNIBSSTL', [pytest.mark.step1])


# ================================================================================================================================
# GENERATE (runs first)
# ================================================================================================================================
@pytest.mark.generate_outputs
@pytest.mark.parametrize('case', FULL_PIPELINE_CASES)
def test_generate_regression_full_pipeline(pipeline, case):
    logging.info(f"Case {case.case_id}: saving outputs for suite {FULL_PIPELINE_SUITE}")
    pipeline.ensure_outputs(FULL_PIPELINE_SUITE, case)


@pytest.mark.generate_outputs
@pytest.mark.parametrize('case', FULL_PIPELINE_INTENSIVE_CASES)
def test_generate_regression_full_pipeline_intensive(pipeline, case):
    logging.info(f"Case {case.case_id}: saving outputs for suite {FULL_PIPELINE_INTENSIVE_SUITE}")
    pipeline.ensure_outputs(FULL_PIPELINE_INTENSIVE_SUITE, case)


@pytest.mark.generate_outputs
@pytest.mark.parametrize('case', FULL_PIPELINE_SOFT_TISSUE_CASES)
def test_generate_regression_full_pipeline_soft_tissue(pipeline, case):
    logging.info(f"Case {case.case_id}: saving outputs for suite {FULL_PIPELINE_SOFT_TISSUE_SUITE}")
    pipeline.ensure_outputs(FULL_PIPELINE_SOFT_TISSUE_SUITE, case)


@pytest.mark.generate_outputs
@pytest.mark.parametrize('case', SKULL_MASK_CASES)
def test_generate_regression_GetSkullMaskFromSimbNIBSSTL(pipeline, case):
    logging.info(f"Case {case.case_id}: saving outputs for suite {SKULL_MASK_SUITE}")
    pipeline.ensure_outputs(SKULL_MASK_SUITE, case)


# ================================================================================================================================
# COMPARE AGAINST REFERENCE
# ================================================================================================================================
@pytest.mark.parametrize('case', FULL_PIPELINE_CASES)
def test_regression_full_pipeline(pipeline, case):
    logging.info(f"Case {case.case_id}: comparing with the reference outputs of suite {FULL_PIPELINE_SUITE}")
    ref_dir = find_reference_dir(FULL_PIPELINE_SUITE, case)
    test_dir = pipeline.ensure_outputs(FULL_PIPELINE_SUITE, case)
    logging.info("Checking the thermal outputs match the reference exactly")
    pipeline.assert_thermal_outputs_match(ref_dir, test_dir)


@pytest.mark.parametrize('case', FULL_PIPELINE_INTENSIVE_CASES)
def test_regression_full_pipeline_intensive(pipeline, case):
    logging.info(f"Case {case.case_id}: comparing with the reference outputs of suite {FULL_PIPELINE_INTENSIVE_SUITE}")
    ref_dir = find_reference_dir(FULL_PIPELINE_INTENSIVE_SUITE, case)
    test_dir = pipeline.ensure_outputs(FULL_PIPELINE_INTENSIVE_SUITE, case)
    logging.info("Checking the thermal outputs match the reference exactly")
    pipeline.assert_thermal_outputs_match(ref_dir, test_dir)


@pytest.mark.parametrize('case', FULL_PIPELINE_SOFT_TISSUE_CASES)
def test_regression_full_pipeline_soft_tissue(pipeline, case):
    logging.info(f"Case {case.case_id}: comparing with the reference outputs of suite {FULL_PIPELINE_SOFT_TISSUE_SUITE}")
    ref_dir = find_reference_dir(FULL_PIPELINE_SOFT_TISSUE_SUITE, case)
    test_dir = pipeline.ensure_outputs(FULL_PIPELINE_SOFT_TISSUE_SUITE, case)
    logging.info("Checking the thermal outputs match the reference exactly")
    pipeline.assert_thermal_outputs_match(ref_dir, test_dir)


@pytest.mark.parametrize('case', SKULL_MASK_CASES)
def test_regression_GetSkullMaskFromSimbNIBSSTL(pipeline, case):
    logging.info(f"Case {case.case_id}: comparing with the reference outputs of suite {SKULL_MASK_SUITE}")
    ref_dir = find_reference_dir(SKULL_MASK_SUITE, case)
    test_dir = pipeline.ensure_outputs(SKULL_MASK_SUITE, case)
    logging.info("Checking every output file matches the reference exactly")
    pipeline.assert_all_outputs_match(ref_dir, test_dir)
