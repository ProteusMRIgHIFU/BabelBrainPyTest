# BabelBrain Pytest Guide

How to set up, run and extend the BabelBrain tests. There are two documents:

- **README.md** (this file): setup, layout, and how to run each kind of test.
- **[BabelBrain_Pytest_Usage.md](BabelBrain_Pytest_Usage.md)**: the release test plan, i.e. which tests and parameter combinations minor and major releases require (Tables 1 to 10).

Run every command below from the top level BabelBrain folder, and pass `Tests` (or a path under it) to pytest, e.g. `pytest Tests -m "gpu"`. Pytest then uses [Tests/pytest.ini](pytest.ini) (command line options, live logging, the html report, the required plugins and the markers). A bare `pytest` with no path only finds a `pytest.ini` in the top level folder, which isn't the test suite's.

**Contents**
- [Setup](#setup)
- [Test suite layout](#test-suite-layout)
- [Running tests](#running-tests)
- [Unit tests](#unit-tests)
- [End-to-end tests](#end-to-end-tests)
- [Generating outputs](#generating-outputs)
- [Release testing](#release-testing)
- [Isolating a pytest run (crash and memory triage)](#isolating-a-pytest-run-crash-and-memory-triage)
- [Writing tests](#writing-tests)

# Setup

## 1. BabelBrain environment

The tests import BabelBrain and run it in process, so they run in the same conda environment as BabelBrain. Create it from the environment file for your platform in the BabelBrain repo (e.g. `environment_mac_arm64-314.yml`, `environment_win-314-NoHash.yml`, `environment_linux.yml`) and activate it:

```bash
conda env create -f environment_mac_arm64-314.yml
conda activate BabelP314
```

Check that BabelBrain itself starts (`python BabelBrain/BabelBrain.py`) and that your GPU backend (CUDA, OpenCL or Metal) is available in its computing engine list before running GPU or full pipeline tests.

## 2. Test suite

The tests are a separate git repository, cloned into a folder named `Tests` in the BabelBrain checkout under test (BabelBrain's `.gitignore` ignores it):

```bash
cd <BabelBrain checkout>
git clone <BabelBrain pytest repository URL> Tests
```

To test another BabelBrain version, check that version out in the BabelBrain checkout; the tests read transducers, frequencies and options from whatever is checked out, and skip tests for features it doesn't have.

## 3. Packages

The tests need pytest and these plugins (they are listed under `required_plugins` in [pytest.ini](pytest.ini), and pytest stops if one is missing):

- pytest-html (html test reports)
- pytest-metadata
- pytest-qt (driving the Qt GUI)

The macOS and Windows environment files already include them. Otherwise install them into the environment:

```bash
pip install pytest pytest-html pytest-metadata pytest-qt
```

Everything else the tests import (numpy, nibabel, SimpleITK, trimesh, vtk, h5py, PIL, yaml, psutil, ...) comes with the BabelBrain environment.

Optional packages:

- pytest-xdist (parallel test runs; disabled in `pytest.ini`, avoid for GPU tests)
- pytest-benchmark (code performance)
- pytest-profiling (finding bottlenecks)
- pytest-sugar (more readable output)

```bash
pip install pytest-xdist pytest-benchmark pytest-profiling pytest-sugar
```

## 4. Optional external programs

Tests needing these are skipped, not failed, when they're missing:

- **SimNIBS** (set `simnibs_root_path`, below): soft tissue segmentation regression cases and the PlanTUS integration test.
- **Blender**: the `blender` variants of the DoIntersect unit tests (trimesh must find the Blender executable). The `manifold` variants need nothing extra.
- **openssl** on the `PATH`: the TLS cases of the remote server tests.

## 5. config.ini

Tests read user-specific paths and the GPU device from `Tests/config.ini`, which is untracked. Copy [example_config.ini](example_config.ini) to `config.ini` and update it:

```ini
[Paths]
data_folder_path  = <path to test data>
ref_output_folder_1 = <path to reference BabelBrain output data set 1>
ref_output_folder_2 = <path to reference BabelBrain output data set 2>
gen_output_folder = <path to folder to store outputs from generate_outputs tests>
simnibs_root_path = <path to SimNIBS install (optional)>

[GPU]
device_name = <GPU device name>

[Remote]   # optional, see example_config.ini
host = <server address>
```

- `data_folder_path` contains the test data, including image sets (structure below).
- `ref_output_folder_1` contains previously generated BabelBrain outputs used as the reference in regression tests.
- `ref_output_folder_2` contains a second set of generated outputs, for comparing two output folders with each other.
- `gen_output_folder` is where generated outputs are stored (generate_outputs and regression tests).
- `device_name` picks the GPU when a backend lists several: the first computing engine whose name contains it (e.g. `M3`, `A6000`), otherwise the backend's first device.
- `simnibs_root_path` is the SimNIBS install folder (the one containing `simnibs_env`). Only the soft tissue segmentation regression cases and the PlanTUS integration test need it; they are skipped without it.
- The optional `[Remote]` section points the remote server E2E test at an existing BabelBrain job server; without it the test starts a local one.

Every `[Paths]` entry except `simnibs_root_path`, and `device_name`, must be present even if the tests you run don't use it (point unused ones at any folder): the file is read when the tests are collected.

## 6. Test data

`data_folder_path` should have the structure below.

<pre>
BabelBrain_Test_Data/
├── ID_0082/
│     ├── m2m_ID_0082/            (charm output, incl. final_tissues.nii.gz and the .msh mesh)
│     ├── T1W.nii.gz
│     ├── CT.nii.gz
│     ├── ZTE.nii.gz
│     └── ...                     (PETRA.nii.gz / Density.nii.gz if available)
├── ernie/
│     └── m2m_ernie/              (charm output, T1 is m2m_ernie/T1.nii.gz)
└── Trajectories/
      ├── &lt;type&gt;_&lt;dataset&gt;_superficial_target_left.txt
      ├── &lt;type&gt;_&lt;dataset&gt;_deep_target_left.txt
      ├── &lt;type&gt;_&lt;dataset&gt;_superficial_target_right.txt
      ├── &lt;type&gt;_&lt;dataset&gt;_deep_target_right.txt
      ├── &lt;type&gt;_&lt;dataset&gt;_superficial_targets.txt   (combined, .yaml for slicer)
      └── &lt;type&gt;_&lt;dataset&gt;_deep_targets.txt         (combined, .yaml for slicer)
</pre>

> **Note:** ID_0082 is a private dataset and isn't shared. ernie is public: it's the example dataset on SimNIBS' [Example Dataset page](https://simnibs.github.io/simnibs/build/html/dataset.html). The trajectory files aren't shared either, so create your own for each dataset you use.

`<type>` is brainsight, slicer or localite (.xml). For ernie, run `charm ernie <T1> <T2>` on the example dataset's scans and copy the resulting `m2m_ernie` folder (keep the whole charm output; PlanTUS needs `final_tissues.nii.gz` and `eeg_positions/` next to the mesh). The GPU unit tests and most VTK viewer and PlanTUS unit tests use a small synthetic head instead, generated automatically (see [synthetic_dataset.py](helpers/synthetic_dataset.py)) into the gitignored `Tests/.cache` folder, so they don't need the data folder. The thermal profiles used by the tests are in this repo's [Profiles](Profiles) folder; the multipoint profile comes from BabelBrain's own `Profiles` folder.

Datasets, targets, profiles and the release test matrix are defined in [parameters.py](helpers/parameters.py). Transducers, their frequencies and the drop-down options are read from the BabelBrain checkout under test, so new ones are picked up automatically.

## 7. Check the setup

```bash
# test parameters still match BabelBrain, and the test data is in place (no GPU, a few seconds)
pytest Tests/Consistency

# unit tests that need no test data or GPU
pytest Tests/Unit -m "not real_data and not slow and not gpu"

# your GPU backend works (replace Metal with CUDA or OpenCL)
pytest Tests/Unit -m "gpu and low_res" -k "synthetic and Metal"
```

`test_optional_test_inputs_exist` is skipped, listing the missing files in the log, when inputs only some tests need are missing; those tests are skipped too.

# Test suite layout

<pre>
BabelBrain/
├── pytest.ini
├── BabelBrain/
├── TranscranialModeling/
├── Tests/
│     ├── config.ini
│     ├── conftest.py               (fixtures and hooks, delegating to helpers/)
│     ├── helpers/                  (code the tests import, never run directly)
│     │    ├── parameters.py        (datasets, targets, profiles, transducers, release test matrix)
│     │    ├── collection.py        (default parametrization and markers)
│     │    ├── pipeline_helpers.py  (set up, run and compare BabelBrain pipelines)
│     │    ├── comparisons.py       (array, mask, mesh and h5 output comparisons)
│     │    ├── plots.py             (report plots: volume slices, mesh views)
│     │    ├── report_viewers.py    (slider viewer and report entry html)
│     │    ├── mesh_checks.py       (point-in-mesh checks for intersections)
│     │    ├── blender_view.py      (Blender's own boolean result and .blend scenes)
│     │    └── synthetic_dataset.py (small synthetic head for the GPU unit tests)
│     ├── scripts/                  (run by hand around pytest, never imported by tests)
│     │    ├── isolate_pytest.py    (one pytest process per test)
│     │    ├── combine_html_reports.py
│     │    └── examples/            (example regression shell scripts)
│     ├── Profiles/                 (pytest thermal profiles)
│     ├── Consistency/              (checks parameters.py still matches BabelBrain and the test data)
│     ├── Unit/                     (mirrors the BabelBrain repo: &lt;source folder&gt;/test_&lt;module&gt;.py)
│     │    ├── BabelBrain/
│     │    │    ├── test_BabelBrain.py
│     │    │    ├── test_BabelDatasetPreps.py
│     │    │    ├── test_FileManager.py
│     │    │    ├── test_RemoteServers.py, test_client_functions.py, test_server.py
│     │    │    ├── GPUFunctions/
│     │    │    │    ├── GPUBinaryClosing/test_BinaryClosing.py
│     │    │    │    ├── GPUResample/test_Resample.py
│     │    │    │    └── ...
│     │    │    ├── GUIComponents/
│     │    │    │    └── test_nifti_viewer.py
│     │    │    └── PlanTUSViewer/
│     │    │         └── test_RunPlanTUS.py
│     │    └── TranscranialModeling/
│     │         └── babel_integration/
│     │              └── test_babel_integration_helpers.py
│     ├── Integration/              (components working together, e.g. test_vtk_visualization.py, test_remote_server_connection.py, test_plantus.py)
│     ├── E2E/
│     ├── Regression/
│     └── Generate_Outputs/
└── ...
</pre>

Where new code goes:

- **Unit tests** mirror the BabelBrain repo: the tests for `BabelBrain/GPUFunctions/GPUResample/Resample.py` are in `Tests/Unit/BabelBrain/GPUFunctions/GPUResample/test_Resample.py`. Put a new unit test in the folder matching its module's path.
- **`helpers/`** holds code that pytest imports through `conftest.py` and the test files, so the suite breaks without it. New code a test or fixture needs goes here.
- **`scripts/`** holds command line tools you run yourself before or after a pytest run (e.g. `python Tests/scripts/combine_html_reports.py ./reports`). No test imports them.

Types of tests:

- **Unit tests** check individual functions or "units" of BabelBrain.
- **Integration tests** check that multiple components work together, e.g. [test_vtk_visualization.py](Integration/test_vtk_visualization.py) runs BabelBrain's VTK visualization methods against the real viewer. They're in `Tests/Integration/`.
- **E2E (end-to-end) tests** check that the entire pipeline works.
- **Regression tests** check BabelBrain outputs stay consistent with previously generated reference outputs.
- **Generate_Outputs** is a "test" that batch runs and saves BabelBrain outputs to use as references in later regression tests.

# Running tests

## In a terminal

Running `pytest Tests` from the top level BabelBrain folder discovers and runs **all** tests, configured by [pytest.ini](pytest.ini). An html report is written to `PyTest_Reports/` and renamed with the time the run finished.

Select tests by name with `-k` and by marker with `-m`:

```bash
# All tests in test_FileManager.py
pytest Tests -k "test_FileManager.py"

# Tests whose id contains generate_hash OR nibabel
pytest Tests -k "generate_hash or nibabel"

# Tests whose id contains both test_full_pipeline_normal AND Metal
pytest Tests -k "test_full_pipeline_normal and Metal"

# Tests with the gpu marker but not the slow one
pytest Tests -m "gpu and not slow"
```

The markers are listed in [pytest.ini](pytest.ini). Add `--collect-only` to see which tests are selected without running them:

```bash
pytest Tests -m "basic_babelbrain_params" --collect-only
```

See the [pytest documentation](https://docs.pytest.org/en/stable/how-to/usage.html) for other options.

## In Visual Studio Code

Install the Python extension, then open the Command Palette (Ctrl+Shift+P, or Cmd+Shift+P on macOS), select "Open Workspace Settings (JSON)" and set:

```json
{
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestPath": "<path_to_your_pytest>"
}
```

For a conda environment the pytest path looks like `<user>/miniconda3/envs/<env>/Lib/site-packages/pytest`. The tests then show in the Testing tab and can be run from there; the command line options in `pytest.ini` still apply. If they don't appear, select "Python: Select Interpreter" and choose the environment holding pytest.

## Reports

Every test's html report entry shows its plots through the same viewer ([report_viewers.py](helpers/report_viewers.py)): drag the slider (slice position, camera position, ...) or press Play to step through. All panels of a viewer stay on the same frame, and nothing plays until you press Play. Screenshots of the GUI (e.g. after each step of the full pipeline tests) are shown as plain images. The VTK visualization tests add a captioned screenshot of the viewer at each point they check (its three views with the colour bar, and its layer panel) through the `viewer_screenshots` fixture ([vtk_viewer_helpers.py](helpers/vtk_viewer_helpers.py)); use it the same way in new viewer tests.

## Logs

Tests log what they're doing at INFO level. The messages are shown live in the terminal (`log_cli` in [pytest.ini](pytest.ini)), saved in `PyTest_Reports/pytest.log`, and added to a failing test's report entry. Every test starts with a `START <test id>` line listing its parameters, and ends with its outcome and duration. In between:

- **Pipeline tests** (E2E, regression, generate outputs): the inputs of the run (dataset, trajectory, scan type, transducer, backend, profiles, output folder), whether saved outputs are reused or regenerated and why, each step (and trajectory, and Combine Trajectories) starting and finishing with its duration, and for comparisons the reference used and a match/DIFFERS line per file plus a summary.
- **GPU unit tests**: the input, the GPU call's duration and output, whether the CPU truth was reloaded or computed (and how long it took), and the comparison metric.
- **Other tests**: their main inputs and results (e.g. mesh sizes and the intersection volume for DoIntersect, PlanTUS's own output as it runs, the layers of every viewer after each BabelBrain call in the VTK visualization integration tests).

To see less, raise the level for one run (`pytest -o log_cli_level=WARNING ...`), or comment out `log_cli = true` to only keep the log file.

# Unit tests

Run a unit test by its name. For example, after changing `generate_hash` in FileManager.py,

```bash
pytest Tests -k "test_generate_hash"
```

runs `test_generate_hash`, `test_generate_hash_file_not_found` and `test_generate_hash_read_error`. To run the tests of every function in FileManager.py:

```bash
pytest Tests/Unit/BabelBrain/test_FileManager.py
```

Some unit tests are parametrized. `test_nibabel_to_sitk` reads T1 data, so it runs once per dataset (`test_nibabel_to_sitk[ID_0082]` and `test_nibabel_to_sitk[ernie]`). To run one dataset:

```bash
pytest "Tests/Unit/BabelBrain/test_FileManager.py::TestFileManager::test_nibabel_to_sitk[ID_0082]"
```

(`-k "test_nibabel_to_sitk and ID_0082"` also works, but matches `test_nibabel_to_sitk_and_back[ID_0082]` too, since `-k` matches substrings.)

## Unit tests and real data

Unit tests that read the real datasets (ID_0082, ernie) get the `real_data` marker automatically. Add `-m "not real_data"` to run only the unit tests that use data they generate themselves (e.g. the synthetic head), which is quicker and needs no test data:

```bash
pytest Tests/Unit -m "not real_data"         # unit tests without real data
pytest Tests/Unit                            # all unit tests
pytest Tests/Unit -m "real_data"             # only the unit tests using real data
pytest Tests/Unit -m "gpu and not real_data" # combined with other markers
```

Only unit tests are marked: E2E, regression and generate_outputs tests always use the real datasets.

## GPU unit tests

GPU unit tests have the `gpu` marker and are parametrized by computing backend (CUDA, OpenCL and Metal; backends your operating system doesn't support are skipped) and spatial step:

- low_res
- medium_res
- high_res (marked `slow`)
- stress_res (marked `slow`)

Every spatial step runs on the real datasets (ID_0082, ernie) and on a small synthetic head (dataset id `synthetic`, generated automatically by [synthetic_dataset.py](helpers/synthetic_dataset.py) into the gitignored `Tests/.cache` folder), which makes the high resolutions practical.

```bash
pytest Tests/Unit -m "gpu and not slow" -k "synthetic and Metal"   # quick check on your machine's backend
pytest Tests/Unit -m "gpu"                                         # all GPU unit tests
pytest Tests/Unit -m "gpu and not slow"                           # without high_res and stress_res
pytest Tests/Unit -k "ID_0082-High_Res and Metal"                  # one dataset and resolution
```

Each GPU test's report entry shows the inputs, the CPU and GPU outputs and a |GPU - CPU| difference panel (with the number of differing voxels in the whole volume), with a slider that moves the slice position through all three axes. The voxelization test shows the skin mesh, both voxelizations and the voxels only one of them found, with a slider that moves the camera around the mesh.

## Step specific unit tests

Unit tests for the functions each BabelBrain step calls are marked `step1`, `step2` and `step3`:

```bash
pytest Tests/Unit -m "step1"
```

# End-to-end tests

End-to-end tests run the full BabelBrain pipeline. They are parametrized by the values in [parameters.py](helpers/parameters.py) (options the BabelBrain version under test doesn't offer are left out):

- Trajectory type: brainsight, slicer, localite
- Target: Superficial_Target_Left, Deep_Target_Left
- CT scan file: NONE, CT, ZTE, PETRA, Density
- Dataset: ID_0082, ernie
- Transducer: Single, CTX_500, ...
- Frequency: depends on the transducer
- Computing backend: CUDA, OpenCL, Metal (MLX isn't tested until BabelBrain fully supports it)

That is over 100,000 versions of one full pipeline test, so narrow the selection down. Either name the parameters:

```bash
pytest Tests -k "test_full_pipeline_normal and slicer and Deep_Target_Left and ernie and NONE and Single and 200kHz and CUDA"
```

or use one of the configuration markers below.

## Configurations

The `basic_babelbrain_params` marker selects the basic configuration, the standard full pipeline set up:

- Dataset: ernie
- CT scan file: NONE
- Trajectory type / target: brainsight / Superficial_Target_Left
- Transducer / frequency: Single 500 kHz, CTX_500 500 kHz, H246 500 kHz, TheraPhas 490 kHz, H317 250 kHz, DomeTx 220 kHz (one per transducer geometry type)
- Computing backend: CUDA, OpenCL, Metal (backends not available on your system are skipped)

It's defined by `BASIC_PARAMS` in [parameters.py](helpers/parameters.py). The major release `regression_test_full_pipeline` (see [BabelBrain_Pytest_Usage.md](BabelBrain_Pytest_Usage.md), Table 10) runs this same configuration: its `REGRESSION_CASES` entry is `BASIC_PARAMS`, so changing the configuration there updates both.

The `intensive_babelbrain_params` marker selects the intensive configuration: ID_0082 with CT, Deep_Target_Left, H317 at 825 kHz on CUDA, OpenCL and Metal. Tests with this marker also use the configuration's intensive thermal profile, so they run it exactly. It's defined by `INTENSIVE_SIM_PARAMS` in [parameters.py](helpers/parameters.py), and the major release `regression_test_full_pipeline_intensive` runs this same configuration. Multipoint focusing is tested by the E2E test `e2e_test_multipoint_full_pipeline` instead, on the quick configuration (`QUICK_SIM_PARAMS`: ernie, MRI only, superficial target) with H317 at 250 kHz, since multipoint profiles need a phased array.

```bash
pytest Tests -k "test_full_pipeline_normal" -m "intensive_babelbrain_params"
```

The other `*_babelbrain_params` markers choose their transducers and frequencies with `MARKER_TRANSDUCERS` in the same file. These configurations are for checking a change by hand; releases are tested with the `minor_release` / `major_release` markers (see [Release testing](#release-testing)).

## test_full_pipeline_normal

Checks that BabelBrain runs to completion without errors. It saves plots but doesn't check outputs, so review the results to verify they make sense visually.

```bash
pytest Tests -k "test_full_pipeline_normal" -m "basic_babelbrain_params"
```

## test_full_pipeline_regression_normal

Compares the results of the current setup with a reference folder of previously generated outputs (see [Generating outputs](#generating-outputs) to create one). Set `ref_output_folder_1` in `config.ini` to the reference, e.g. to compare against BabelBrain v0.4.2:

```ini
[Paths]
...
ref_output_folder_1 = /<User>/BabelBrain_Unit_Testing/Generated_Outputs/BabelBrain_0_4_2
```

then run

```bash
pytest Tests -k "test_full_pipeline_regression_normal" -m "basic_babelbrain_params"
```

Outputs are compared with 0 tolerance, so any difference from the reference fails the test and shows up in the report.

## test_full_pipeline_two_outputs

Compares two previously generated output folders without running BabelBrain. Set `ref_output_folder_1` and `ref_output_folder_2` in `config.ini`, e.g. to compare Metal with MLX outputs:

```ini
[Paths]
...
ref_output_folder_1 = /<User>/BabelBrain_Unit_Testing/Generated_Outputs/METAL
ref_output_folder_2 = /<User>/BabelBrain_Unit_Testing/Generated_Outputs/MLX
```

then run

```bash
pytest Tests -k "test_full_pipeline_two_outputs"
```

Outputs are compared with 0 tolerance, as above. There is no `basic_babelbrain_params` marker here: it applies when generating the outputs beforehand.

The html report has one section per compared subfolder (parameter set/device), each showing the reference and test file paths. For each mismatched 3D field it embeds a slice viewer (reference, test and difference side by side, with a fixed colour scale across slices). Each viewer samples up to 24 slices spread evenly across the volume, to keep the report size reasonable. Mismatched single values (e.g. scalar summary metrics) are listed in a table with their reference and test values and absolute and relative differences.

## Release E2E tests

[E2E/test_e2e_release.py](E2E/test_e2e_release.py) holds the major release E2E tests (remote server, custom transducer, multipoint and combined trajectories full pipelines, Table 9 of [BabelBrain_Pytest_Usage.md](BabelBrain_Pytest_Usage.md)).

# Generating outputs

`test_generate_valid_outputs` doesn't check BabelBrain: it uses pytest and qtbot to generate outputs for many BabelBrain parameters, to use later as regression references. Set `gen_output_folder` in `config.ini` to a folder whose name describes what you're generating, e.g.

```ini
[Paths]
...
gen_output_folder = /<user>/BabelBrain_Unit_Testing/Generated_Outputs/BabelBrain_v0_4_2
```

(or `.../Metal` for outputs using only the Metal backend, `.../Cupy_v13_6` for a newer cupy version, ...), then run

```bash
pytest Tests -k "test_generate_valid_outputs" -m "basic_babelbrain_params"
```

Each case is saved in `<gen_output_folder>/<case id>` together with a `pytest_case_complete.yaml` marker written once the run succeeded. A later run skips cases already generated by the same BabelBrain commit with the same trajectory and profile files, and reruns (from an empty folder) anything else, including folders from a run that crashed halfway. Set `BABEL_PYTEST_REGENERATE=1` to rerun every case.

# Release testing

The tests required for minor and major releases are described in [BabelBrain_Pytest_Usage.md](BabelBrain_Pytest_Usage.md). The tests in each release set are marked `minor_release` / `major_release`, and these markers are how releases are tested:

```bash
# minor release set on this machine's backend
pytest Tests -m "minor_release" -k "not gpu or Metal"

# major release set: unit, integration and consistency tests directly, full pipeline tests one per process
pytest Tests/Unit Tests/Integration Tests/Consistency -m "major_release"
python Tests/scripts/isolate_pytest.py --run -- Tests/E2E Tests/Regression -m "major_release"
```

Add `--collect-only` to the pytest command (or leave out `--run`, which only writes the per-case runner script) to see what a selection contains without running it.

What each set contains:

- **Unit tests and consistency checks** (marked automatically): GPU unit tests run the synthetic head at every resolution in both sets, and the real datasets only in the major set, at medium_res (`RELEASE_SPATIAL_STEPS` in [parameters.py](helpers/parameters.py)). Every other unit test and the consistency checks are in the major set, and in the minor set too unless they're slow or read the real datasets, so minor releases need no test data. The DoIntersect tests on the real datasets (`TestDoIntersectRealData`) are slow, so only major releases run them; minor releases run `TestDoIntersectSynthetic`.
- **Integration tests** (marked in each file): [test_vtk_visualization.py](Integration/test_vtk_visualization.py) and [test_remote_server_connection.py](Integration/test_remote_server_connection.py) are in both sets (Tables 4 and 8), [test_plantus.py](Integration/test_plantus.py) only in the major set (it's slow and needs SimNIBS and ernie).
- **E2E and regression tests**: [E2E/test_e2e_release.py](E2E/test_e2e_release.py) (Table 9) and [Regression/test_regression_release.py](Regression/test_regression_release.py) (Table 10) are marked `major_release`. For a minor release, run only the E2E tests of the features the release affects (Table 5), selected from test_e2e_release.py by name, e.g. `pytest Tests/E2E -k "vtk_visualization and Metal"`. Regression tests aren't run for minor releases (Table 6).

Full pipeline tests take long and use a lot of memory and GPU resources, and running many in one process can crash (especially on Windows). Run them through [isolate_pytest.py](#isolating-a-pytest-run-crash-and-memory-triage), which gives each test its own process, log and report.

The regression tests compare against reference outputs that have to be generated beforehand, see [Regression tests and reference outputs](#regression-tests-and-reference-outputs).

The parameters of each regression test (one entry per row of Table 10: dataset, trajectory, target, scan types, transducers, profiles, ...) are in `REGRESSION_CASES` in [parameters.py](helpers/parameters.py); change a row there and the regression cases and the consistency checks follow. `regression_test_full_pipeline` and `regression_test_full_pipeline_intensive` use the basic and intensive configurations, `BASIC_PARAMS` and `INTENSIVE_SIM_PARAMS`, so they're edited there and the `basic_babelbrain_params` / `intensive_babelbrain_params` markers follow too. `regression_test_full_pipeline_soft_tissue` uses `SOFT_TISSUE_SIM_PARAMS` (`QUICK_SIM_PARAMS` with soft tissue segmentation; needs SimNIBS). `regression_test_GetSkullMaskFromSimbNIBSSTL` runs Step 1 for every scan type on `QUICK_SIM_PARAMS` with ID_0082 (ernie has no CT, ZTE, PETRA or Density scans). The E2E tests that run a parameter set, `e2e_test_remote_full_pipeline`, `e2e_test_multipoint_full_pipeline`, `e2e_test_combined_trajectories_full_pipeline` and `e2e_test_vtk_visualization_full_pipeline` (Table 9), are in `E2E_CASES` and use `REMOTE_SIM_PARAMS`, `MULTIPOINT_SIM_PARAMS`, `COMBINED_TRAJECTORIES_SIM_PARAMS` and `VTK_VISUALIZATION_SIM_PARAMS`, all built on the quick configuration `QUICK_SIM_PARAMS`. [test_suite_consistency.py](Consistency/test_suite_consistency.py) checks that every dataset, target, scan type, transducer/frequency and input file a regression or E2E case names exists.

Tests for features an older BabelBrain version doesn't have (remote servers, custom transducers, PlanTUS) are skipped automatically.

## Regression tests and reference outputs

A regression test compares outputs of the BabelBrain version being released with **reference outputs** made earlier with the **reference BabelBrain version** (Table 1 of [BabelBrain_Pytest_Usage.md](BabelBrain_Pytest_Usage.md)). The references aren't made during the release run: generate them first, on the release machine.

### 1. Generate the reference outputs (once per reference version and machine)

1. In the BabelBrain checkout, check out the reference version (`git checkout <reference tag or commit>`). `Tests/` is git-ignored, so it stays on the current test suite; tests for features the reference version doesn't have are skipped.
2. In `Tests/config.ini`, point `gen_output_folder` at the reference location, e.g. `.../Generated_Outputs/v0.8.1`.
3. Generate every regression case, one process per case:

    ```bash
    python Tests/scripts/isolate_pytest.py --run -- Tests/Regression -m generate_outputs
    ```

4. Check out the version being released again, set `ref_output_folder_1` to the reference location and set `gen_output_folder` back to a folder for the new outputs.

Each case is saved in `<folder>/<suite>/<case id>/`, e.g. `regression_full_pipeline/ernie_brainsight_CT-NONE_Superficial_Target_Left_H317_Freq-250kHz_Metal/`. Generate the references on every backend the release is tested on (see the pitfalls below).

### 2. Run the regression tests

```bash
python Tests/scripts/isolate_pytest.py --run -- Tests/Regression -m "major_release"
```

Every regression test comes in two halves, both selected by `major_release`:

- **`test_generate_<name>`** (marked `generate_outputs`) runs the pipeline for a case and saves its outputs in `<gen_output_folder>/<suite>/<case id>/`. Only once the run succeeds does it write `pytest_case_complete.yaml` there, recording the BabelBrain version and commit and a hash of the trajectory and profile files.
- **`test_<name>`** looks up the case's reference, gets the case's outputs and compares them:
  1. The reference is `<ref_output_folder_1>/<suite>/<case id>/`. Without one, the same case from another backend is used (with a warning in the log); without any, the test is **skipped**, before anything is run.
  2. The outputs saved by `test_generate_<name>` are reused (see below), otherwise the case is run now.
  3. Outputs must match **exactly** (0 tolerance). The full pipeline tests compare the thermal outputs (`.h5`); `regression_test_GetSkullMaskFromSimbNIBSSTL` compares every NIfTI, npy, stl and h5 output of Step 1 (not the execution times or the completion marker). Differences are listed in the log and shown in the html report.

Because the outputs are saved before anything is compared, a failed comparison never throws away a finished simulation.

### Saved outputs are reused

A case's saved outputs are reused, by the other half of the test, by later pytest processes (each case is its own process under isolate_pytest.py) and by later sessions, when its `pytest_case_complete.yaml` shows:

- the same BabelBrain commit (the current `git rev-parse HEAD`, with `-dirty` added when BabelBrain has uncommitted changes), and
- the same trajectory and profile files.

Otherwise the case folder is deleted and the case is run again. A folder without `pytest_case_complete.yaml` is a run that crashed halfway, and is rerun too. Set `BABEL_PYTEST_REGENERATE=1` to rerun every case regardless.

### Pitfalls

- **Generate references for each backend.** When a backend has no reference, another backend's is used, but CUDA, OpenCL and Metal results usually differ slightly, which fails at 0 tolerance and looks like a regression. Generate the references for every backend you test on (on each machine) so the fallback isn't needed.
- **The reference version isn't checked.** The references' `pytest_case_complete.yaml` records the BabelBrain version that made them, but the tests don't compare it with Table 1: whatever is in `ref_output_folder_1` is used. Check the folder holds the right version.
- **Uncommitted BabelBrain changes.** With uncommitted changes the commit is recorded as `<commit>-dirty`, which doesn't change when you edit BabelBrain again without committing, so outdated outputs would be reused. Use `BABEL_PYTEST_REGENERATE=1` while changing BabelBrain, and run release tests on a committed version.
- **Changing a regression case changes its case ids.** A change to `REGRESSION_CASES` (another transducer, dataset, ...) changes the case ids, so the existing references no longer match and those tests are skipped until references are generated for the new cases.

# Isolating a pytest run (crash and memory triage)

Use this for full pipeline tests (E2E, regression, generate outputs) on every platform. A large selection is a single long-lived process: `pytest Tests/Regression -m "major_release"` runs all its cases in one interpreter, and GPU code tends to crash when many simulations run in a row. When that process dies halfway through (killed by a signal, or by the OS after memory creeps up), the terminal scrollback is all you get, the html report is never written, and there is no record of which case was running.

[scripts/isolate_pytest.py](scripts/isolate_pytest.py) takes the same arguments you would give pytest, asks pytest which cases they select, and runs **each case in its own pytest process**. A process per case is the only bullet-proof deallocation: when it exits, every host buffer, Metal/CUDA/OpenCL context and Qt object is released back to the OS, so anything that still grows across cases is on disk or in the GPU driver rather than a leak that teardown would fix.

```bash
# generate a per-case runner script (run_isolated.sh, or .bat on Windows), run nothing
python Tests/scripts/isolate_pytest.py -- Tests/E2E Tests/Regression -m "major_release"

# collect and run now, one case per process
python Tests/scripts/isolate_pytest.py --run -- Tests/E2E Tests/Regression -m "major_release"

# rerun only the cases that did not pass
python Tests/scripts/isolate_pytest.py --run --from-file PyTest_Reports/isolated/<run>/failed_nodeids.txt
```

Everything after `--` goes to pytest verbatim. With no path argument the whole `Tests` tree is selected. The tool works from any directory: it finds the BabelBrain checkout holding this `Tests` clone from its own location and runs pytest from there (`--root` points it at a different checkout).

## What you get

Artifacts are written to `<BabelBrain>/PyTest_Reports/isolated/<timestamp>/`:

| file | contents |
| --- | --- |
| `nodeids.txt` | every collected case, one per line |
| `run_isolated.sh` / `.bat` | one `run_case` line per case; edit or comment out lines freely |
| `logs/NNNN_<case>.log` | full output of that case, flushed line by line so a hard kill still leaves a readable file |
| `summary.csv` | status, exit code, duration, peak RSS and free memory per case |
| `html/NNNN_<case>.html` | that case's pytest-html report, with the GUI screenshots embedded |
| `failed_nodeids.txt` | feed straight back in with `--from-file` |

Exit codes are decoded per case: `PASS`, `FAIL`, `CRASH` with the signal name (`SIGSEGV`, `SIGBUS`, `SIGABRT`, `SIGKILL`, the last usually meaning the OS ran out of memory), `TIMEOUT`, `MEMLIMIT`, and `NO_TESTS_RUN`. Child processes run with `PYTHONFAULTHANDLER=1`, so a native segfault prints the Python line that caused it instead of dying silently.

## Chasing a memory problem

Every row of `summary.csv` records peak RSS of the whole process tree plus free system memory before, during and after the case.

```bash
# is one case at fault, or is it accumulation? passes at 1, dies at 20 => accumulation
python Tests/scripts/isolate_pytest.py --run --group 20 -- Tests/Regression -m "major_release"

# does one case leak? run it 5x inside a single process and watch peak RSS
python Tests/scripts/isolate_pytest.py --run --repeat-each 5 --group 5 -- -k "H317 and Metal and NONE" -m basic_babelbrain_params

# stop a runaway before the OS kills your session, and bound a hang
python Tests/scripts/isolate_pytest.py --run --max-rss-mb 60000 --timeout 3600 -- Tests/Regression -m "major_release"
```

Other useful options: `--stop-on-crash`, `--start-at`/`--max-cases` to resume a long run, `--cooldown` to let the GPU driver settle between cases, `--no-html` to skip the per-case html report (keep it on if a test takes screenshots: `conftest.py` attaches them through `pytest_html.extras`, which only exists while the html plugin is loaded), `--quiet-child` to keep the console tidy (the logs are unaffected), and `--dry-run` to see the commands. Long GPU steps produce no output for minutes, so the runner prints a heartbeat with elapsed time and current RSS while a case is silent (`--heartbeat SEC`, `0` disables). Run `--help` for the full list.

# Writing tests

Pytest tests use Python's `assert` statement. Test files and test functions start with `test_` so pytest discovers them:

```python
# test_example.py
import pytest

def test_addition():
    assert 1 + 2 == 3, "message shown if assertion fails"

@pytest.mark.custom_marker_1 # example of a marker
def test_subtraction():
    assert 2 - 1 == 1, "message shown if assertion fails"
```

Put the file where [Test suite layout](#test-suite-layout) says, and register any new marker in [pytest.ini](pytest.ini). Tests that ask for the `dataset`, `transducer`, `computing_backend`, `spatial_step`, ... fixtures are parametrized over them automatically ([collection.py](helpers/collection.py)); parametrize one yourself with `@pytest.mark.parametrize` to narrow it.

For more on pytest, see the [pytest documentation](https://docs.pytest.org/en/stable/).
