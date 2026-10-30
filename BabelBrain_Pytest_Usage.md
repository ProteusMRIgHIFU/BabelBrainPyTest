# BabelBrain Pytest Usage

<style>
  /* Table dropdowns: add a subtle note when a table is hidden */
  details > summary { cursor: pointer; }
  details:not([open]) > summary::after { content: " (hidden)"; color: gray; font-style: italic; font-weight: normal; }
</style>

## Revision History

<br>

<details open>
<summary><b>Table 1: Revision History</b></summary>

<table width="100%" border="1" cellpadding="6">
  <thead>
    <tr>
      <th width="10%" style="background-color:#d6001c; color:white"><b>Version</b></th>
      <th width="20%" style="background-color:#d6001c; color:white"><b>Date</b></th>
      <th width="25%" style="background-color:#d6001c; color:white"><b>Author</b></th>
      <th width="45%" style="background-color:#d6001c; color:white"><b>Description</b></th>
    </tr>
  </thead>
  <tbody>
    <!-- 0.1 -->
    <tr>
      <td>0.1</td>
      <td>Sep 24, 2026</td>
      <td>Alan Coreas</td>
      <td>
        Initial Draft<br>
        Regression testing is against v0.8.1
      </td>
    </tr>
  </tbody>
</table>

</details>

<br>

## Table of Contents

- [BabelBrain Pytest Usage](#babelbrain-pytest-usage)
  - [Revision History](#revision-history)
  - [Table of Contents](#table-of-contents)
  - [1. Purpose](#1-purpose)
  - [2. Available Test Parameters](#2-available-test-parameters)
  - [3. Minor Release Testing](#3-minor-release-testing)
    - [3.1 Unit Tests](#31-unit-tests)
    - [3.2 Integration Tests](#32-integration-tests)
    - [3.3 End-to-End Tests](#33-end-to-end-tests)
    - [3.4 Regression Tests](#34-regression-tests)
  - [4. Major Release Testing](#4-major-release-testing)
    - [4.1 Unit Tests](#41-unit-tests)
    - [4.2 Integration Tests](#42-integration-tests)
    - [4.3 End-to-End Tests](#43-end-to-end-tests)
    - [4.4 Regression Tests](#44-regression-tests)
  - [5. Release Testing Summary](#5-release-testing-summary)
  - [6. Selecting Parameter Combinations](#6-selecting-parameter-combinations)
  - [7. Adding New Test Combinations](#7-adding-new-test-combinations)
  - [8. Pass Criteria](#8-pass-criteria)

## 1. Purpose

This document defines the minimum test and parameter combinations required for software releases.

The goal is to provide sufficient coverage while avoiding unnecessary testing of every possible parameter combination.

Testing is organized into four categories:

- **Unit Tests** – Verify individual functions or components.
- **Integration Tests** – Verify that multiple components work together correctly.
- **End-to-End (E2E) Tests** – Verify complete user or application workflows.
- **Regression Tests** – Verify that previously working functionality continues to behave as expected.

The required level of testing depends on whether the release is considered **minor** or **major**.

## 2. Available Test Parameters

List the parameters that can be varied during testing.

<br>

<details open>
<summary><b>Table 2: Available Test Parameters</b></summary>

<table width="100%" border="1" cellpadding="6">
  <thead>
    <tr>
      <th width="20%" style="background-color:#d6001c; color:white"><b>Parameter</b></th>
      <th width="25%" style="background-color:#d6001c; color:white"><b>Available Values</b></th>
      <th width="55%" style="background-color:#d6001c; color:white"><b>Notes</b></th>
    </tr>
  </thead>
  <tbody>
    <!-- Computing Backend -->
    <tr>
      <td>Computing Backend</td>
      <td>
        <ul>
          <li>CUDA</li>
          <li>OpenCL</li>
          <li>Metal</li>
          <li>Remote</li>
        </ul>
      </td>
      <td>Different backends for GPU calculations</td>
    </tr>
    <!-- Coregistration -->
    <tr>
      <td>Coregistration</td>
      <td>
        <ul>
          <li>No</li>
          <li>CT to MRI</li>
        </ul>
      </td>
      <td>
        Parameter on whether to skip coregistration of CT to MRI in case it has already been done.<br>
        <b>Default:</b> No if CT Type is None, otherwise CT to MRI
      </td>
    </tr>
    <!-- CT Mapping -->
    <tr>
      <td>CT Mapping</td>
      <td>
        <ul>
          <li>Various values</li>
        </ul>
      </td>
      <td>Specific to scanner</td>
    </tr>
    <!-- CT Type -->
    <tr>
      <td>CT Type</td>
      <td>
        <ul>
          <li>NONE</li>
          <li>CT</li>
          <li>Density</li>
          <li>Petra</li>
          <li>ZTE</li>
        </ul>
      </td>
      <td>
        Additional image data used for improving BabelBrain simulations, dependent on data available in selected test dataset.<br>
        <b>Default:</b> NONE
      </td>
    </tr>
    <!-- Custom Transducer Template -->
    <tr>
      <td>Custom Transducer Template</td>
      <td>
        <ul>
          <li>Simple Focused</li>
          <li>Flat Annular Array</li>
          <li>Focused Annular Array</li>
          <li>Flat Array 2D</li>
          <li>Focused Array</li>
        </ul>
      </td>
      <td>
        Files that replicate existing BabelBrain transducers:
        <ul>
          <li>Simple Focused → Single 500 kHz</li>
          <li>Flat Annular Array → H246</li>
          <li>Focused Annular Array → CTX-500</li>
          <li>Flat Array 2D → TheraPhas (when using a Slicer target, TheraPhas flipping doesn't happen, therefore a 1 to 1 comparison can be done)</li>
          <li>Focused Array → H317</li>
        </ul>
      </td>
    </tr>
    <!-- Frequency -->
    <tr>
      <td>Frequency</td>
      <td>
        <ul>
          <li>Varies from 200 kHz to 1100 kHz</li>
        </ul>
      </td>
      <td>Frequency of simulation, dependent on available frequencies for transducer</td>
    </tr>
    <!-- Multipoint Profile -->
    <tr>
      <td>Multipoint Profile</td>
      <td>
        <ul>
          <li>MultiFocus_Profile1</li>
        </ul>
      </td>
      <td>File that specifies additional targets to be sonicated in relation to the initial central target</td>
    </tr>
    <!-- Multipoint -->
    <tr>
      <td>Multipoint</td>
      <td>
        <ul>
          <li>Yes</li>
          <li>No</li>
        </ul>
      </td>
      <td>
        Enables multifocal operation.<br>
        <b>Default:</b> No
      </td>
    </tr>
    <!-- Preprocessed Data Generator -->
    <tr>
      <td>Preprocessed Data Generator</td>
      <td>
        <ul>
          <li>Charm</li>
          <li>Headreco</li>
        </ul>
      </td>
      <td>
        Library used to generate processed data from the test dataset MRI images. Headreco is essentially obsolete.<br>
        <b>Default:</b> Charm
      </td>
    </tr>
    <!-- Steering -->
    <tr>
      <td>Steering</td>
      <td>
        <ul>
          <li>X Steering value</li>
          <li>Y Steering value</li>
          <li>Z Steering value</li>
        </ul>
      </td>
      <td>
        Electronic steering of focal spot in transducers with that capability.<br>
        <b>Default:</b> 0.0 for all axes
      </td>
    </tr>
    <!-- Soft Tissue Segmentation -->
    <tr>
      <td>Soft Tissue Segmentation</td>
      <td>
        <ul>
          <li>Enabled</li>
          <li>Disabled</li>
        </ul>
      </td>
      <td>Relevant only to Step 1. Configured in the advanced options.</td>
    </tr>
    <!-- Test Dataset -->
    <tr>
      <td>Test Dataset</td>
      <td>
        <ul>
          <li>ID_0082</li>
          <li>ernie</li>
        </ul>
      </td>
      <td>MRI image data + preprocessed data</td>
    </tr>
    <!-- Thermal Profile -->
    <tr>
      <td>Thermal Profile</td>
      <td>
        <ul>
          <li>pytest_thermal_profile_1</li>
          <li>pytest_thermal_profile_2</li>
        </ul>
      </td>
      <td>
        Files specifying the ultrasound parameters (e.g. duty cycle, pulse repetition frequency, etc.), kept in the pytest repo's Profiles folder. Please note that BabelBrain comes with 6 example thermal profiles, but the ones listed here provide better coverage for pytests.<br>
        <b>Default:</b> pytest_thermal_profile_1
      </td>
    </tr>
    <!-- Trajectory File Generator -->
    <tr>
      <td>Trajectory File Generator</td>
      <td>
        <ul>
          <li>Brainsight</li>
          <li>Localite</li>
          <li>Slicer</li>
        </ul>
      </td>
      <td>
        Software used to generate the trajectory file.<br>
        <b>Default:</b> Brainsight
      </td>
    </tr>
    <!-- Trajectory -->
    <tr>
      <td>Trajectory</td>
      <td>
        <ul>
          <li>Superficial Target (left / right)</li>
          <li>Deep Target (left / right)</li>
          <li>Superficial Targets (combined left and right)</li>
          <li>Deep Targets (combined left and right)</li>
          <li>Invalid Target</li>
        </ul>
      </td>
      <td>
        Target trajectory for simulation. Individualized to test datasets and generator. Tests use the left side targets unless stated otherwise.<br>
        <b>Default:</b> Superficial Target (left)
      </td>
    </tr>
    <!-- Transducer -->
    <tr>
      <td>Transducer</td>
      <td>
        <ul>
          <li>Single</li>
          <li>CTX_500</li>
          <li>CTX_250</li>
          <li>CTX_250_2ch</li>
          <li>DPX_500</li>
          <li>DPXPC_300</li>
          <li>H317</li>
          <li>H246</li>
          <li>BSonix</li>
          <li>TheraPhas</li>
          <li>I12378</li>
          <li>ATAC</li>
          <li>R15148</li>
          <li>R15287</li>
          <li>R15473</li>
          <li>R15646</li>
          <li>IGT64_500</li>
          <li>H301</li>
          <li>DomeTx</li>
          <li>R16081</li>
        </ul>
      </td>
      <td>List of supported transducers</td>
    </tr>
  </tbody>
</table>

</details>

<br>

This section should be updated whenever new parameters or supported values are introduced.

## 3. Minor Release Testing

Minor releases should focus on confirming that changed functionality works correctly and that common workflows have not regressed.

The goal is to use a **small but representative set of parameter combinations**.

### 3.1 Unit Tests

Unit tests should focus primarily on code affected by the release. Minor release unit tests need no test data: the unit tests that read the real datasets (ID_0082, ernie) or are slow are only run for major releases (Table 7).

<br>

<details open>
<summary><b>Table 3: Minor Release Unit Tests</b></summary>

<table width="100%" border="1" cellpadding="6">
  <thead>
    <tr>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Test</b></th>
      <th width="40%" style="background-color:#d6001c; color:white"><b>Parameter Combination</b></th>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Reason</b></th>
    </tr>
  </thead>
  <tbody>
    <!-- test_suite_consistency.py (all tests) -->
    <tr>
      <td>test_suite_consistency.py (all tests)</td>
      <td>
        None (no GPU or test data needed)
      </td>
      <td>
        <ul>
          <li>Confirms the test parameters still match the BabelBrain version under test (transducers, frequencies, GUI options, custom transducer templates) and that the test data is in place</li>
          <li>Runs in seconds, run it first</li>
        </ul>
      </td>
    </tr>
    <!-- test_FileManager.py (hashing, wait_for_file, check_reuse_files) -->
    <tr>
      <td>test_FileManager.py (hashing, wait_for_file, check_reuse_files)</td>
      <td>
        <b>CT Type (check_reuse_files tests):</b><br>
        <ul>
          <li>All</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Hash-based caching decides whether Step 1 outputs are reused, a regression here silently reuses stale results</li>
          <li>No GPU needed</li>
        </ul>
      </td>
    </tr>
    <!-- TestDoIntersectSynthetic -->
    <tr>
      <td>TestDoIntersectSynthetic (test_BabelDatasetPreps.py)</td>
      <td>
        None (spheres, boxes and cones built in the test)<br>
        <b>Boolean engine:</b><br>
        <ul>
          <li>manifold (default)</li>
          <li>blender (skipped when Blender isn't installed)</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>DoIntersect cuts the skin with the transducer cone and skin/bone/CSF with the FOV box in Step 1</li>
          <li>Intersection volumes are checked against analytic values (overlapping spheres, spherical sector), plus contained/disjoint cases and the mesh-repair paths (multi-body and open meshes)</li>
          <li>Every test runs for both engines. Blender variants also show Blender's own result (boolean modifier evaluated in Blender) and save the scene to PyTest_Reports/blender_scenes/ (real-data scenes only with BABEL_PYTEST_SAVE_BLENDER_SCENES=1). If trimesh's Blender booleans return empty meshes, Blender tests that expect an intersection fail with that reason; non-overlapping cases always run</li>
          <li>Runs in seconds</li>
        </ul>
      </td>
    </tr>
    <!-- test_BinaryClosing_vs_CPU -->
    <tr>
      <td>test_BinaryClosing_vs_CPU<br>test_LabelImage_vs_CPU<br>test_MappingFilter_vs_CPU<br>test_MedianFilter_vs_CPU<br>test_Resample_vs_CPU<br>test_Voxelize_vs_CPU</td>
      <td>
        <b>Computing Backend:</b><br>
        <ul>
          <li>The backend of the release machine</li>
        </ul>
        <b>Test dataset / Spatial Step:</b><br>
        <ul>
          <li>synthetic (small generated head, see synthetic_dataset.py): Low_Res, Med_Res, High_Res, Stress_Res</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Catches GPU kernel regressions against CPU (scipy/scikit-image/trimesh) reference outputs</li>
          <li>The small synthetic head covers every resolution quickly and needs no test data; the full head datasets are only run for major releases (RELEASE_SPATIAL_STEPS in parameters.py)</li>
          <li>CPU references are cached after the first run</li>
        </ul>
      </td>
    </tr>
    <!-- test_RunPlanTUS.py (all tests) -->
    <tr>
      <td>test_RunPlanTUS.py (all tests)</td>
      <td>
        <b>Test dataset:</b><br>
        <ul>
          <li>synthetic (T1 of the synthetic head, trajectories made for it)</li>
        </ul>
        <b>Transducer/Frequency:</b><br>
        <ul>
          <li>CTX_500/500kHz</li>
        </ul>
        <b>Trajectory:</b><br>
        <ul>
          <li>Brainsight and Slicer, single and combined targets</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>BabelBrain's side of PlanTUS without running it: PlanTUSTxConfig.yaml (has every key PlanTUS_wrapper.py requires), target mask, PlanTUS command line, recalculate/reload prompt, success/error reporting, and the accepted PlanTUS trajectory written back as the _PlanTUS file</li>
          <li>No SimNIBS, GPU or test data needed, runs in seconds</li>
        </ul>
      </td>
    </tr>
    <!-- test_nifti_viewer.py (all tests) -->
    <tr>
      <td>test_nifti_viewer.py (all tests)</td>
      <td>
        <b>Test dataset:</b><br>
        <ul>
          <li>Small block volumes (geometry, loading, navigation, layer controls)</li>
          <li>synthetic (tissue mask, T1W and CT of the synthetic head, rendered)</li>
        </ul>
        <b>View modes:</b><br>
        <ul>
          <li>Affine, medical radiological, medical neurological</li>
        </ul>
        No GPU or test data needed
      </td>
      <td>
        <ul>
          <li>VTK visualization window (GUIComponents/nifti_viewer.py): affine/RAS geometry, NIfTI loading, colour maps and cutoff, slice navigation in affine and medical modes, layer controls, overlay removal, window/level reset, screenshots, one viewer per trajectory</li>
          <li>Synthetic head rendered through the viewer tab's toolbar (view modes, neurological flip, crosshairs, RAS boxes, reset, Screenshot button), with screenshots of every view in the report</li>
          <li>Crosshair drawn when the target is on the middle slices, medical slice grid centre with partial voxel extents</li>
          <li>Tissue label colours are a strict xfail: the label table is applied over the mask's data range, so the labels show the wrong colours</li>
          <li>Skipped on BabelBrain versions without the VTK viewer</li>
        </ul>
      </td>
    </tr>
    <!-- test_RemoteServers.py, test_client_functions.py, test_server.py (all tests) -->
    <tr>
      <td>test_RemoteServers.py<br>test_client_functions.py<br>test_server.py<br>(all tests)</td>
      <td>
        None (no network, GPU or test data needed; TLS certificates are made with openssl)
      </td>
      <td>
        <ul>
          <li>Remote server settings: saved server list, URLs, bearer token headers, client TLS settings (trusted CA, insecure, mutual TLS)</li>
          <li>Server side: HTTPS / mutual TLS context, loopback detection, refusing to start on a network address without a token or with a bad certificate, upload names can't leave their workspace, job spec validation</li>
          <li>Skipped on BabelBrain versions without remote servers</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

</details>

<br>

Where practical, all unit tests should still be run because they are generally fast.

```bash
pytest Tests/Unit Tests/Consistency -m "minor_release" -k "not gpu or Metal"
```

(replace Metal with the backend of the release machine)

### 3.2 Integration Tests

Integration testing should cover the primary interactions affected by the release.

<br>

<details open>
<summary><b>Table 4: Minor Release Integration Tests</b></summary>

<table width="100%" border="1" cellpadding="6">
  <thead>
    <tr>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Test</b></th>
      <th width="40%" style="background-color:#d6001c; color:white"><b>Parameter Combination</b></th>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Reason</b></th>
    </tr>
  </thead>
  <tbody>
    <!-- test_vtk_visualization.py (all tests) -->
    <tr>
      <td>test_vtk_visualization.py (all tests)</td>
      <td>
        <b>Trajectories:</b><br>
        <ul>
          <li>Single trajectory</li>
          <li>Two trajectories (combined trajectory file)</li>
        </ul>
        <b>Test dataset:</b><br>
        <ul>
          <li>Small block volumes</li>
          <li>synthetic (Step 1 domain from the synthetic head, with and without CT and air regions)</li>
        </ul>
        Synthetic Step 1/2/3 results, no GPU or test data needed
      </td>
      <td>
        <ul>
          <li>BabelBrain's VTK visualization methods against the real viewer: opening the window after Step 1, adding/replacing Step 2 and Step 3 results per trajectory, merged results in every tab, closing and reopening</li>
          <li>Rerunning Step 1 with the viewer open refreshes it and still updates the Step 2 tab (caught the missing bFullyPopulate default fixed in 76718715)</li>
          <li>Synthetic head: CT (HU from its look-up table) and air region overlays, each step's layers and display settings, two trajectory tabs and merged fields, with screenshots of the viewer after each step in the report</li>
          <li>Skipped on BabelBrain versions without the VTK viewer</li>
        </ul>
      </td>
    </tr>
    <!-- test_remote_server_connection.py (all tests) -->
    <tr>
      <td>test_remote_server_connection.py (all tests)</td>
      <td>
        <b>Connection protocols:</b><br>
        <ul>
          <li>HTTP, HTTP + bearer token</li>
          <li>HTTPS (private CA, self-signed, insecure), HTTPS + token</li>
          <li>Mutual TLS (client certificate) + token</li>
        </ul>
        Each with the right and the wrong client settings. No GPU or test data needed
      </td>
      <td>
        <ul>
          <li>The server's real HTTP front end on a loopback port, against BabelBrain's connection check (RemoteServers.test_connection) and the remote pipeline's requests (client_functions)</li>
          <li>Right settings must connect and move data (workspace upload); wrong ones must be refused with a readable reason (missing/wrong token, untrusted certificate, protocol mismatch, missing or foreign client certificate)</li>
          <li>Uploads can't leave their workspace, every endpoint except /healthz needs the token</li>
          <li>TLS cases need openssl to make test certificates; the full pipeline on a remote server is e2e_test_remote_full_pipeline</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

</details>

<br>

Only representative combinations are required unless the release changes the interaction between specific components.

```bash
pytest Tests/Integration -m "minor_release"
```

### 3.3 End-to-End Tests

E2E testing for a minor release only covers the features affected by the release, as with the integration tests. Pick the E2E tests for those features from Table 9 (Major Release End-to-End Tests) and run them on the backend of the release machine.

<br>

<details open>
<summary><b>Table 5: Minor Release End-to-End Tests</b></summary>

<table width="100%" border="1" cellpadding="6">
  <thead>
    <tr>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Test</b></th>
      <th width="40%" style="background-color:#d6001c; color:white"><b>Parameter Combination</b></th>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Reason</b></th>
    </tr>
  </thead>
  <tbody>
    <!-- affected features -->
    <tr>
      <td>E2E tests of the features affected by the release (from Table 9)</td>
      <td>The parameter combination listed for each test in Table 9, on the backend of the release machine</td>
      <td>
        <ul>
          <li>Checks the full workflow of each changed feature</li>
          <li>Features the release doesn't affect are covered by the next major release</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

</details>

<br>

If the release doesn't affect any feature with an E2E test, no E2E tests are needed.

### 3.4 Regression Tests

Regression tests are not run for minor releases. Outputs are compared against the reference BabelBrain version in major releases only (Table 10).

<br>

<details open>
<summary><b>Table 6: Minor Release Regression Tests</b></summary>

<table width="100%" border="1" cellpadding="6">
  <thead>
    <tr>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Test</b></th>
      <th width="40%" style="background-color:#d6001c; color:white"><b>Parameter Combination</b></th>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Reason</b></th>
    </tr>
  </thead>
  <tbody>
    <!-- none -->
    <tr>
      <td>None</td>
      <td>&ndash;</td>
      <td>Regression tests are only run for major releases (Table 10)</td>
    </tr>
  </tbody>
</table>

</details>

<br>

## 4. Major Release Testing

Major releases should provide broader coverage because they typically contain larger or more widespread changes.

The goal is to test **multiple representative configurations across supported execution paths**, rather than only the default configuration.

### 4.1 Unit Tests

Major releases run every minor release unit test (Table 3), on every GPU backend instead of only the release machine's, plus the tests below, which need the real datasets or are slow.

<br>

<details open>
<summary><b>Table 7: Major Release Unit Tests</b></summary>

<table width="100%" border="1" cellpadding="6">
  <thead>
    <tr>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Test</b></th>
      <th width="40%" style="background-color:#d6001c; color:white"><b>Parameter Combination</b></th>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Reason</b></th>
    </tr>
  </thead>
  <tbody>
    <!-- test_FileManager.py (nibabel/SimpleITK conversion) -->
    <tr>
      <td>test_FileManager.py (nibabel/SimpleITK conversion: test_nibabel_to_sitk, test_sitk_to_nibabel and their _and_back round trips)</td>
      <td>
        <b>Test dataset:</b><br>
        <ul>
          <li>ID_0082</li>
          <li>ernie</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Conversions BabelBrain uses when loading the real T1W and CT scans</li>
          <li>Need the real datasets, so they're only run for major releases</li>
        </ul>
      </td>
    </tr>
    <!-- test_save_T1W_iso -->
    <tr>
      <td>test_save_T1W_iso</td>
      <td>
        <b>Test dataset:</b><br>
        <ul>
          <li>ID_0082</li>
          <li>ernie</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Isotropic T1W is the starting point of Step 1</li>
        </ul>
      </td>
    </tr>
    <!-- test_SaveNiftiEnforcedISO -->
    <tr>
      <td>test_SaveNiftiEnforcedISO</td>
      <td>
        <b>Test dataset:</b><br>
        <ul>
          <li>ID_0082</li>
          <li>ernie</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Isotropic resampling used by the transcranial modeling</li>
          <li>Skipped on BabelBrain versions before babel_integration_helpers existed</li>
        </ul>
      </td>
    </tr>
    <!-- TestDoIntersectRealData -->
    <tr>
      <td>TestDoIntersectRealData (test_BabelDatasetPreps.py)</td>
      <td>
        <b>Test dataset:</b><br>
        <ul>
          <li>ID_0082</li>
          <li>ernie (its charm meshes aren't closed, so DoIntersect's repair path runs)</li>
        </ul>
        <b>Trajectory:</b><br>
        <ul>
          <li>Superficial Target (left)</li>
          <li>Deep Target (left)</li>
        </ul>
        <b>Mesh / cutter:</b><br>
        <ul>
          <li>skin / cone, skin / box, bone / box, CSF / box (as in Step 1)</li>
          <li>skin / box outside the head (must raise "Trajectory is outside headspace")</li>
        </ul>
        <b>Boolean engine:</b><br>
        <ul>
          <li>manifold (default)</li>
          <li>blender (skipped when Blender isn't installed)</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Real charm meshes cut with a Step 1-like cone and FOV box</li>
          <li>No truth files: results must be closed, inside both inputs and agree with point-inside tests of both inputs (mesh_checks.py)</li>
          <li>About a minute in total per engine</li>
        </ul>
      </td>
    </tr>
    <!-- test_BinaryClosing_vs_CPU -->
    <tr>
      <td>test_BinaryClosing_vs_CPU<br>test_LabelImage_vs_CPU<br>test_MappingFilter_vs_CPU<br>test_MedianFilter_vs_CPU<br>test_Resample_vs_CPU<br>test_Voxelize_vs_CPU</td>
      <td>
        <b>Computing Backend:</b><br>
        <ul>
          <li>CUDA</li>
          <li>OpenCL</li>
          <li>Metal</li>
        </ul>
        <b>Test dataset / Spatial Step:</b><br>
        <ul>
          <li>synthetic: Low_Res, Med_Res, High_Res, Stress_Res (as in Table 3, but on every backend)</li>
          <li>ID_0082: Med_Res</li>
          <li>ernie: Med_Res (tests needing a CT or T1W.nii.gz in the dataset folder are skipped)</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Covers every GPU backend and resolution, higher resolutions exercise large arrays and memory limits</li>
          <li>All resolutions run on the small synthetic head, the full head datasets only at Med_Res as their high resolutions take very long (RELEASE_SPATIAL_STEPS in parameters.py; the other real dataset resolutions are still available as unit tests)</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

</details>

<br>

Important parameterized unit tests should cover representative values, boundary cases, and known failure cases.

```bash
pytest Tests/Unit Tests/Consistency -m "major_release"
```

### 4.2 Integration Tests

Integration testing should cover the major supported component interactions.

<br>

<details open>
<summary><b>Table 8: Major Release Integration Tests</b></summary>

<table width="100%" border="1" cellpadding="6">
  <thead>
    <tr>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Test</b></th>
      <th width="40%" style="background-color:#d6001c; color:white"><b>Parameter Combination</b></th>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Reason</b></th>
    </tr>
  </thead>
  <tbody>
    <!-- test_vtk_visualization.py (all tests) -->
    <tr>
      <td>test_vtk_visualization.py (all tests)</td>
      <td>
        <b>Trajectories:</b><br>
        <ul>
          <li>Single trajectory</li>
          <li>Two trajectories (combined trajectory file)</li>
        </ul>
        <b>Test dataset:</b><br>
        <ul>
          <li>Small block volumes</li>
          <li>synthetic (Step 1 domain from the synthetic head, with and without CT and air regions)</li>
        </ul>
        Synthetic Step 1/2/3 results, no GPU or test data needed
      </td>
      <td>
        <ul>
          <li>BabelBrain's VTK visualization methods against the real viewer: opening the window after Step 1, adding/replacing Step 2 and Step 3 results per trajectory, merged results in every tab, closing and reopening</li>
          <li>Rerunning Step 1 with the viewer open refreshes it and still updates the Step 2 tab (caught the missing bFullyPopulate default fixed in 76718715)</li>
          <li>Synthetic head: CT (HU from its look-up table) and air region overlays, each step's layers and display settings, two trajectory tabs and merged fields, with screenshots of the viewer after each step in the report</li>
          <li>Skipped on BabelBrain versions without the VTK viewer</li>
        </ul>
      </td>
    </tr>
    <!-- test_plantus.py (all tests) -->
    <tr>
      <td>test_plantus.py (all tests)</td>
      <td>
        <b>Test dataset:</b><br>
        <ul>
          <li>ernie</li>
        </ul>
        <b>Transducer/Frequency/Trajectory:</b><br>
        <ul>
          <li>CTX_500/500kHz/Superficial Target (left), Brainsight</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>The real PlanTUS run from BabelBrain (RUN_PLAN_TUS.Execute, RunPlanTUSBackground, the bundled run script and SimNIBS): skin maps for the target, reloading existing results, the placement for the best scoring vertex, the _PlanTUS trajectory BabelBrain continues with, and BabelBrain's PlanTUS viewers on the outputs</li>
          <li>PlanTUS' own viewer is skipped (--skip_viewer) and the placement it would hand to BabelBrain is taken directly</li>
          <li>PlanTUS runs on the CPU, so it doesn't depend on a GPU or computing backend. Slow (~2 min); requires simnibs_root_path in config.ini, skipped on BabelBrain versions without PlanTUS</li>
        </ul>
      </td>
    </tr>
    <!-- test_remote_server_connection.py (all tests) -->
    <tr>
      <td>test_remote_server_connection.py (all tests)</td>
      <td>
        <b>Connection protocols:</b><br>
        <ul>
          <li>HTTP, HTTP + bearer token</li>
          <li>HTTPS (private CA, self-signed, insecure), HTTPS + token</li>
          <li>Mutual TLS (client certificate) + token</li>
        </ul>
        Each with the right and the wrong client settings. No GPU or test data needed
      </td>
      <td>
        <ul>
          <li>The server's real HTTP front end on a loopback port, against BabelBrain's connection check (RemoteServers.test_connection) and the remote pipeline's requests (client_functions)</li>
          <li>Right settings must connect and move data (workspace upload); wrong ones must be refused with a readable reason (missing/wrong token, untrusted certificate, protocol mismatch, missing or foreign client certificate)</li>
          <li>Uploads can't leave their workspace, every endpoint except /healthz needs the token</li>
          <li>TLS cases need openssl to make test certificates; the full pipeline on a remote server is e2e_test_remote_full_pipeline</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

</details>

<br>

Additional combinations should be included when parameter interactions are likely to affect component behavior.

```bash
pytest Tests/Integration -m "major_release"
```

### 4.3 End-to-End Tests

Major releases should exercise the main supported workflows across representative configurations.

<br>

<details open>
<summary><b>Table 9: Major Release End-to-End Tests</b></summary>

<table width="100%" border="1" cellpadding="6">
  <thead>
    <tr>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Test</b></th>
      <th width="40%" style="background-color:#d6001c; color:white"><b>Parameter Combination</b></th>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Reason</b></th>
    </tr>
  </thead>
  <tbody>
    <!-- e2e_test_remote_full_pipeline -->
    <tr>
      <td>e2e_test_remote_full_pipeline</td>
      <td>
        <b>Computing backend:</b><br>
        <ul>
          <li>Remote server (the [Remote] server in config.ini, otherwise a local headless BabelBrain server)</li>
        </ul>
        <b>Test dataset:</b><br>
        <ul>
          <li>ernie</li>
        </ul>
        <b>CT Type:</b><br>
        <ul>
          <li>NONE</li>
        </ul>
        <b>Thermal Profile:</b><br>
        <ul>
          <li>pytest_thermal_profile_1</li>
        </ul>
        <b>Transducer/Frequency/Trajectory:</b>
        <ul>
          <li>CTX_500/500kHz/Superficial Target (left)</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Test full workflow using remote server, on the quick configuration (QUICK_SIM_PARAMS)</li>
          <li>Don't care too much about results so doesn't need to be a regression test</li>
          <li>1 total test</li>
        </ul>
      </td>
    </tr>
    <!-- e2e_test_custom_tx_full_pipeline -->
    <tr>
      <td>e2e_test_custom_tx_full_pipeline</td>
      <td>
        <b>Computing backend:</b><br>
        <ul>
          <li>CUDA</li>
          <li>OpenCL</li>
          <li>Metal</li>
        </ul>
        <b>Test dataset:</b><br>
        <ul>
          <li>ernie</li>
        </ul>
        <b>CT Type:</b><br>
        <ul>
          <li>NONE</li>
        </ul>
        <b>Thermal Profile:</b><br>
        <ul>
          <li>pytest_thermal_profile_1</li>
        </ul>
        <b>Custom Tx template → built-in Transducer/Frequency:</b>
        <ul>
          <li>simple_focused → Single/500kHz</li>
          <li>flat_annular_array → H246/500kHz</li>
          <li>focused_annular_array → CTX_500/500kHz</li>
          <li>flat_array_2D → TheraPhas/490kHz</li>
          <li>focused_array → H317/250kHz</li>
        </ul>
        <b>Trajectory:</b>
        <ul>
          <li>Superficial Target (left), Slicer</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Tests the custom transducer feature: a custom transducer is created from each template in custom_tx_templates/ and its full pipeline outputs are compared with the built-in transducer it replicates (CUSTOM_TX_EQUIVALENTS)</li>
          <li>Slicer trajectories for every pair, so TheraPhas isn't flipped and compares 1 to 1 with its custom version</li>
          <li>Both runs are saved in &lt;gen_output_folder&gt;/e2e_custom_tx_full_pipeline/ and reused on the same BabelBrain commit</li>
          <li>15 total tests (5 templates × 3 backends)</li>
        </ul>
      </td>
    </tr>
    <!-- e2e_test_multipoint_full_pipeline -->
    <tr>
      <td>e2e_test_multipoint_full_pipeline</td>
      <td>
        <b>Computing backend:</b><br>
        <ul>
          <li>CUDA</li>
          <li>OpenCL</li>
          <li>Metal</li>
        </ul>
        <b>Test dataset:</b><br>
        <ul>
          <li>ernie</li>
        </ul>
        <b>CT Type:</b><br>
        <ul>
          <li>NONE</li>
        </ul>
        <b>Multipoint Profile:</b><br>
        <ul>
          <li>MultiFocus_Profile1</li>
        </ul>
        <b>Thermal Profile:</b><br>
        <ul>
          <li>pytest_thermal_profile_1</li>
        </ul>
        <b>Transducer/Frequency/Trajectory:</b>
        <ul>
          <li>H317/250kHz/Superficial Target (left)</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Tests multipoint focusing on the quick configuration (QUICK_SIM_PARAMS), so it runs much faster than the intensive sim</li>
          <li>Multipoint profiles only work with phased arrays (transducers with steering), so it uses H317 at its lowest frequency instead of the quick configuration's CTX_500 (an annular array, which can only steer along its axis)</li>
          <li>Only checks that the run completes and writes thermal outputs, so it doesn't need reference outputs</li>
          <li>3 total tests</li>
        </ul>
      </td>
    </tr>
    <!-- e2e_test_combined_trajectories_full_pipeline -->
    <tr>
      <td>e2e_test_combined_trajectories_full_pipeline</td>
      <td>
        <b>Computing backend:</b><br>
        <ul>
          <li>CUDA</li>
          <li>OpenCL</li>
          <li>Metal</li>
        </ul>
        <b>Test dataset:</b><br>
        <ul>
          <li>ernie</li>
        </ul>
        <b>CT Type:</b><br>
        <ul>
          <li>NONE</li>
        </ul>
        <b>Thermal Profile:</b><br>
        <ul>
          <li>pytest_thermal_profile_1</li>
        </ul>
        <b>Transducer/Frequency/Trajectory:</b>
        <ul>
          <li>CTX_500/500kHz/Superficial Targets (combined left and right)</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Tests combined trajectories on the quick configuration (QUICK_SIM_PARAMS): steps 2 and 3 run on each trajectory tab, then Combine Trajectories merges them</li>
          <li>Checks that each trajectory and the merged result (Merged_NORM.nii.gz, merged thermal output) are written</li>
          <li>3 total tests</li>
        </ul>
      </td>
    </tr>
    <!-- e2e_test_vtk_visualization_full_pipeline -->
    <tr>
      <td>e2e_test_vtk_visualization_full_pipeline</td>
      <td>
        <b>Computing backend:</b><br>
        <ul>
          <li>CUDA</li>
          <li>OpenCL</li>
          <li>Metal</li>
        </ul>
        <b>Test dataset:</b><br>
        <ul>
          <li>ernie</li>
        </ul>
        <b>CT Type:</b><br>
        <ul>
          <li>NONE</li>
        </ul>
        <b>Thermal Profile:</b><br>
        <ul>
          <li>pytest_thermal_profile_1</li>
        </ul>
        <b>Transducer/Frequency/Trajectory:</b>
        <ul>
          <li>CTX_500/500kHz/Superficial Target (left)</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Tests the VTK visualization on the quick configuration (QUICK_SIM_PARAMS): the viewer is opened after step 1, so steps 2 and 3 push their results into it as in the GUI</li>
          <li>Checks that each step's layers hold the same data BabelBrain keeps for its plots, with its display settings, and that closing and reopening the viewer rebuilds every layer</li>
          <li>Screenshots of the viewer and the GUI after each step in the report</li>
          <li>3 total tests</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

</details>

<br>

The intent is not necessarily to test every permutation, but to cover each major execution path at least once.

### 4.4 Regression Tests

Regression testing for a major release should provide broader coverage than for a minor release. Any parameters not specified under the Parameter Combination assume default values. Reference outputs are generated with the reference BabelBrain version in Table 1, before the release run and for every backend the release is tested on; see [Regression tests and reference outputs](README.md#regression-tests-and-reference-outputs) in the README for how to generate them, how the tests use them and the pitfalls.

<br>

<details open>
<summary><b>Table 10: Major Release Regression Tests</b></summary>

<table width="100%" border="1" cellpadding="6">
  <thead>
    <tr>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Test</b></th>
      <th width="40%" style="background-color:#d6001c; color:white"><b>Parameter Combination</b></th>
      <th width="30%" style="background-color:#d6001c; color:white"><b>Reason</b></th>
    </tr>
  </thead>
  <tbody>
    <!-- regression_test_full_pipeline -->
    <tr>
      <td>regression_test_full_pipeline</td>
      <td>
        <b>Computing backend:</b><br>
        <ul>
          <li>CUDA</li>
          <li>OpenCL</li>
          <li>Metal</li>
        </ul>
        <b>Test dataset:</b><br>
        <ul>
          <li>Ernie</li>
        </ul>
        <b>CT Type:</b><br>
        <ul>
          <li>NONE</li>
        </ul>
        <b>Transducer/Frequency/Trajectory:</b>
        <ul>
          <li>Single (simple focused type)/500kHz/Superficial</li>
          <li>CTX_500 (focused annular array)/500kHz/Superficial</li>
          <li>H246 (flat annular array)/500kHz/Superficial</li>
          <li>TheraPhas (flat array 2D)/490kHz/Superficial</li>
          <li>H317 (focused array)/250kHz/Superficial</li>
          <li>DomeTx (focused array)/220kHz/Superficial</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Tests the 5 general transducer types (6 with DomeTx) for basic case</li>
          <li>Ernie is messy data, that is more likely to trigger errors, better for testing</li>
          <li>18 total tests</li>
        </ul>
      </td>
    </tr>
    <!-- regression_test_full_pipeline_intensive -->
    <tr>
      <td>regression_test_full_pipeline_intensive</td>
      <td>
        <b>Computing backend:</b><br>
        <ul>
          <li>CUDA</li>
          <li>OpenCL</li>
          <li>Metal</li>
        </ul>
        <b>Test dataset:</b><br>
        <ul>
          <li>ID_0082</li>
        </ul>
        <b>CT Type:</b><br>
        <ul>
          <li>CT</li>
        </ul>
        <b>Multipoint:</b><br>
        <ul>
          <li>No</li>
        </ul>
        <b>Thermal Profile:</b><br>
        <ul>
          <li>pytest_thermal_profile_2</li>
        </ul>
        <b>Transducer/Frequency/Trajectory:</b>
        <ul>
          <li>H317/825kHz/Deep Target</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Tests more intensive sim</li>
          <li>3 total tests</li>
        </ul>
      </td>
    </tr>
    <!-- regression_test_full_pipeline_soft_tissue -->
    <tr>
      <td>regression_test_full_pipeline_soft_tissue</td>
      <td>
        <b>Computing backend:</b><br>
        <ul>
          <li>CUDA</li>
          <li>OpenCL</li>
          <li>Metal</li>
        </ul>
        <b>Test dataset:</b><br>
        <ul>
          <li>Ernie</li>
        </ul>
        <b>CT Type:</b><br>
        <ul>
          <li>NONE</li>
        </ul>
        <b>Transducer/Frequency/Trajectory:</b>
        <ul>
          <li>CTX_500/500kHz/Superficial</li>
        </ul>
        <b>Advanced Options – Step 1 soft tissue segmentation:</b>
        <ul>
          <li>Enabled</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Tests full pipeline with soft tissue segmentation</li>
          <li>Requires SimNIBS</li>
          <li>3 total tests</li>
        </ul>
      </td>
    </tr>
    <!-- regression_test_GetSkullMaskFromSimbNIBSSTL -->
    <tr>
      <td>regression_test_GetSkullMaskFromSimbNIBSSTL</td>
      <td>
        <b>Computing backend:</b><br>
        <ul>
          <li>CUDA</li>
          <li>OpenCL</li>
          <li>Metal</li>
        </ul>
        <b>Test dataset:</b><br>
        <ul>
          <li>ID_0082</li>
        </ul>
        <b>CT Type:</b><br>
        <ul>
          <li>NONE</li>
          <li>CT</li>
          <li>Density</li>
          <li>Petra</li>
          <li>ZTE</li>
        </ul>
        <b>Transducer/Frequency/Trajectory:</b>
        <ul>
          <li>CTX_500/500kHz/Superficial</li>
        </ul>
        <b>Pipeline:</b>
        <ul>
          <li>Step 1 only</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Tests Step 1's skull mask for every CT type, on the quick configuration (QUICK_SIM_PARAMS) with ID_0082 instead of ernie, which has no CT, ZTE, PETRA or Density scans</li>
          <li>Soft tissue segmentation isn't varied here, regression_test_full_pipeline_soft_tissue covers it</li>
          <li>Compare all generated output files</li>
          <li>15 total tests (CT types the dataset doesn't have are skipped)</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

</details>

<br>

Major releases should include known regression cases and representative coverage of all important supported configurations.

## 5. Release Testing Summary

<br>

<details open>
<summary><b>Table 11: Release Testing Summary</b></summary>

<table width="100%" border="1" cellpadding="6">
  <thead>
    <tr>
      <th width="16%" style="background-color:#d6001c; color:white"><b>Test Type</b></th>
      <th width="42%" style="background-color:#d6001c; color:white"><b>Minor Release</b></th>
      <th width="42%" style="background-color:#d6001c; color:white"><b>Major Release</b></th>
    </tr>
  </thead>
  <tbody>
    <!-- Unit -->
    <tr>
      <td>Unit</td>
      <td>Changed areas and important parameter cases</td>
      <td>All applicable tests with broader parameter coverage</td>
    </tr>
    <!-- Integration -->
    <tr>
      <td>Integration</td>
      <td>Primary affected interactions</td>
      <td>Major supported component combinations</td>
    </tr>
    <!-- E2E -->
    <tr>
      <td>E2E</td>
      <td>Only the features affected by the release</td>
      <td>Multiple representative supported workflows</td>
    </tr>
    <!-- Regression -->
    <tr>
      <td>Regression</td>
      <td>Not run</td>
      <td>Broad representative coverage and known regression cases</td>
    </tr>
  </tbody>
</table>

</details>

<br>

## 6. Selecting Parameter Combinations

Parameter combinations should generally prioritize:

- Default or most commonly used configurations.
- Functionality changed by the release.
- Important alternate execution paths.
- Different supported backends or hardware where applicable.
- Boundary or edge-case values.
- Previously failing configurations.

Testing every permutation is generally unnecessary unless there is evidence that parameters interact in a way that could affect results.

## 7. Adding New Test Combinations

A new test combination should be added when:

- A bug is discovered for a specific configuration.
- A new parameter or supported value is introduced.
- A new backend, device, or execution path is added.
- Existing combinations do not adequately cover changed functionality.
- An interaction between two or more parameters is found to affect behavior.

Each added combination should include a short explanation of why it is necessary.

## 8. Pass Criteria

A release passes testing when:

- All required test combinations complete successfully.
- Numerical or output comparisons remain within defined tolerances.
- Previously working functionality does not unexpectedly change.
- Any expected differences from previous results are reviewed and documented.

