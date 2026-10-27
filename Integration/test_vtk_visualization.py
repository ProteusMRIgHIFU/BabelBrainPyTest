"""
Integration tests for BabelBrain's VTK visualization window (BabelBrain_Pytest_Usage.md, Tables 4
and 8): BabelBrain's own methods that open the viewer and push each step's results into it, run
against the real viewer (GUIComponents/nifti_viewer.NiftiViewerWindow).

The steps themselves aren't run. A small harness holds the per-trajectory state BabelBrain keeps
(Config['ID'], _MaskNib, _NiftiSkull, ...) and borrows BabelBrain's methods, so the tests feed it
synthetic Step 1/2/3 results the way the GUI does when each step finishes:

    Step 1  GenerateMask resets the results, then UpdateMask runs for each finished trajectory
    Step 2  UpdateNiftiAcResults (and UpdateNiftiMergedAcResults after Combine Trajectories)
    Step 3  UpdateNiftiTemperatureResults (and UpdateNiftiMergedThermalResults)

Only the parts unrelated to the viewer (Step 1 plots, the Step 2 tab, the clock dialog) are stubbed,
so no GPU or test data is needed.
"""
import functools
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import nibabel
import numpy as np
import pytest

from Tests.helpers.parameters import requires_feature

pytestmark = [requires_feature('vtk_viewer'), pytest.mark.minor_release, pytest.mark.major_release]

SHAPE = (24, 28, 32)
AFFINE = np.diag([0.5, 0.5, 0.5, 1.0])

# BabelBrain methods taking part in the VTK visualization
VTK_METHODS = [
    'UpdateMask', 'OpenVTKVisualization', '_FindViewerRow', '_UpdateVTKDomain',
    'UpdateNiftiAcResults', '_UpdateVTKAcResults',
    'UpdateNiftiTemperatureResults', '_UpdateVTKThermal',
    'UpdateNiftiMergedAcResults', '_UpdateVTKMergedAcResults',
    'UpdateNiftiMergedThermalResults', '_UpdateVTKMergedThermalResults',
    '_closingVtkVisualization',
]
# What each BabelBrain entry point stands for, logged when a test calls it
LOGGED_METHODS = {
    'UpdateMask': 'Step 1 finished a trajectory',
    'OpenVTKVisualization': '"VTK visualization" clicked',
    'UpdateNiftiAcResults': 'Step 2 results',
    'UpdateNiftiTemperatureResults': 'Step 3 results',
    'UpdateNiftiMergedAcResults': 'Step 2 Combine Trajectories results',
    'UpdateNiftiMergedThermalResults': 'Step 3 Combine Trajectories results',
}


# ================================================================================================================================
# HARNESS
# ================================================================================================================================
def _image(data):
    return nibabel.Nifti1Image(data.astype(np.float32), AFFINE)


def _mask(target):
    """Step 1 mask: scalp, skull and brain labels with the focal point voxel (label 5) at target."""
    data = np.zeros(SHAPE)
    data[2:-2, 2:-2, 2:-2] = 1
    data[4:-4, 4:-4, 4:-4] = 2
    data[6:-6, 6:-6, 6:-6] = 4
    data[target] = 5
    return _image(data)


def _field(peak):
    """Normalised field peaking at `peak` (Step 2 pressure / Step 3 intensity)."""
    grid = np.indices(SHAPE)
    dist2 = sum((g - p) ** 2 for g, p in zip(grid, peak))
    return _image(np.exp(-dist2 / 20.))


def _layers(viewer):
    """Result ids shown by a trajectory viewer (the base tissue mask has none)."""
    return [row._id for row in viewer._layer_panel._rows]


def _logged(name, method):
    """BabelBrain `method` that logs the call and then the layers of every open trajectory viewer."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        trajectory = args[-1] if name in ('UpdateNiftiAcResults', 'UpdateNiftiTemperatureResults') \
            else self._TrajectoryNumber if name == 'UpdateMask' else None
        logging.info(f"BabelBrain.{name}: {LOGGED_METHODS[name]}"
                     f"{f' (trajectory {trajectory})' if trajectory is not None else ''}")
        result = method(self, *args, **kwargs)
        if self.viewer_open:
            for n, viewer in enumerate(self._vtk_visualization.viewer):
                layers = ['base' if i == '' else i for i in _layers(viewer)]
                logging.info(f"  viewer {n} layers: {layers}, slices {[vp._slider.value() for vp in viewer._vps]}")
        return result
    return wrapper


def _visible(viewer, layer_id):
    return viewer._volumes[_layers(viewer).index(layer_id)].visible


class _Harness:
    """Per-trajectory state of a BabelBrain window plus its VTK visualization methods."""

    def __init__(self, qtbot, tmp_path, ids):
        from PySide6.QtWidgets import QPushButton, QTabWidget

        self._qtbot = qtbot
        self._tmp_path = tmp_path
        self.Config = {'ID': list(ids), 'bUseCT': False, 'bExtractAirRegions': False}
        n = len(ids)
        self._prefix_path = [str(tmp_path / f"{traj}_") for traj in ids]
        self._outnameMask = [p + 'BabelViscoInput.nii.gz' for p in self._prefix_path]
        self._T1W_resampled_fname = [p + 'T1W_Resampled.nii.gz' for p in self._prefix_path]
        self._TrajectoryNumber = 0

        tabs, button = QTabWidget(), QPushButton()
        qtbot.addWidget(tabs)
        qtbot.addWidget(button)
        self.Widget = SimpleNamespace(tabWidget=tabs, vtkVisualizationqPushButton=button)
        self.AcSim = MagicMock()
        self.hideClockDialog = MagicMock()
        self._showMatplotlibVisualization = MagicMock()
        self.UpdateAcousticTab = MagicMock()
        self._reset_results(n)

    def _reset_results(self, n):
        """Per-trajectory results cleared by GenerateMask when Step 1 starts."""
        for name in ['_MaskNib', '_T1WNib', '_NiftiCT', '_NiftiAirMask', 'FinalMaskRaw', '_FinalMask',
                     '_T1WDataRaw', '_NiftiSkull', '_NiftiWater', '_NiftiIntensity', '_NiftiTemperature']:
            setattr(self, name, [None] * n)

    def run_step1(self, targets):
        """Step 1: GenerateMask resets the results, then each trajectory's mask is saved and
        UpdateMask runs when it finishes."""
        logging.info(f"Step 1 on {len(targets)} trajectory(ies), focal point voxels {targets}")
        self._reset_results(len(targets))
        for n, target in enumerate(targets):
            nibabel.save(_mask(target), self._outnameMask[n])
            nibabel.save(_field(target), self._T1W_resampled_fname[n])
            self._TrajectoryNumber = n
            self.UpdateMask()

    def open_viewer(self):
        """Click "VTK visualization"."""
        self.OpenVTKVisualization()
        self._qtbot.addWidget(self._vtk_visualization)
        return self._vtk_visualization

    @property
    def viewer_open(self):
        return hasattr(self, '_vtk_visualization')


@pytest.fixture
def harness_factory(qtbot, tmp_path):
    from BabelBrain.BabelBrain import BabelBrain

    for name in VTK_METHODS:
        method = getattr(BabelBrain, name)
        setattr(_Harness, name, _logged(name, method) if name in LOGGED_METHODS else method)
    created = []

    def _make(ids=('traj',)):
        harness = _Harness(qtbot, tmp_path, ids)
        created.append(harness)
        return harness

    yield _make
    for harness in created:
        if harness.viewer_open:
            harness._vtk_visualization.close()


@pytest.fixture
def harness(harness_factory):
    """Single trajectory window that finished Step 1."""
    h = harness_factory()
    h.run_step1([(12, 14, 16)])
    return h


# ================================================================================================================================
# TESTS
# ================================================================================================================================
def test_viewer_button_enabled_after_last_trajectory(harness_factory):
    h = harness_factory(['left', 'right'])
    h.Widget.vtkVisualizationqPushButton.setEnabled(False)
    h._reset_results(2)
    nibabel.save(_mask((10, 10, 10)), h._outnameMask[0])
    nibabel.save(_field((10, 10, 10)), h._T1W_resampled_fname[0])
    h._TrajectoryNumber = 0
    h.UpdateMask()
    assert not h.Widget.vtkVisualizationqPushButton.isEnabled()
    h.run_step1([(10, 10, 10), (14, 18, 20)])
    assert h.Widget.vtkVisualizationqPushButton.isEnabled()


def test_open_viewer_shows_step1_domain(harness):
    window = harness.open_viewer()
    viewer = window.viewer[0]
    assert _layers(viewer) == ['', 'T1W']
    assert viewer._volumes[0].cmap == 'TissueLabel'
    assert viewer._volumes[1].opacity == pytest.approx(0.5)
    # Opens on the focal point voxel (label 5 of the mask)
    assert [vp._slider.value() for vp in viewer._vps] == [14, 12, 16]


def test_open_viewer_twice_reuses_window(harness):
    window = harness.open_viewer()
    harness.OpenVTKVisualization()
    assert harness._vtk_visualization is window
    assert _layers(window.viewer[0]) == ['', 'T1W']


def test_one_tab_per_trajectory_at_its_own_target(harness_factory):
    h = harness_factory(['left', 'right'])
    h.run_step1([(8, 10, 12), (16, 18, 20)])
    window = h.open_viewer()
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == ['left', 'right']
    assert [vp._slider.value() for vp in window.viewer[0]._vps] == [10, 8, 12]
    assert [vp._slider.value() for vp in window.viewer[1]._vps] == [18, 16, 20]


def test_step2_results_added_and_replaced(harness):
    window = harness.open_viewer()
    viewer = window.viewer[0]
    harness.UpdateNiftiAcResults(_field((12, 14, 16)), _field((12, 14, 18)), 0)
    assert _layers(viewer) == ['', 'T1W', 'Skull', 'Water']
    assert _visible(viewer, 'Skull') and not _visible(viewer, 'Water')
    assert viewer._volumes[2].cutoff == 0.25

    # Rerunning Step 2 replaces the previous fields instead of stacking new ones
    harness.UpdateNiftiAcResults(_field((12, 14, 17)), _field((12, 14, 19)), 0)
    assert _layers(viewer) == ['', 'T1W', 'Skull', 'Water']
    assert all(len(vp._layers) == 4 for vp in viewer._vps)


def test_step3_results_hide_step2_fields(harness):
    window = harness.open_viewer()
    viewer = window.viewer[0]
    harness.UpdateNiftiAcResults(_field((12, 14, 16)), _field((12, 14, 18)), 0)
    harness.UpdateNiftiTemperatureResults(_field((12, 14, 16)), _image(37 + 2 * _field((12, 14, 16)).get_fdata()), 0)
    assert _layers(viewer) == ['', 'T1W', 'Skull', 'Water', 'Intensity', 'Temperature']
    assert not _visible(viewer, 'Skull') and not _visible(viewer, 'Water')
    assert _visible(viewer, 'Intensity') and not _visible(viewer, 'Temperature')
    assert viewer._volumes[_layers(viewer).index('Temperature')].cutoff == 37.05

    harness.UpdateNiftiTemperatureResults(_field((12, 14, 17)), _image(37 + _field((12, 14, 17)).get_fdata()), 0)
    assert _layers(viewer).count('Intensity') == 1 and _layers(viewer).count('Temperature') == 1


def test_results_only_go_to_their_trajectory(harness_factory):
    h = harness_factory(['left', 'right'])
    h.run_step1([(8, 10, 12), (16, 18, 20)])
    window = h.open_viewer()
    h.UpdateNiftiAcResults(_field((16, 18, 20)), _field((16, 18, 22)), 1)
    assert _layers(window.viewer[0]) == ['', 'T1W']
    assert _layers(window.viewer[1]) == ['', 'T1W', 'Skull', 'Water']


def test_merged_results_go_to_every_trajectory(harness_factory):
    h = harness_factory(['left', 'right'])
    h.run_step1([(8, 10, 12), (16, 18, 20)])
    window = h.open_viewer()
    for n, peak in enumerate([(8, 10, 12), (16, 18, 20)]):
        h.UpdateNiftiAcResults(_field(peak), _field(peak), n)

    merged = _field((12, 14, 16))
    h.UpdateNiftiMergedAcResults(merged)
    h.UpdateNiftiMergedAcResults(merged)   # recombining replaces the merged field
    for viewer in window.viewer:
        assert _layers(viewer) == ['', 'T1W', 'Skull', 'Water', 'MergedAc']
        assert not _visible(viewer, 'Skull') and _visible(viewer, 'MergedAc')

    h.UpdateNiftiMergedThermalResults(merged, _image(37 + merged.get_fdata()))
    for viewer in window.viewer:
        assert _layers(viewer)[-2:] == ['MergedIntensity', 'MergedTemperature']
        assert not _visible(viewer, 'MergedAc') and _visible(viewer, 'MergedIntensity')


def test_reopening_viewer_shows_existing_results(harness):
    harness.UpdateNiftiAcResults(_field((12, 14, 16)), _field((12, 14, 18)), 0)   # viewer closed: only stored
    harness.UpdateNiftiTemperatureResults(_field((12, 14, 16)), _image(37 + _field((12, 14, 16)).get_fdata()), 0)
    window = harness.open_viewer()
    assert _layers(window.viewer[0]) == ['', 'T1W', 'Skull', 'Water', 'Intensity', 'Temperature']


def test_closing_viewer_releases_it(harness):
    window = harness.open_viewer()
    window.close()
    assert not harness.viewer_open
    # Results arriving after the window was closed are only stored
    harness.UpdateNiftiAcResults(_field((12, 14, 16)), _field((12, 14, 18)), 0)
    assert harness._NiftiSkull[0] is not None
    window = harness.open_viewer()
    assert 'Skull' in _layers(window.viewer[0])


def test_rerunning_step1_with_viewer_open(harness):
    """Step 1 finishing while the viewer is open must refresh the viewer with the new domain and
    still update the Step 2 tab (UpdateMask continues after the viewer update)."""
    window = harness.open_viewer()
    harness.UpdateNiftiAcResults(_field((12, 14, 16)), _field((12, 14, 18)), 0)
    harness.UpdateAcousticTab.reset_mock()

    harness.run_step1([(10, 18, 20)])

    viewer = window.viewer[0]
    harness.UpdateAcousticTab.assert_called_once()
    # New domain, at the new target, without the Step 2 results of the previous domain
    assert _layers(viewer) == ['', 'T1W']
    assert [vp._slider.value() for vp in viewer._vps] == [18, 10, 20]


def test_rerunning_step1_with_viewer_open_multiple_trajectories(harness_factory):
    """Every trajectory of a combined trajectory file is redone and shown again."""
    h = harness_factory(['left', 'right'])
    h.run_step1([(8, 10, 12), (16, 18, 20)])
    window = h.open_viewer()
    h.run_step1([(9, 11, 13), (15, 17, 19)])
    assert h.UpdateAcousticTab.call_count == 4
    assert [vp._slider.value() for vp in window.viewer[0]._vps] == [11, 9, 13]
    assert [vp._slider.value() for vp in window.viewer[1]._vps] == [17, 15, 19]
