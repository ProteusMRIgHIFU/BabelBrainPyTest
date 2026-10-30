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
so no GPU or test data is needed. Most tests use small block volumes; the synthetic head tests use
the synthetic dataset's tissue mask, T1W and CT (with BabelBrain's CT and air region overlays) and
put screenshots of the viewer after each step in the report. The real pipeline with the viewer
open is e2e_test_vtk_visualization_full_pipeline (E2E/test_e2e_release.py).
"""
import functools
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import nibabel
import numpy as np
import pytest

from Tests.helpers import vtk_viewer_helpers as vh
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

        logging.info(f"Creating a BabelBrain harness for trajectories {list(ids)} (Step 2 tab and clock stubbed)")
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

    def run_step1_synthetic(self, targets, ct=False, air=False):
        """Step 1 on the synthetic head (one focal point per trajectory), saving what Step 1 writes
        with a CT: <prefix>CT.nii.gz (HU look-up indices), CT-cal.npz and AirRegions.nii.gz."""
        logging.info(f"Step 1 on the synthetic head, {len(targets)} trajectory(ies), focal point voxels {targets}"
                     f"{', with CT' if ct else ''}{' and air regions' if air else ''}")
        self.Config['bUseCT'] = ct
        self.Config['bExtractAirRegions'] = air
        self._reset_results(len(targets))
        for n, target in enumerate(targets):
            nibabel.save(vh.synthetic_mask(target), self._outnameMask[n])
            nibabel.save(vh.synthetic_t1(), self._T1W_resampled_fname[n])
            if ct:
                indices, unique_hu = vh.synthetic_ct_indices()
                nibabel.save(indices, self._prefix_path[n] + 'CT.nii.gz')
                np.savez(self._prefix_path[n] + 'CT-cal.npz', UniqueHU=unique_hu)
            if air:
                nibabel.save(vh.synthetic_air_mask(), self._prefix_path[n] + 'AirRegions.nii.gz')
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

    logging.info(f"Borrowing BabelBrain's {len(VTK_METHODS)} VTK visualization methods for the harness")
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
            logging.info("Closing the viewer window left open by the test")
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
    logging.info("Disabling the VTK visualization button and finishing Step 1 on the first trajectory only")
    h.Widget.vtkVisualizationqPushButton.setEnabled(False)
    h._reset_results(2)
    nibabel.save(_mask((10, 10, 10)), h._outnameMask[0])
    nibabel.save(_field((10, 10, 10)), h._T1W_resampled_fname[0])
    h._TrajectoryNumber = 0
    h.UpdateMask()
    logging.info("Checking the button is still disabled with a trajectory left")
    assert not h.Widget.vtkVisualizationqPushButton.isEnabled()
    h.run_step1([(10, 10, 10), (14, 18, 20)])
    logging.info("Checking the button is enabled once the last trajectory finished")
    assert h.Widget.vtkVisualizationqPushButton.isEnabled()


def test_open_viewer_shows_step1_domain(harness):
    window = harness.open_viewer()
    viewer = window.viewer[0]
    logging.info("Checking the Step 1 domain: tissue colours base, T1W overlay at 50 %")
    assert _layers(viewer) == ['', 'T1W']
    assert viewer._volumes[0].cmap == 'TissueLabel'
    assert viewer._volumes[1].opacity == pytest.approx(0.5)
    logging.info("Checking the viewer opens on the focal point voxel (12, 14, 16)")
    # Opens on the focal point voxel (label 5 of the mask)
    assert [vp._slider.value() for vp in viewer._vps] == [14, 12, 16]


def test_open_viewer_twice_reuses_window(harness):
    window = harness.open_viewer()
    logging.info("Clicking VTK visualization again with the window open")
    harness.OpenVTKVisualization()
    logging.info("Checking the same window is reused, without duplicate layers")
    assert harness._vtk_visualization is window
    assert _layers(window.viewer[0]) == ['', 'T1W']


def test_one_tab_per_trajectory_at_its_own_target(harness_factory):
    h = harness_factory(['left', 'right'])
    h.run_step1([(8, 10, 12), (16, 18, 20)])
    window = h.open_viewer()
    logging.info("Checking there is one tab per trajectory, each on its own focal point")
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == ['left', 'right']
    assert [vp._slider.value() for vp in window.viewer[0]._vps] == [10, 8, 12]
    assert [vp._slider.value() for vp in window.viewer[1]._vps] == [18, 16, 20]


def test_step2_results_added_and_replaced(harness):
    window = harness.open_viewer()
    viewer = window.viewer[0]
    harness.UpdateNiftiAcResults(_field((12, 14, 16)), _field((12, 14, 18)), 0)
    logging.info("Checking Skull and Water are added, Skull shown with a 0.25 cutoff, Water hidden")
    assert _layers(viewer) == ['', 'T1W', 'Skull', 'Water']
    assert _visible(viewer, 'Skull') and not _visible(viewer, 'Water')
    assert viewer._volumes[2].cutoff == 0.25

    logging.info("Rerunning Step 2 with shifted fields")
    # Rerunning Step 2 replaces the previous fields instead of stacking new ones
    harness.UpdateNiftiAcResults(_field((12, 14, 17)), _field((12, 14, 19)), 0)
    logging.info("Checking the fields were replaced, not added again")
    assert _layers(viewer) == ['', 'T1W', 'Skull', 'Water']
    assert all(len(vp._layers) == 4 for vp in viewer._vps)


def test_step3_results_hide_step2_fields(harness):
    window = harness.open_viewer()
    viewer = window.viewer[0]
    harness.UpdateNiftiAcResults(_field((12, 14, 16)), _field((12, 14, 18)), 0)
    harness.UpdateNiftiTemperatureResults(_field((12, 14, 16)), _image(37 + 2 * _field((12, 14, 16)).get_fdata()), 0)
    logging.info("Checking Step 3 adds Intensity (shown) and Temperature (hidden, cutoff 37.05) and hides Step 2 fields")
    assert _layers(viewer) == ['', 'T1W', 'Skull', 'Water', 'Intensity', 'Temperature']
    assert not _visible(viewer, 'Skull') and not _visible(viewer, 'Water')
    assert _visible(viewer, 'Intensity') and not _visible(viewer, 'Temperature')
    # 37.05: just above body temperature, so only heating shows
    assert viewer._volumes[_layers(viewer).index('Temperature')].cutoff == 37.05

    logging.info("Rerunning Step 3, checking Intensity and Temperature are replaced, not duplicated")
    harness.UpdateNiftiTemperatureResults(_field((12, 14, 17)), _image(37 + _field((12, 14, 17)).get_fdata()), 0)
    assert _layers(viewer).count('Intensity') == 1 and _layers(viewer).count('Temperature') == 1


def test_results_only_go_to_their_trajectory(harness_factory):
    h = harness_factory(['left', 'right'])
    h.run_step1([(8, 10, 12), (16, 18, 20)])
    window = h.open_viewer()
    logging.info("Step 2 results for the right trajectory only")
    h.UpdateNiftiAcResults(_field((16, 18, 20)), _field((16, 18, 22)), 1)
    logging.info("Checking only the right trajectory's tab got Skull and Water")
    assert _layers(window.viewer[0]) == ['', 'T1W']
    assert _layers(window.viewer[1]) == ['', 'T1W', 'Skull', 'Water']


def test_merged_results_go_to_every_trajectory(harness_factory):
    h = harness_factory(['left', 'right'])
    h.run_step1([(8, 10, 12), (16, 18, 20)])
    window = h.open_viewer()
    logging.info("Step 2 results for both trajectories")
    for n, peak in enumerate([(8, 10, 12), (16, 18, 20)]):
        h.UpdateNiftiAcResults(_field(peak), _field(peak), n)

    logging.info("Combining trajectories twice (Step 2)")
    merged = _field((12, 14, 16))
    h.UpdateNiftiMergedAcResults(merged)
    h.UpdateNiftiMergedAcResults(merged)   # recombining replaces the merged field
    logging.info("Checking each tab has one MergedAc layer, shown, with Skull hidden")
    for viewer in window.viewer:
        assert _layers(viewer) == ['', 'T1W', 'Skull', 'Water', 'MergedAc']
        assert not _visible(viewer, 'Skull') and _visible(viewer, 'MergedAc')

    logging.info("Combining trajectories for Step 3")
    h.UpdateNiftiMergedThermalResults(merged, _image(37 + merged.get_fdata()))
    logging.info("Checking each tab shows MergedIntensity and hides MergedAc")
    for viewer in window.viewer:
        assert _layers(viewer)[-2:] == ['MergedIntensity', 'MergedTemperature']
        assert not _visible(viewer, 'MergedAc') and _visible(viewer, 'MergedIntensity')


def test_reopening_viewer_shows_existing_results(harness):
    logging.info("Step 2 and 3 results arrive with the viewer closed")
    harness.UpdateNiftiAcResults(_field((12, 14, 16)), _field((12, 14, 18)), 0)   # viewer closed: only stored
    harness.UpdateNiftiTemperatureResults(_field((12, 14, 16)), _image(37 + _field((12, 14, 16)).get_fdata()), 0)
    window = harness.open_viewer()
    logging.info("Checking the viewer opens with all stored results")
    assert _layers(window.viewer[0]) == ['', 'T1W', 'Skull', 'Water', 'Intensity', 'Temperature']


def test_closing_viewer_releases_it(harness):
    window = harness.open_viewer()
    logging.info("Closing the viewer window")
    window.close()
    logging.info("Checking BabelBrain dropped its reference to the window")
    assert not harness.viewer_open
    # Results arriving after the window was closed are only stored
    harness.UpdateNiftiAcResults(_field((12, 14, 16)), _field((12, 14, 18)), 0)
    logging.info("Checking Step 2 results were stored, and shown when the viewer is reopened")
    assert harness._NiftiSkull[0] is not None
    window = harness.open_viewer()
    assert 'Skull' in _layers(window.viewer[0])


def test_rerunning_step1_with_viewer_open(harness):
    """Step 1 finishing while the viewer is open must refresh the viewer with the new domain and
    still update the Step 2 tab (UpdateMask continues after the viewer update)."""
    window = harness.open_viewer()
    harness.UpdateNiftiAcResults(_field((12, 14, 16)), _field((12, 14, 18)), 0)
    harness.UpdateAcousticTab.reset_mock()

    logging.info("Rerunning Step 1 with the viewer open, new focal point (10, 18, 20)")
    harness.run_step1([(10, 18, 20)])

    viewer = window.viewer[0]
    logging.info("Checking the Step 2 tab was updated once")
    harness.UpdateAcousticTab.assert_called_once()
    logging.info("Checking the viewer shows the new domain at the new target, without the old Step 2 fields")
    # New domain, at the new target, without the Step 2 results of the previous domain
    assert _layers(viewer) == ['', 'T1W']
    assert [vp._slider.value() for vp in viewer._vps] == [18, 10, 20]


def test_rerunning_step1_with_viewer_open_multiple_trajectories(harness_factory):
    """Every trajectory of a combined trajectory file is redone and shown again."""
    h = harness_factory(['left', 'right'])
    h.run_step1([(8, 10, 12), (16, 18, 20)])
    window = h.open_viewer()
    logging.info("Rerunning Step 1 on both trajectories with the viewer open")
    h.run_step1([(9, 11, 13), (15, 17, 19)])
    logging.info(f"Checking the Step 2 tab was updated 4 times (2 runs x 2 trajectories), "
                 f"got {h.UpdateAcousticTab.call_count}")
    assert h.UpdateAcousticTab.call_count == 4
    logging.info("Checking each tab moved to its new focal point")
    assert [vp._slider.value() for vp in window.viewer[0]._vps] == [11, 9, 13]
    assert [vp._slider.value() for vp in window.viewer[1]._vps] == [17, 15, 19]


# ================================================================================================================================
# SYNTHETIC HEAD (screenshots in the report)
# ================================================================================================================================
LEFT, RIGHT = (14, 24, 26), (26, 16, 24)  # focal point voxels of the synthetic head's two trajectories


def _row(viewer, layer_id):
    return viewer._layer_panel._rows[_layers(viewer).index(layer_id)]


def _steps_fields(t1, peak):
    """Step 2 (skull, water) and Step 3 (intensity, temperature) results peaking at `peak`."""
    return (vh.gaussian_field(t1, peak), vh.gaussian_field(t1, peak, width=5),
            vh.gaussian_field(t1, peak, scale=0.8), vh.gaussian_field(t1, peak, scale=3, offset=37))


def test_synthetic_head_ct_and_air_overlays(harness_factory, viewer_screenshots):
    """With a CT, Step 1's CT (HU from its look-up table, bone only) and air regions are overlaid on
    the domain the way _UpdateVTKDomain sets them up."""
    h = harness_factory(['synthetic'])
    h.run_step1_synthetic([vh.SYNTHETIC_TARGET], ct=True, air=True)
    viewer = h.open_viewer().viewer[0]
    logging.info("Taking a screenshot of the Step 1 domain")
    pixels = viewer_screenshots.add(viewer, "Step 1 domain with CT and air regions")

    logging.info("Checking CT (HU from its look-up table) and Air layers hold Step 1's data")
    assert _layers(viewer) == ['', 'T1W', 'CT', 'Air']
    indices, unique_hu = vh.synthetic_ct_indices()
    np.testing.assert_allclose(vh.layer_data(viewer, 'CT'), unique_hu[np.asarray(indices.dataobj)])
    np.testing.assert_array_equal(vh.layer_data(viewer, 'Air'), np.asarray(vh.synthetic_air_mask().dataobj))
    logging.info("Checking CT and Air display settings (50 % opacity, cutoff 1, visible), and the focal point slices")
    for layer_id in ('CT', 'Air'):
        rec = vh.layer(viewer, layer_id)
        assert rec.opacity == pytest.approx(0.5) and rec.cutoff == 1.0 and rec.visible
    assert vh.layer(viewer, 'Air').cmap == _row(viewer, 'Air')._cmap_combo.itemText(4)
    assert vh.slider_values(viewer) == [vh.SYNTHETIC_TARGET[1], vh.SYNTHETIC_TARGET[0], vh.SYNTHETIC_TARGET[2]]
    assert pixels.std() > 20


def test_synthetic_head_all_steps(harness_factory, viewer_screenshots):
    """The viewer open through Steps 1 to 3 on the synthetic head: each step's results are added
    with BabelBrain's display settings, and the previous step's fields are hidden."""
    h = harness_factory(['synthetic'])
    h.run_step1_synthetic([vh.SYNTHETIC_TARGET])
    window = h.open_viewer()
    viewer = window.viewer[0]
    step1 = viewer_screenshots.add(viewer, "Step 1: tissue mask and T1W (50 %)")
    assert _layers(viewer) == ['', 'T1W']

    logging.info(f"Making Step 2/3 fields peaking at the focal point {vh.SYNTHETIC_TARGET}")
    t1 = vh.synthetic_t1()
    skull, water, intensity, temperature = _steps_fields(t1, vh.SYNTHETIC_TARGET)
    h.UpdateNiftiAcResults(skull, water, 0)
    step2 = viewer_screenshots.add(viewer, "Step 2: skull pressure (Jet, cutoff 0.25), water field hidden")
    logging.info("Checking Step 2: Skull data and visibility, window/level on the Skull layer")
    assert _layers(viewer) == ['', 'T1W', 'Skull', 'Water']
    np.testing.assert_allclose(vh.layer_data(viewer, 'Skull'), skull.get_fdata())
    assert _visible(viewer, 'Skull') and not _visible(viewer, 'Water')
    assert viewer._selected_vol == _layers(viewer).index('Skull')  # window/level follows the skull field

    h.UpdateNiftiTemperatureResults(intensity, temperature, 0)
    step3 = viewer_screenshots.add(viewer, "Step 3: intensity (cutoff 0.1), temperature hidden, Step 2 fields hidden")
    logging.info("Checking Step 3: only Intensity shown (cutoff 0.1) and selected, Temperature data stored")
    assert _layers(viewer)[-2:] == ['Intensity', 'Temperature']
    np.testing.assert_allclose(vh.layer_data(viewer, 'Temperature'), temperature.get_fdata())
    assert [_visible(viewer, i) for i in ('Skull', 'Water', 'Intensity', 'Temperature')] == [False, False, True, False]
    assert vh.layer(viewer, 'Intensity').cutoff == 0.1
    assert viewer._selected_vol == _layers(viewer).index('Intensity')

    logging.info("Showing Temperature instead of Intensity with the eye buttons")
    _row(viewer, 'Temperature')._eye_btn.setChecked(True)
    _row(viewer, 'Intensity')._eye_btn.setChecked(False)
    temperature_view = viewer_screenshots.add(viewer, "Step 3: temperature shown instead (cutoff 37.05)")

    logging.info("Checking each step changed the views around the focal point only")
    # Each step changes the views, around the focal point only
    for before, after in [(step1, step2), (step2, step3), (step3, temperature_view)]:
        changed = np.abs(before.astype(int) - after.astype(int)).max(axis=-1) > 30
        logging.info(f"  {changed.mean():.2%} of pixels changed")
        assert 0.001 < changed.mean() < 0.5


def test_synthetic_head_two_trajectories(harness_factory, viewer_screenshots):
    """Two trajectories: each tab opens on its own focal point and gets its own results; Combine
    Trajectories adds the merged fields to both tabs."""
    h = harness_factory(['left', 'right'])
    h.run_step1_synthetic([LEFT, RIGHT])
    window = h.open_viewer()
    viewer_screenshots.add_window(window, "Step 1")
    logging.info(f"Checking the tabs open on their focal points, left {LEFT} and right {RIGHT}")
    assert vh.slider_values(window.viewer[0]) == [LEFT[1], LEFT[0], LEFT[2]]
    assert vh.slider_values(window.viewer[1]) == [RIGHT[1], RIGHT[0], RIGHT[2]]

    t1 = vh.synthetic_t1()
    for n, peak in enumerate((LEFT, RIGHT)):
        skull, water, _, _ = _steps_fields(t1, peak)
        h.UpdateNiftiAcResults(skull, water, n)
    viewer_screenshots.add_window(window, "Step 2 per trajectory")
    logging.info("Checking each tab's Skull field peaks at its own focal point")
    for viewer, peak in zip(window.viewer, (LEFT, RIGHT)):
        data = vh.layer_data(viewer, 'Skull')
        assert np.unravel_index(np.argmax(data), data.shape) == peak

    logging.info("Combining trajectories: merged field is the maximum of both fields")
    merged = nibabel.Nifti1Image(np.maximum(vh.gaussian_field(t1, LEFT).get_fdata(),
                                            vh.gaussian_field(t1, RIGHT).get_fdata()).astype(np.float32), t1.affine)
    h.UpdateNiftiMergedAcResults(merged)
    viewer_screenshots.add_window(window, "Step 2 Combine Trajectories (merged field, per trajectory fields hidden)")
    logging.info("Checking both tabs show MergedAc with its data and hide their own Skull field")
    for viewer in window.viewer:
        assert _layers(viewer)[-1] == 'MergedAc'
        assert _visible(viewer, 'MergedAc') and not _visible(viewer, 'Skull')
        np.testing.assert_allclose(vh.layer_data(viewer, 'MergedAc'), merged.get_fdata())
