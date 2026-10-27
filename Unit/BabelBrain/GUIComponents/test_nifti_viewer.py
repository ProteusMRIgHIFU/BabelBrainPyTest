"""
Unit tests for the VTK NIfTI viewer (GUIComponents/nifti_viewer.py), opened from BabelBrain's
"VTK visualization" button (BabelBrain_Pytest_Usage.md, Tables 3 and 7).

The viewer only needs small synthetic volumes, so these tests use no GPU or test data:

  * geometry/colour helpers  (pure functions: affine decomposition, bounding box, VTK image and
                              transform conversion, plane geometries, lookup tables)
  * load_volume_record       (NIfTI image/file -> VolumeRecord)
  * NiftiViewer              (base/overlay loading, navigation, coordinate conversion, layer
                              controls, overlay removal, window/level reset, screenshots)
  * NiftiViewerWindow        (one tab and viewer per trajectory, closed signal)
"""
import nibabel
import numpy as np
import pytest

from Tests.helpers.parameters import requires_feature

pytestmark = requires_feature('vtk_viewer')

SHAPE = (20, 30, 40)
SPACING = (2.0, 1.0, 1.5)
ORIGIN = (-20.0, -15.0, -30.0)


# ================================================================================================================================
# FIXTURES
# ================================================================================================================================
@pytest.fixture(scope='module')
def nv():
    """The viewer module (imported here so collection works on versions without it)."""
    from GUIComponents import nifti_viewer
    return nifti_viewer


def _affine(spacing=SPACING, origin=ORIGIN, flip_x=False):
    """Axis aligned affine (RAS, or LAS when flip_x)."""
    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = origin
    if flip_x:
        affine[0, 0] *= -1
    return affine


def _volume(shape=SHAPE, affine=None, data=None):
    """Synthetic volume whose voxel value encodes its index (i + 100 j + 10000 k)."""
    if data is None:
        i, j, k = np.indices(shape)
        data = (i + 100 * j + 10000 * k).astype(np.float32)
    return nibabel.Nifti1Image(data, _affine() if affine is None else affine)


def _tissue_mask(shape=SHAPE, target=(10, 15, 20)):
    """Tissue label volume (labels 0-4) with the focal point voxel (label 5) at `target`."""
    data = np.zeros(shape, np.float32)
    data[2:-2, 2:-2, 2:-2] = 1
    data[4:-4, 4:-4, 4:-4] = 2
    data[6:-6, 6:-6, 6:-6] = 4
    data[target] = 5
    return nibabel.Nifti1Image(data, _affine())


@pytest.fixture
def viewer(qtbot, nv):
    """NiftiViewer showing the tissue mask as base volume, focused on its target voxel."""
    widget = nv.NiftiViewer()
    qtbot.addWidget(widget)
    widget.resize(900, 300)
    widget.show()
    widget.load_base(_tissue_mask(), np.array([10, 15, 20]), 'Tissue Type', tissue_label=True)
    return widget


# ================================================================================================================================
# GEOMETRY / COLOUR HELPERS
# ================================================================================================================================
def test_hex_rgb(nv):
    assert nv._hex_rgb('#ff8000') == pytest.approx((1.0, 128 / 255, 0.0))
    assert nv._hex_rgb('00ff00') == pytest.approx((0.0, 1.0, 0.0))


def test_decompose_returns_spacing_and_unit_directions(nv):
    affine = _affine(flip_x=True)
    spacing, directions = nv._decompose(affine)
    np.testing.assert_allclose(spacing, SPACING)
    np.testing.assert_allclose(directions, np.diag([-1.0, 1.0, 1.0]))


def test_ras_bbox_spans_first_to_last_voxel_centres(nv):
    ras_min, ras_max = nv._ras_bbox(_affine(), SHAPE)
    np.testing.assert_allclose(ras_min, ORIGIN)
    np.testing.assert_allclose(ras_max, np.array(ORIGIN) + (np.array(SHAPE) - 1) * SPACING)


def test_ras_bbox_with_flipped_axis(nv):
    ras_min, ras_max = nv._ras_bbox(_affine(flip_x=True), SHAPE)
    x_last = ORIGIN[0] - (SHAPE[0] - 1) * SPACING[0]
    assert ras_min[0] == pytest.approx(x_last)
    assert ras_max[0] == pytest.approx(ORIGIN[0])


def test_numpy_to_vtk_index_keeps_voxel_order(nv):
    data = _volume().get_fdata(dtype=np.float32)
    img = nv.numpy_to_vtk_index(data, np.array(SPACING))
    assert img.GetDimensions() == SHAPE
    assert img.GetSpacing() == pytest.approx(SPACING)
    for ijk in [(0, 0, 0), (3, 7, 11), (19, 29, 39)]:
        assert img.GetScalarComponentAsDouble(*ijk, 0) == pytest.approx(data[ijk])


def test_make_vtk_transform_maps_vtk_points_to_ras(nv):
    """VTK image points (index * spacing) must land on the affine's RAS position."""
    affine = _affine(flip_x=True)
    xform = nv._make_vtk_transform(affine, np.array(SPACING))
    for ijk in [(0, 0, 0), (5, 10, 20), (19, 29, 39)]:
        vtk_point = np.array(ijk) * SPACING
        expected = (affine @ np.array([*ijk, 1.0]))[:3]
        np.testing.assert_allclose(xform.TransformPoint(*vtk_point), expected, atol=1e-9)


def test_affine_plane_geoms(nv):
    geoms = nv.affine_plane_geoms(_affine(), SHAPE)
    # Viewports scroll along j, i and k
    assert [g.n for g in geoms] == [SHAPE[1], SHAPE[0], SHAPE[2]]
    assert [g.step for g in geoms] == [SPACING[1], SPACING[0], SPACING[2]]
    centre = np.array(ORIGIN) + (np.array(SHAPE) - 1) / 2 * SPACING
    for g in geoms:
        np.testing.assert_allclose(g.centre, centre)
        assert abs(np.dot(g.normal, g.right)) < 1e-12 and abs(np.dot(g.normal, g.up)) < 1e-12


@pytest.mark.parametrize('neurological', [False, True])
def test_medical_plane_geoms(nv, neurological):
    ras_min, ras_max = np.array([-10., -20., -30.]), np.array([10., 20., 30.])
    geoms = nv.medical_plane_geoms(ras_min, ras_max, 1.0, neurological)
    # Axial, coronal, sagittal scroll along RAS Z, Y, X with one slice per mm (plus the last edge)
    np.testing.assert_allclose([g.normal for g in geoms], [[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    assert [g.n for g in geoms] == [61, 41, 21]
    # The neurological flip mirrors axial and coronal only
    assert [g.flip_lr for g in geoms] == [not neurological, neurological, True]


def test_volume_record_default_window_level(nv):
    rec = nv.VolumeRecord(name='v', vtk_idx=None, vtk_xform=None, lo=-10., hi=30.)
    assert (rec.wl_window, rec.wl_level) == (40., 10.)
    flat = nv.VolumeRecord(name='v', vtk_idx=None, vtk_xform=None, lo=5., hi=5.)
    assert flat.wl_window == 1.  # constant volume still gets a usable window


@pytest.mark.parametrize('cmap', ['Grey', 'Hot', 'Cool', 'Green', 'Red', 'Jet'])
def test_make_lut_is_opaque_without_cutoff(nv, cmap):
    lut = nv._make_lut(nv.CMAPS[cmap], 0., 1.)
    assert lut.GetNumberOfTableValues() == 1024
    assert all(lut.GetTableValue(i)[3] == 1.0 for i in (0, 511, 1023))


def test_make_lut_grey_and_jet_end_colours(nv):
    colour = 1 / 255  # VTK stores the table as 8 bit colours
    grey = nv._make_lut(None, 0., 1.)
    assert grey.GetTableValue(0)[:3] == pytest.approx((0., 0., 0.), abs=colour)
    assert grey.GetTableValue(1023)[:3] == pytest.approx((1., 1., 1.), abs=colour)
    jet = nv._make_lut('jet', 0., 1.)
    assert jet.GetTableValue(0)[:3] == pytest.approx((0., 0., 0.5), abs=colour)
    assert jet.GetTableValue(1023)[:3] == pytest.approx((0.5, 0., 0.), abs=colour)


def test_make_lut_cutoff_hides_lower_values_and_rescales(nv):
    plain = nv._make_lut('jet', 0., 10.)
    lut = nv._make_lut('jet', 0., 10., cutoff=5.)
    below = [lut.GetTableValue(i) for i in range(0, 500)]
    assert all(entry[3] == 0.0 for entry in below)
    # The first visible entry starts the full gradient, the last one is unchanged
    first_visible = next(i for i in range(1024) if lut.GetTableValue(i)[3] == 1.0)
    assert lut.GetTableValue(first_visible)[:3] == pytest.approx(plain.GetTableValue(0)[:3])
    assert lut.GetTableValue(1023) == pytest.approx(plain.GetTableValue(1023))


def test_tissue_label_lut_matches_babelbrain_colours(nv):
    lut = nv._make_lut('tissue_label', 0., 8.)
    assert lut.GetNumberOfTableValues() == len(nv._TISSUE_RGBA)
    for label, rgba in nv._TISSUE_RGBA.items():
        assert lut.GetTableValue(label) == pytest.approx(tuple(c / 255 for c in rgba))
    assert lut.GetTableValue(0)[3] == 0.0  # background is transparent


# ================================================================================================================================
# load_volume_record
# ================================================================================================================================
def test_load_volume_record_from_image(nv):
    img = _volume(affine=_affine(flip_x=True))
    rec, shape, zooms, code, affine = nv.load_volume_record(img, inname='T1W', id='T1W')
    assert (rec.name, rec.id) == ('T1W', 'T1W')
    assert shape == SHAPE
    assert zooms == pytest.approx(SPACING)
    assert code == 'LAS'
    np.testing.assert_allclose(affine, img.affine)
    data = img.get_fdata()
    assert (rec.lo, rec.hi) == (data.min(), data.max())


def test_load_volume_record_from_file(nv, tmp_path):
    path = tmp_path / 'volume.nii.gz'
    nibabel.save(_volume(), path)
    rec, shape, *_ = nv.load_volume_record(str(path))
    assert rec.name == 'volume.nii.gz'
    assert shape == SHAPE


def test_load_volume_record_percentile_range(nv):
    data = np.zeros(SHAPE, np.float32)
    data[0, 0, 0] = 1e6     # single outlier, ignored by the 0.5-99.5 percentile range
    data[1:, :, :] = np.linspace(0., 100., data[1:].size).reshape(data[1:].shape)
    rec, *_ = nv.load_volume_record(_volume(data=data), use_percentile=True)
    full, *_ = nv.load_volume_record(_volume(data=data))
    assert full.hi == 1e6
    assert rec.hi < 100.5 and rec.lo >= 0.


def test_load_volume_record_uses_first_volume_of_4d(nv):
    data = np.stack([np.full(SHAPE, 1., np.float32), np.full(SHAPE, 2., np.float32)], axis=-1)
    rec, shape, *_ = nv.load_volume_record(nibabel.Nifti1Image(data, _affine()))
    assert shape == SHAPE
    assert (rec.lo, rec.hi) == (1., 1.)


def test_load_volume_record_rejects_2d(nv):
    with pytest.raises(ValueError):
        nv.load_volume_record(nibabel.Nifti1Image(np.zeros((10, 10), np.float32), np.eye(4)))


# ================================================================================================================================
# NiftiViewer
# ================================================================================================================================
def test_add_overlay_requires_base(qtbot, nv):
    widget = nv.NiftiViewer()
    qtbot.addWidget(widget)
    with pytest.raises(RuntimeError):
        widget.add_overlay(_volume())
    with pytest.raises(RuntimeError):
        widget.affine_to_medical_coords((0, 0, 0))
    assert widget.get_current_ras() == (0., 0., 0.)


def test_load_base_starts_at_target(viewer):
    """The viewports open on the focal point voxel passed by BabelBrain (affine mode), with the
    crosshair on that voxel's centre."""
    sliders = [vp._slider for vp in viewer._vps]
    assert [s.maximum() for s in sliders] == [SHAPE[1] - 1, SHAPE[0] - 1, SHAPE[2] - 1]
    assert [s.value() for s in sliders] == [15, 10, 20]
    target_ras = (_affine() @ np.array([10, 15, 20, 1.]))[:3]
    np.testing.assert_allclose(viewer.get_current_ras(), target_ras)
    assert len(viewer._layer_panel._rows) == 1


def test_load_base_replaces_previous_volumes(viewer):
    viewer.add_overlay(_volume(), 'T1W', id='T1W')
    viewer.load_base(_tissue_mask(), np.array([10, 15, 20]), 'Tissue Type', tissue_label=True)
    assert len(viewer._volumes) == 1
    assert len(viewer._layer_panel._rows) == 1
    assert all(len(vp._layers) == 1 for vp in viewer._vps)


def test_add_overlay_adds_layer_everywhere(viewer):
    viewer.add_overlay(_volume(), 'T1W', id='T1W')
    assert [rec.name for rec in viewer._volumes] == ['Tissue Type', 'T1W']
    assert [row._id for row in viewer._layer_panel._rows] == ['', 'T1W']
    assert all(len(vp._layers) == 2 for vp in viewer._vps)


@pytest.mark.parametrize('mode', ['affine', 'medical'])
def test_navigate_to_ras_round_trip(viewer, mode):
    """Navigating to a voxel centre puts the crosshair (and displayed slices) exactly on it. The
    voxel is also on the medical mode grid (ras_min + 1 mm steps, the smallest voxel size)."""
    viewer.set_mode(mode)
    point = (_affine() @ np.array([4, 22, 30, 1.]))[:3]
    viewer.navigate_to_ras(*point)
    np.testing.assert_allclose(viewer.get_current_ras(), point, atol=1e-6)


@pytest.mark.parametrize('mode', ['affine', 'medical'])
def test_navigate_to_ras_round_trip_flipped_odd_volume(qtbot, nv, mode):
    """Same on a flipped (LAS) volume with odd dimensions, where the grid middle is a voxel centre."""
    shape, affine = (21, 27, 33), _affine(spacing=(1.0, 1.0, 1.0), flip_x=True)
    widget = nv.NiftiViewer()
    qtbot.addWidget(widget)
    widget.load_base(nibabel.Nifti1Image(np.ones(shape, np.float32), affine), np.array([10, 13, 16]))
    widget.set_mode(mode)
    for ijk in [(0, 0, 0), (3, 20, 9), (20, 26, 32)]:
        point = (affine @ np.array([*ijk, 1.]))[:3]
        widget.navigate_to_ras(*point)
        np.testing.assert_allclose(widget.get_current_ras(), point, atol=1e-6)


def test_navigate_to_ras_clips_outside_points(viewer):
    viewer.navigate_to_ras(1e4, -1e4, 1e4)
    assert [vp.current_slice() for vp in viewer._vps] == [0, SHAPE[0] - 1, SHAPE[2] - 1]


def test_medical_mode_starts_at_target(viewer):
    viewer.set_mode('medical')
    target_ras = (_affine() @ np.array([10, 15, 20, 1.]))[:3]
    np.testing.assert_allclose(viewer.get_current_ras(), target_ras, atol=1.0)


def test_affine_to_medical_coords(viewer):
    # Isotropic spacing is the smallest voxel size (1 mm); the medical grid starts at ras_min
    i_ax, i_co, i_sa = viewer.affine_to_medical_coords((10, 15, 20))
    assert (i_ax, i_co, i_sa) == pytest.approx((20 * SPACING[2], 15 * SPACING[1], 10 * SPACING[0]))


def test_layer_controls_update_record_and_vtk_property(viewer):
    viewer.add_overlay(_volume(), 'T1W', id='T1W')
    row = viewer._layer_panel._rows[1]
    rec = viewer._volumes[1]

    row._opacity_slider.setValue(40)
    assert rec.opacity == pytest.approx(0.4)
    assert all(vp._layers[1][1].GetOpacity() == pytest.approx(0.4) for vp in viewer._vps)

    row._cmap_combo.setCurrentText('Jet')
    assert rec.cmap == 'Jet'

    row._cutoff_edit.setText('5000')
    row._on_cutoff_changed()
    assert rec.cutoff == 5000.
    row._cutoff_edit.setText('')
    row._on_cutoff_changed()
    assert rec.cutoff is None

    row._eye_btn.setChecked(False)
    assert rec.visible is False
    assert all(vp._layers[1][1].GetOpacity() == 0. and not vp._layers[1][0].GetVisibility() for vp in viewer._vps)


def test_window_level_edit_and_reset(viewer):
    viewer.add_overlay(_volume(), 'T1W', id='T1W')
    rec = viewer._volumes[1]
    default = (rec.wl_window, rec.wl_level)

    viewer._on_wl_edited(1, 100., 50.)
    assert (rec.wl_window, rec.wl_level) == (100., 50.)
    assert all(vp._layers[1][1].GetColorWindow() == 100. for vp in viewer._vps)

    viewer._on_wl_reset(1)
    assert (rec.wl_window, rec.wl_level) == default


def test_reset_view_restores_window_level_and_target(viewer):
    viewer.add_overlay(_volume(), 'T1W', id='T1W')
    viewer._on_wl_edited(1, 100., 50.)
    viewer.navigate_to_ras(0., 0., 0.)
    viewer.reset_view()
    rec = viewer._volumes[1]
    assert (rec.wl_window, rec.wl_level) == (rec.hi - rec.lo, (rec.hi + rec.lo) / 2)
    assert [vp._slider.value() for vp in viewer._vps] == [15, 10, 20]


def test_remove_overlay_reindexes_remaining_layers(viewer):
    """BabelBrain removes and re-adds results by id, so later rows must follow their volume."""
    for name in ['T1W', 'Skull', 'Water']:
        viewer.add_overlay(_volume(), name, id=name)
    viewer._on_remove_requested(2)   # Skull
    assert [rec.name for rec in viewer._volumes] == ['Tissue Type', 'T1W', 'Water']
    assert [row._id for row in viewer._layer_panel._rows] == ['', 'T1W', 'Water']
    assert [row._vol_idx for row in viewer._layer_panel._rows] == [0, 1, 2]
    assert all(len(vp._layers) == 3 for vp in viewer._vps)

    # Controls of the row that moved up now act on the right volume
    viewer._layer_panel._rows[2]._opacity_slider.setValue(30)
    assert viewer._volumes[2].opacity == pytest.approx(0.3)
    assert viewer._volumes[1].opacity == 1.0


def test_base_volume_cannot_be_removed(viewer):
    viewer._on_remove_requested(0)
    viewer._on_remove_requested(5)
    assert len(viewer._volumes) == 1


def test_crosshair_visibility(viewer):
    viewer.set_crosshair_visible(False)
    assert not any(vp._cross_h.GetVisibility() or vp._cross_v.GetVisibility() for vp in viewer._vps)
    viewer.set_crosshair_visible(True)
    assert all(vp._cross_h.GetVisibility() and vp._cross_v.GetVisibility() for vp in viewer._vps)


def test_grab_screenshot_writes_png(viewer, tmp_path):
    from PySide6.QtGui import QImage

    viewer.add_overlay(_volume(), 'T1W', id='T1W')
    path = tmp_path / 'viewer.png'
    viewer.grab_screenshot(str(path))
    image = QImage(str(path))
    assert not image.isNull()
    # Three viewports plus the colour bar, side by side
    assert image.width() > sum(vp.vtk_widget.width() for vp in viewer._vps) * 0.9


# ================================================================================================================================
# NiftiViewerWindow
# ================================================================================================================================
@pytest.mark.parametrize('trajectories,titles', [
    (None, ['Trajectory 1']),
    (3, ['Trajectory 1', 'Trajectory 2', 'Trajectory 3']),
    (['traj_left', 'traj_right'], ['traj_left', 'traj_right']),
    ([], ['Trajectory 1']),
])
def test_window_has_one_viewer_per_trajectory(qtbot, nv, trajectories, titles):
    window = nv.NiftiViewerWindow(trajectories=trajectories)
    qtbot.addWidget(window)
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == titles
    assert len(window.viewer) == len(titles)
    assert len({id(v) for v in window.viewer}) == len(titles)  # independent viewers


def test_window_emits_closed(qtbot, nv):
    window = nv.NiftiViewerWindow(trajectories=['traj'])
    qtbot.addWidget(window)
    window.show()
    with qtbot.waitSignal(window.closed, timeout=2000):
        window.close()
