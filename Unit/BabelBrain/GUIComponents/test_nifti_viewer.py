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
  * synthetic head           (the synthetic dataset's tissue mask, T1W and CT rendered in every view
                              mode through the viewer tab's toolbar; screenshots go to the report)
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


# ================================================================================================================================
# SYNTHETIC HEAD (rendered views, screenshots in the report)
# ================================================================================================================================
@pytest.fixture
def head_tab(qtbot):
    """Viewer tab (toolbar + viewer) showing the synthetic head's Step 1 mask with its T1W overlay,
    set up the way BabelBrain's _UpdateVTKDomain does (tissue colours, T1W at 50 % opacity)."""
    from GUIComponents.nifti_viewer import NiftiViewerTab
    from Tests.helpers import vtk_viewer_helpers as vh

    tab = NiftiViewerTab()
    qtbot.addWidget(tab)
    tab.resize(1400, 520)
    tab.show()
    viewer = tab.viewer
    viewer.load_base(vh.synthetic_mask(), np.array(vh.SYNTHETIC_TARGET), 'Tissue Type', tissue_label=True)
    viewer.add_overlay(vh.synthetic_t1(), 'T1W', use_percentile=True, id='T1W')
    viewer._on_cmap_changed(0, 'TissueLabel')
    viewer._layer_panel._rows[1]._opacity_slider.setValue(50)
    qtbot.wait(50)
    return tab


def _target_ras():
    from Tests.helpers import vtk_viewer_helpers as vh
    return nibabel.affines.apply_affine(vh.synthetic_t1().affine, vh.SYNTHETIC_TARGET)


def _thirds(pixels):
    """The three viewports of a viewport screenshot."""
    w = pixels.shape[1] // 3
    return [pixels[:, i * w:(i + 1) * w] for i in range(3)]


def _count(pixels, rgb, tol=40):
    return int((np.abs(pixels.astype(int) - np.array(rgb)).max(axis=-1) <= tol).sum())


@pytest.mark.parametrize('mode,neurological', [('affine', False), ('medical', False), ('medical', True)],
                         ids=['affine', 'medical-radiological', 'medical-neurological'])
def test_head_views(qtbot, head_tab, viewer_screenshots, mode, neurological):
    """Every view mode of the toolbar opens on the focal point with the crosshair on it, and draws
    the head in all three viewports."""
    viewer = head_tab.viewer
    if mode == 'medical':
        head_tab._rb_medical.click()
        head_tab._cb_radio.setChecked(neurological)
    qtbot.wait(50)
    assert viewer._mode == mode and viewer._neurogical == neurological

    np.testing.assert_allclose(viewer.get_current_ras(), _target_ras(), atol=1e-6)
    for vp in viewer._vps:
        assert vp._cross_h.GetVisibility() and vp._cross_v.GetVisibility()
        np.testing.assert_allclose(np.dot(_crosshair(vp) - _target_ras(), vp._pg.right), 0, atol=1e-6)
    # The RAS boxes follow the crosshair
    np.testing.assert_allclose([float(e.text()) for e in head_tab._ras_edits], _target_ras(), atol=0.01)

    pixels = viewer_screenshots.add(viewer, f"Synthetic head, {mode} mode"
                                    f"{' (neurological)' if neurological else ''}, focal point at the crosshair")
    for view in _thirds(pixels):
        assert view.std() > 20, "viewport is blank"


def _crosshair(vp):
    from Tests.helpers import vtk_viewer_helpers as vh
    return vh.crosshair_centre(vp)


def _crosshair_column(view, axis=0):
    """Pixel column (axis=0) or row (axis=1) of a viewport's crosshair line (nifti_viewer.ACCENT)."""
    hits = (np.abs(view.astype(int) - np.array((0, 200, 255))).max(axis=-1) <= 60).mean(axis=axis)
    return int(np.argmax(hits))


def test_head_neurological_mirrors_axial_and_coronal(qtbot, head_tab, viewer_screenshots):
    """The neurological convention mirrors the axial and coronal views left-right (crosshair and
    R/L labels swap sides), and leaves the sagittal view as it is."""
    viewer = head_tab.viewer
    head_tab._rb_medical.click()
    radiological = _thirds(viewer_screenshots.add(viewer, "Medical mode, radiological"))
    labels_rad = [[a.GetInput() for a in vp._orient_actors] for vp in viewer._vps]
    head_tab._cb_radio.setChecked(True)
    neurological = _thirds(viewer_screenshots.add(viewer, "Medical mode, neurological (axial and coronal mirrored)"))
    labels_neu = [[a.GetInput() for a in vp._orient_actors] for vp in viewer._vps]

    assert labels_rad[:2] == [['R', 'L', 'A', 'P'], ['R', 'L', 'S', 'I']]
    assert labels_neu[:2] == [['L', 'R', 'A', 'P'], ['L', 'R', 'S', 'I']]
    assert labels_rad[2] == labels_neu[2] == ['A', 'P', 'S', 'I']
    # The target is 0.5 mm right of the head centre, so its crosshair isn't on the middle column
    for n, (rad, neu) in enumerate(zip(radiological, neurological)):
        width = rad.shape[1]
        col_rad, col_neu = _crosshair_column(rad), _crosshair_column(neu)
        if n < 2:
            assert abs(col_rad - (width / 2)) > 3
            assert abs((width - 1 - col_rad) - col_neu) <= 3, f"view {n} not mirrored"
        else:
            assert col_rad == col_neu


@pytest.mark.xfail(strict=True, reason="the tissue label look-up table (labels 0-8) is applied over the mask's "
                                       "data range (0-5 without brain segmentation), so label n shows the colour "
                                       "of label 8n/5, and linear interpolation blends labels at their borders")
def test_head_tissue_colours(qtbot, head_tab, viewer_screenshots):
    """Each tissue label of the Step 1 mask shows its colour (nifti_viewer._TISSUE_RGBA): scalp,
    cortical and trabecular bone, brain, and the focal point voxel at the crosshair."""
    from GUIComponents.nifti_viewer import _TISSUE_RGBA
    from Tests.helpers import vtk_viewer_helpers as vh

    viewer = head_tab.viewer
    viewer._layer_panel._rows[1]._eye_btn.setChecked(False)
    head_tab._cb_crosshair.setChecked(False)
    pixels = viewer_screenshots.add(viewer, "Tissue mask only: expected scalp blue, cortical bone light blue, "
                                    "trabecular bone cyan, brain green and the focal point yellow")
    target = _target_ras()
    problems = []
    for n, vp in enumerate(viewer._vps):
        # The head centre (the origin, in the brain) moved onto this viewport's slice through the target
        centre = np.dot(target, vp._pg.normal) * vp._pg.normal
        for name, world, label, tol in [('brain', centre, 4, 20), ('focal point', target, 5, 40)]:
            row, col = vh.screenshot_pixel(viewer, n, world)
            colour = pixels[row - 2:row + 3, col - 2:col + 3].reshape(-1, 3).mean(axis=0)
            if np.abs(colour - _TISSUE_RGBA[label][:3]).max() > tol:
                problems.append(f"view {n}: {name} shown as {np.round(colour).astype(int).tolist()}")
    # Labels the mask doesn't have (white and grey matter, CSF) must not show up
    for label in (6, 7, 8):
        share = _count(pixels, _TISSUE_RGBA[label][:3], tol=12) / pixels[..., 0].size
        if share > 1e-3:
            problems.append(f"colour of label {label}, not in the mask, covers {share:.1%} of the views")
    assert not problems, '; '.join(problems)


def test_head_layers_change_rendering(qtbot, head_tab, viewer_screenshots):
    """Turning the T1W layer off changes the views (the mask is then drawn alone)."""
    viewer = head_tab.viewer
    with_t1 = viewer_screenshots.add(viewer, "Tissue mask with T1W at 50 %")
    viewer._layer_panel._rows[1]._eye_btn.setChecked(False)
    mask_only = viewer_screenshots.add(viewer, "Tissue mask only (T1W hidden)")
    assert np.abs(with_t1.astype(int) - mask_only.astype(int)).mean() > 1


def test_head_overlays_ct_and_field(qtbot, head_tab, viewer_screenshots):
    """CT (bone only through its cutoff) and a Step 2 like field (Jet, cutoff 0.25) drawn over the head."""
    from Tests.helpers import vtk_viewer_helpers as vh

    viewer = head_tab.viewer
    before = viewer_screenshots.add(viewer, "Before overlays")
    indices, unique_hu = vh.synthetic_ct_indices()
    ct = nibabel.Nifti1Image(unique_hu[np.asarray(indices.dataobj)], indices.affine)
    viewer.add_overlay(ct, 'CT', id='CT')
    row = viewer._layer_panel._rows[-1]
    row._cutoff_edit.setText('1')
    row._on_cutoff_changed()
    field = vh.gaussian_field(vh.synthetic_t1(), vh.SYNTHETIC_TARGET)
    viewer.add_overlay(field, 'Skull', id='Skull')
    row = viewer._layer_panel._rows[-1]
    row._cmap_combo.setCurrentText('Jet')
    row._cutoff_edit.setText('0.25')
    row._on_cutoff_changed()
    after = viewer_screenshots.add(viewer, "CT (cutoff 1 HU) and a field at the focal point (Jet, cutoff 0.25)")

    assert vh.layer_ids(viewer) == ['', 'T1W', 'CT', 'Skull']
    assert vh.layer(viewer, 'CT').cutoff == 1.0 and vh.layer(viewer, 'Skull').cutoff == 0.25
    # The field only covers the focal region: views change, but most pixels don't
    changed = np.abs(before.astype(int) - after.astype(int)).max(axis=-1) > 30
    assert 0.001 < changed.mean() < 0.5


def test_head_crosshair_checkbox_and_reset(qtbot, head_tab, viewer_screenshots):
    viewer = head_tab.viewer
    head_tab._cb_crosshair.setChecked(False)
    hidden = viewer_screenshots.add(viewer, "Crosshairs off")
    assert not any(vp._cross_h.GetVisibility() for vp in viewer._vps)
    head_tab._cb_crosshair.setChecked(True)

    # A voxel centre of the synthetic head (its grid is at -19.5 + i mm)
    for edit, value in zip(head_tab._ras_edits, (-5.5, 3.5, 8.5)):
        edit.setText(str(value))
    head_tab._on_ras_entered()
    np.testing.assert_allclose(viewer.get_current_ras(), (-5.5, 3.5, 8.5), atol=1e-6)
    moved = viewer_screenshots.add(viewer, "Navigated to RAS (-5.5, 3.5, 8.5) from the toolbar")
    assert np.abs(hidden.astype(int) - moved.astype(int)).mean() > 0

    head_tab._btn_reset.click()
    np.testing.assert_allclose(viewer.get_current_ras(), _target_ras(), atol=1e-6)


def test_head_screenshot_button(qtbot, head_tab, monkeypatch, tmp_path, viewer_screenshots):
    """The toolbar's Screenshot button saves the three views and colour bar as a PNG."""
    import GUIComponents.nifti_viewer as nv_module
    from PIL import Image

    assert head_tab._btn_screenshot.isEnabled()
    target = tmp_path / 'saved'
    monkeypatch.setattr(nv_module.QFileDialog, 'getSaveFileName', lambda *a, **k: (str(target), 'PNG'))
    head_tab._btn_screenshot.click()
    saved = target.with_suffix('.png')
    assert saved.is_file()
    reference = viewer_screenshots.add(head_tab.viewer, "Same views as the saved screenshot")
    image = np.asarray(Image.open(saved).convert('RGB'))
    np.testing.assert_array_equal(image[:, :reference.shape[1]], reference)


def test_crosshair_drawn_when_target_on_middle_slices(qtbot, nv):
    """A target on the slices the viewports open on (the middle ones) still gets its crosshair: no
    slider changes, so load_base draws it itself."""
    from Tests.helpers import vtk_viewer_helpers as vh

    widget = nv.NiftiViewer()
    qtbot.addWidget(widget)
    widget.resize(900, 300)
    widget.show()
    middle = np.array(vh.synthetic_t1().shape) // 2
    widget.load_base(vh.synthetic_mask(middle), middle, 'Tissue Type', tissue_label=True)
    target = nibabel.affines.apply_affine(vh.synthetic_t1().affine, middle)
    for vp in widget._vps:
        assert vp._cross_h.GetVisibility()
        np.testing.assert_allclose(np.dot(vh.crosshair_centre(vp) - target, vp._pg.normal), 0, atol=1e-6)
    np.testing.assert_allclose(widget.get_current_ras(), target, atol=1e-6)


@pytest.mark.parametrize('extent', [(10.0, 20.0, 30.0), (10.5, 20.25, 29.7)], ids=['whole-mm', 'partial-mm'])
def test_medical_grid_centre_on_slice_grid(nv, extent):
    """The medical planes' centre is the middle of their slice grid (ras_min + i * iso), also when
    an extent isn't a whole number of iso steps, so slice i of a viewport is at ras_min + i * iso."""
    ras_min = np.array([-5.0, -10.0, -15.0])
    ras_max = ras_min + np.array(extent)
    for g in nv.medical_plane_geoms(ras_min, ras_max, 1.0, False):
        axis = int(np.argmax(np.abs(g.normal)))
        assert np.dot(g.centre, g.normal) == pytest.approx(ras_min[axis] + (g.n - 1) / 2)
