"""
Helpers for the VTK visualization tests (Unit/BabelBrain/GUIComponents/test_nifti_viewer.py,
Integration/test_vtk_visualization.py and the VTK visualization test in E2E/test_e2e_release.py).

  * synthetic head volumes as BabelBrain hands them to the viewer: the Step 1 tissue mask (BabelBrain
    labels, focal point voxel = 5), T1W, CT (BabelBrain's HU look-up indices and their table) and an
    air region mask, all from the synthetic dataset (synthetic_dataset.py)
  * reading back what a viewer shows (layer ids, a layer's data)
  * screenshots of viewers for the html report (ViewerScreenshots, the viewer_screenshots fixture)
"""
from pathlib import Path

import nibabel
import numpy as np
from PIL import Image

from Tests.helpers.parameters import SYNTHETIC_DATASET
from Tests.helpers.plots import image_to_base64

# charm label (synthetic_dataset.LAYERS) -> BabelBrain Step 1 mask label (nifti_viewer._TISSUE_RGBA)
CHARM_TO_BABELBRAIN = {5: 1, 7: 2, 8: 3, 1: 4, 2: 4, 3: 4}
FOCAL_LABEL = 5
# Focal point voxel of the synthetic head (in the brain, off the volume's middle slices)
SYNTHETIC_TARGET = (20, 24, 26)

SCREENSHOT_WIDTH = 900  # px each viewer screenshot is shown at in the report


# ================================================================================================================================
# SYNTHETIC HEAD VOLUMES
# ================================================================================================================================
def _dataset_file(name):
    return Path(SYNTHETIC_DATASET['folder_path']) / name


def synthetic_t1():
    return nibabel.load(str(_dataset_file('T1W.nii.gz')))


def synthetic_mask(target=SYNTHETIC_TARGET):
    """Step 1 mask of the synthetic head: BabelBrain tissue labels with the focal point at `target`."""
    t1 = synthetic_t1()
    charm = np.asarray(nibabel.load(str(_dataset_file('m2m_synthetic/final_tissues.nii.gz'))).dataobj)[..., 0]
    mask = np.zeros(charm.shape, np.float32)
    for charm_label, label in CHARM_TO_BABELBRAIN.items():
        mask[charm == charm_label] = label
    mask[tuple(target)] = FOCAL_LABEL
    return nibabel.Nifti1Image(mask, t1.affine)


def synthetic_ct_indices():
    """Synthetic CT as Step 1 saves it with a CT (<prefix>CT.nii.gz): indices into the HU look-up
    table (<prefix>CT-cal.npz UniqueHU), bone only. Returns (index image, UniqueHU)."""
    ct = nibabel.load(str(_dataset_file('CT.nii.gz')))
    hu = np.asarray(ct.dataobj, dtype=np.float32)
    hu = np.where(hu > 300, hu, 0)
    unique_hu, indices = np.unique(hu, return_inverse=True)
    return nibabel.Nifti1Image(indices.reshape(hu.shape).astype(np.int16), ct.affine), unique_hu.astype(np.float32)


def synthetic_air_mask():
    """Air regions (<prefix>AirRegions.nii.gz): the synthetic CT's background, outside the head."""
    ct = nibabel.load(str(_dataset_file('CT.nii.gz')))
    air = (np.asarray(ct.dataobj) <= -900).astype(np.float32)
    return nibabel.Nifti1Image(air, ct.affine)


def gaussian_field(reference, peak, width=4.0, scale=1.0, offset=0.0):
    """Field on `reference`'s grid peaking at voxel `peak` (Step 2 pressure, Step 3 intensity or
    temperature = offset + scale * gaussian)."""
    grid = np.indices(reference.shape[:3])
    dist2 = sum((g - p) ** 2 for g, p in zip(grid, peak))
    data = offset + scale * np.exp(-dist2 / (2 * width ** 2))
    return nibabel.Nifti1Image(data.astype(np.float32), reference.affine)


# ================================================================================================================================
# WHAT A VIEWER SHOWS
# ================================================================================================================================
def layer_ids(viewer):
    """Result ids of a viewer's layers in order (the base volume has id '')."""
    return [row._id for row in viewer._layer_panel._rows]


def layer(viewer, layer_id):
    """VolumeRecord of the layer with `layer_id`."""
    return viewer._volumes[layer_ids(viewer).index(layer_id)]


def layer_data(viewer, layer_id):
    """Voxel data of a layer as the viewer renders it."""
    from vtk.util import numpy_support
    img = layer(viewer, layer_id).vtk_idx
    return numpy_support.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(img.GetDimensions(), order='F')


def slider_values(viewer):
    return [vp._slider.value() for vp in viewer._vps]


def crosshair_centre(vp):
    """Centre of a viewport's crosshair (where its two lines cross)."""
    pts = vp._cross_h.GetMapper().GetInput().GetPoints()
    return (np.array(pts.GetPoint(0)) + np.array(pts.GetPoint(1))) / 2


# ================================================================================================================================
# SCREENSHOTS
# ================================================================================================================================
def render_viewer(viewer, path):
    """Screenshot of a viewer's three viewports and colour bar (NiftiViewer.grab_screenshot, what
    the viewer's own Screenshot button saves). Returns the RGB pixels."""
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    viewer.grab_screenshot(str(path))
    return np.asarray(Image.open(path).convert('RGB'))


def screenshot_pixel(viewer, n, world):
    """(row, column) of RAS point `world` in viewport n of a render_viewer screenshot (the viewports
    are captured side by side at their render window size)."""
    import vtk
    vp = viewer._vps[n]
    coordinate = vtk.vtkCoordinate()
    coordinate.SetCoordinateSystemToWorld()
    coordinate.SetValue(*map(float, world))
    x, y = coordinate.GetComputedDoubleDisplayValue(vp.renderer)
    offset = sum(v.vtk_widget.GetRenderWindow().GetSize()[0] for v in viewer._vps[:n])
    height = vp.vtk_widget.GetRenderWindow().GetSize()[1]
    return int(round(height - 1 - y)), int(round(offset + x))


def viewport_pixels(pixels, viewer):
    """The viewport part of a render_viewer screenshot (without the colour bar)."""
    return pixels[:, :pixels.shape[1] - viewer._colorbar.width()]


def _bring_to_front(widget):
    """Select the tabs holding `widget` (a viewer in a background trajectory tab is never laid out,
    so its views would render at a tiny default size). Returns [(stack, previously current page)]
    to restore."""
    from PySide6.QtWidgets import QStackedWidget
    changed = []
    child, parent = widget, widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QStackedWidget) and parent.currentWidget() is not child:
            changed.append((parent, parent.currentWidget()))
            parent.setCurrentWidget(child)
        child, parent = parent, parent.parentWidget()
    return changed


class ViewerScreenshots:
    """Screenshots of viewers for this test's html report entry (request.node.screenshots). Each
    add() captures a viewer's viewports and its layer panel (layer names, opacity, colour map,
    cutoff, visibility) side by side under one caption."""

    def __init__(self, request, folder):
        if not hasattr(request.node, 'screenshots'):
            request.node.screenshots = []
        self.entries = request.node.screenshots
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def add(self, viewer, caption):
        """Screenshot `viewer` into the report. Returns the viewport pixels (RGB, without the colour bar)."""
        from PySide6.QtTest import QTest
        shown = _bring_to_front(viewer)
        # Let Qt lay out the viewer (a tab just brought to front) and layer rows BabelBrain just added
        # (otherwise they're captured at their unlaid-out size, squeezed)
        QTest.qWait(100)
        self.count += 1
        path = self.folder / f"viewer_{self.count:02d}.png"
        pixels = render_viewer(viewer, path)
        panel_path = self.folder / f"layers_{self.count:02d}.png"
        # Every layer row, also the ones scrolled out of the panel's view
        viewer._layer_panel._container.grab().save(str(panel_path))
        for stack, current in shown:
            stack.setCurrentWidget(current)
        self.entries.append({'kind': 'html', 'html': f"<div style='margin-top:8px'><strong>{caption}</strong></div>"})
        self.entries.append({'image': image_to_base64(path), 'caption': 'viewports', 'width': SCREENSHOT_WIDTH})
        self.entries.append({'image': image_to_base64(panel_path), 'caption': 'layers', 'width': 250})
        return viewport_pixels(pixels, viewer)

    def add_window(self, window, caption):
        """Screenshot every trajectory tab of a NiftiViewerWindow."""
        return [self.add(viewer, f"{caption} &mdash; {window.tabs.tabText(n)}")
                for n, viewer in enumerate(window.viewer)]
