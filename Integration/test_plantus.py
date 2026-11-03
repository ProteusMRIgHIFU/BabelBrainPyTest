"""
Integration tests for PlanTUS (BabelBrain_Pytest_Usage.md, Table 8): BabelBrain's "Run PlanTUS"
(RUN_PLAN_TUS in PlanTUSViewer/RunPlanTUS.py) run with the bundled PlanTUS and SimNIBS on the
ernie dataset, superficial left target, CTX_500 at 500 kHz.

The run follows BabelBrain's steps with PlanTUS' viewers replaced by their outcome:

  1  Execute writes the transducer config and target mask, RunPlanTUSBackground runs PlanTUS through
     the bundled run script (with --skip_viewer, see plantus_helpers.run_plantus_headless)
  2  Running it again with "reload" reuses the results
  3  The placement for the best scoring skin vertex (the one PlanTUS suggests) is generated, as when
     the user picks it in the viewer, and its trajectory handed to BabelBrain the way PlanTUS'
     "use this placement in BabelBrain" does
  4  check_queue_TUSPlan turns it into the _PlanTUS trajectory BabelBrain continues with
  5  BabelBrain's PlanTUS result viewers open on the real outputs

PlanTUS runs on the CPU, so no GPU is needed. Requires simnibs_root_path in Tests/config.ini.
"""
import logging
import shutil
import time
from pathlib import Path

import nibabel
import numpy as np
import pytest

from Tests.helpers import plantus_helpers as ph
from Tests.helpers.parameters import requires_feature
from Tests.helpers.pipeline_helpers import simnibs_root

pytestmark = [requires_feature('plantus'), pytest.mark.slow, pytest.mark.major_release]

TX_NAME, FREQ = 'CTX_500', 500e3
SKIN_DISTANCE = 0.0


@pytest.fixture(scope='module')
def module_monkeypatch():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


class _QtBot:
    """Keeps the widgets of the module scoped PlanTUS run alive (qtbot is function scoped)."""
    def __init__(self):
        self.widgets = []

    def addWidget(self, widget):
        self.widgets.append(widget)


@pytest.fixture(scope='module')
def plantus_run(qapp, tmp_path_factory, module_monkeypatch):
    """PlanTUS run on a copy of the ernie inputs, as BabelBrain starts it (step 1)."""
    simnibs = simnibs_root()
    python, missing = ph.missing_plantus_modules(simnibs)
    assert python is not None, f"No python found in {simnibs}/simnibs_env"
    assert not missing, (f"PlanTUS dependencies missing from simnibs_env: {missing}, install them with\n"
                         f"  {python} -m pip install {' '.join(missing)}")
    tmp = tmp_path_factory.mktemp('plantus_ernie')
    logging.info(f"Copying the ernie T1, mesh and trajectory into {tmp} (PlanTUS writes next to the mesh)")
    t1, m2m, traj = ph.copy_ernie_inputs(tmp)

    logging.info("Running Execute: writes the transducer config and target mask, starts PlanTUS")
    runner = ph.make_run_plan_tus(_QtBot(), module_monkeypatch, TX_NAME, FREQ, t1, m2m, traj,
                                  skin_distance=SKIN_DISTANCE, simnibs_root=simnibs)
    tx_config = ph.execute(runner)
    logging.info(f"PlanTUS config for {TX_NAME} at {FREQ / 1e3:g} kHz: {tx_config}")

    logging.info("Running PlanTUS planning with SimNIBS (a couple of minutes)")
    start = time.time()
    log = ph.run_plantus_headless(module_monkeypatch, runner.planTUSargs)
    logging.info(f"PlanTUS finished in {(time.time() - start) / 60:.1f} min")
    return {'runner': runner, 'log': log, 'tx_config': tx_config, 't1': t1, 'traj': traj,
            'out': Path(runner.PlanOutputPath), 'mask': Path(runner.planTUSargs[6])}


def _gifti(path):
    return nibabel.load(str(path))


def _metric(path):
    return np.asarray(_gifti(path).darrays[0].data, dtype=float)


def _composite_path(out):
    composite = sorted(out.glob('composite_*_skin.func.gii'))
    assert len(composite) == 1, f"Expected one composite metric in {out}, found {composite}"
    return composite[0]


@pytest.fixture(scope='module')
def placement(plantus_run, module_monkeypatch):
    """Placement for the best scoring vertex, accepted for BabelBrain (step 3)."""
    runner, out = plantus_run['runner'], plantus_run['out']
    # The best scoring vertex is the one PlanTUS' viewer suggests
    vertex = int(np.nanargmax(_metric(_composite_path(out))))
    logging.info(f"Generating the placement for the best scoring skin vertex {vertex}")
    runner.GenerateTrajectory(vertex)
    runner.CaltimerTUSPlan.stop()  # the started process is a fake one, it's run below instead
    process = ph.FakeProcess.created[-1]
    log = ph.run_plantus_headless(module_monkeypatch, process.args, **process.kwargs)

    traj_files = sorted((out / runner.ID).glob('*_Trajectory_BabelBrain.txt'))
    if traj_files:
        logging.info(f"Accepting the placement for BabelBrain ({traj_files[0].name})")
        # What PlanTUS does when the user accepts the placement for BabelBrain
        shutil.copyfile(traj_files[0], runner.PlanTUSBabelBrainResult)
    return {'vertex': vertex, 'log': log, 'traj_files': traj_files}


# ================================================================================================================================
# TESTS
# ================================================================================================================================
def test_plantus_planning(plantus_run):
    """PlanTUS runs from BabelBrain and writes its skin maps for the target."""
    log, out = plantus_run['log'], plantus_run['out']
    logging.info(f"Checking PlanTUS succeeded and wrote its skin maps to {out}")
    assert '--Babel-Brain-Success' in log, f"PlanTUS failed:\n{log[-5000:]}"
    missing = [f for f in ph.PLANTUS_PLANNING_OUTPUTS if not (out / f).is_file()]
    assert not missing, f"PlanTUS outputs missing in {out}: {missing}\n{log[-5000:]}"
    assert (out / plantus_run['mask'].name).is_file(), "PlanTUS keeps a copy of the target mask"

    logging.info("Checking every skin map has one value per skin surface vertex")
    skin = _gifti(out / 'skin.surf.gii')
    n_vertices = skin.darrays[0].data.shape[0]
    assert n_vertices > 1000 and skin.darrays[1].data.shape[1] == 3
    for f in ph.PLANTUS_PLANNING_OUTPUTS[2:] + [_composite_path(out).name]:
        assert _metric(out / f).shape == (n_vertices,), f"{f} doesn't have one value per skin vertex"

    logging.info("Checking the skin to target distances against the target mask centre")
    distances = _metric(out / 'distances_skin.func.gii')
    target = ph.mask_center_ras(plantus_run['mask'])
    coords = skin.darrays[0].data
    valid = np.isfinite(distances) & (distances > 0)
    np.testing.assert_allclose(distances[valid], np.linalg.norm(coords[valid] - target, axis=1), atol=0.5)
    assert 5 < distances[valid].min() < 40, "superficial target should be a few cm under the skin"

    logging.info("Checking the within-reach map: binary, only where the target is within the transducer's maximal distance")
    reachable = _metric(out / 'distances_skin_thresholded.func.gii')
    assert set(np.unique(reachable[np.isfinite(reachable)])) <= {0.0, 1.0}
    assert reachable[np.isfinite(reachable)].sum() > 0, "no skin vertex within reach of the target"
    max_distance = plantus_run['tx_config']['max_distance']
    assert np.all(distances[reachable == 1] <= max_distance + 1e-3)

    logging.info("Checking the skin to target angles are in 0-180 degrees")
    angles = _metric(out / 'angles_skin.func.gii')
    assert np.nanmin(angles) >= 0 and np.nanmax(angles) <= 180


def test_plantus_reuses_results(plantus_run, module_monkeypatch):
    """Running PlanTUS again and choosing reload reuses the results (step 2)."""
    import PlanTUSViewer.RunPlanTUS as RunPlanTUS

    runner, out = plantus_run['runner'], plantus_run['out']
    assert '--Babel-Brain-Success' in plantus_run['log']
    # Modification times, to check the reload doesn't rewrite the outputs
    before = {f: (out / f).stat().st_mtime_ns for f in ph.PLANTUS_PLANNING_OUTPUTS}

    logging.info("Running Execute again and answering 'No' to recalculating the existing results")
    asked = []
    with module_monkeypatch.context() as m:
        m.setattr(RunPlanTUS.QMessageBox, 'question', lambda *a, **k: asked.append(a) or RunPlanTUS.QMessageBox.No)
        ph.execute(runner)
    assert asked, "BabelBrain should ask whether to recalculate existing PlanTUS results"
    assert runner.planTUSargs[-1] is True

    logging.info("Running PlanTUS with --reuse_existing, its outputs must be left as they are")
    log = ph.run_plantus_headless(module_monkeypatch, runner.planTUSargs)
    assert '--Babel-Brain-Success' in log, log[-5000:]
    assert 'Reusing existing data' in log, log[-5000:]
    assert {f: (out / f).stat().st_mtime_ns for f in ph.PLANTUS_PLANNING_OUTPUTS} == before


def test_plantus_placement(plantus_run, placement):
    """The placement for a skin vertex aims the transducer from that vertex at the target (step 3)."""
    runner, out = plantus_run['runner'], plantus_run['out']
    logging.info(f"Checking the placement for vertex {placement['vertex']} wrote its trajectory and transducer files")
    assert '--Babel-Brain-Success' in placement['log'], placement['log'][-5000:]
    assert len(placement['traj_files']) == 1, f"Expected one BabelBrain trajectory in {out / runner.ID}"
    traj_file = placement['traj_files'][0]
    stem = traj_file.name.replace('_Trajectory_BabelBrain.txt', '')
    for suffix in ('_TransducerModel.surf.gii', '_Trajectory_Brainsight.txt', '_PositionMatrix_Localite.txt'):
        assert (traj_file.parent / (stem + suffix)).is_file(), suffix

    logging.info("Checking the trajectory is anchored at the target with a valid rotation")
    TT = ph.read_trajectory('brainsight', traj_file)
    target = ph.mask_center_ras(plantus_run['mask'])
    original_target = ph.read_trajectory('brainsight', plantus_run['traj'])[:3, 3]
    np.testing.assert_allclose(TT[:3, 3], target, atol=0.5)  # anchored at the target mask centre
    np.testing.assert_allclose(TT[:3, 3], original_target, atol=1.5)
    np.testing.assert_allclose(TT[:3, :3] @ TT[:3, :3].T, np.eye(3), atol=1e-4)

    logging.info("Checking the beam axis goes through the chosen skin vertex")
    vertex = _gifti(out / 'skin.surf.gii').darrays[0].data[placement['vertex']]
    assert ph.distance_point_to_line(vertex, TT[:3, 3], TT[:3, 2]) < 1.0
    distance = _metric(out / 'distances_skin.func.gii')[placement['vertex']]
    assert np.linalg.norm(vertex - TT[:3, 3]) == pytest.approx(distance, abs=1.0)


def test_plantus_trajectory_to_babelbrain(plantus_run, placement, module_monkeypatch):
    """The accepted placement becomes the _PlanTUS trajectory BabelBrain continues with (step 4)."""
    runner, traj = plantus_run['runner'], plantus_run['traj']
    assert placement['traj_files'], "no PlanTUS placement to hand to BabelBrain"
    logging.info("Reporting PlanTUS success to check_queue_TUSPlan and accepting the final results dialog")
    shown = ph.finish_plantus(runner, module_monkeypatch)

    assert not shown['critical'], shown['critical']
    assert len(shown['final_results']) == 1 and Path(shown['final_results'][0]).is_file()
    logging.info("Checking BabelBrain now uses a _PlanTUS trajectory file holding the placement")
    final = Path(runner.MainApp.Config['Mat4Trajectory'])
    assert final == traj.parent / f"{traj.stem}_PlanTUS.txt" and final.is_file()
    assert ph.trajectory_ids('brainsight', final) == [runner.ID]
    np.testing.assert_allclose(ph.read_trajectory('brainsight', final),
                               ph.read_trajectory('brainsight', placement['traj_files'][0]), atol=1e-6)
    assert runner.MainApp.Config['ID'] == [runner.ID]
    assert runner.MainApp._bTrajAlreadyModified is False


def test_plantus_viewers(qtbot, plantus_run, placement):
    """BabelBrain's PlanTUS viewers open on the real outputs and pass the selected vertex on (step 5)."""
    from PlanTUSViewer.PlanTUSViewer import FinalResultViewer, MultiGiftiViewerWidget

    out = plantus_run['out']
    skin, reach = str(out / 'skin.surf.gii'), str(out / 'distances_skin_thresholded.func.gii')
    # Same maps as RUN_PLAN_TUS.showTUSPlanViewer
    gifti_files = [(skin, str(out / 'distances_skin.func.gii'), reach, [0, 100], 'Distance'),
                   (skin, str(out / 'target_intersection_skin.func.gii'), reach, None, 'Target Intersection'),
                   (skin, str(out / 'angles_skin.func.gii'), reach, [0, 20], 'Angle'),
                   (skin, str(out / 'skin_skull_angles_skin.func.gii'), reach, [0, 20], 'Skin-Skull Angle')]
    logging.info("Opening the PlanTUS skin map viewer on the real outputs")
    selected = []
    viewer = MultiGiftiViewerWidget(gifti_files, MaxViews=4, callBackAfterGenTrajectory=selected.append)
    qtbot.addWidget(viewer)
    assert len(viewer.viewers) == 4
    assert not viewer.generateTrajectoryPushButton.isEnabled()

    logging.info(f"Selecting vertex {placement['vertex']} and clicking Generate Trajectory")
    surface = _gifti(skin)
    faces = surface.darrays[1].data
    cell = int(np.argmax(np.any(faces == placement['vertex'], axis=1)))
    viewer.broadcast_selection(placement['vertex'], cell, tuple(surface.darrays[0].data[placement['vertex']]))
    assert viewer.generateTrajectoryPushButton.isEnabled()
    viewer.generateTrajectoryPushButton.click()
    assert selected == [placement['vertex']]

    logging.info("Opening the final results viewer on the skin and the placed transducer model")
    assert placement['traj_files'], "no PlanTUS placement to show"
    model = str(placement['traj_files'][0]).replace('_Trajectory_BabelBrain.txt', '_TransducerModel.surf.gii')
    final = FinalResultViewer([skin, model])
    qtbot.addWidget(final)
