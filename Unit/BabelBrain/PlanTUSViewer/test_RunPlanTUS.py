"""
Unit tests for PlanTUSViewer/RunPlanTUS.py (BabelBrain_Pytest_Usage.md, Table 7), BabelBrain's side
of the PlanTUS integration (Options dialog -> "Run PlanTUS"):

  * PlanTUSTxConfig                        (transducer config yaml read by PlanTUS_wrapper.py)
  * RUN_PLAN_TUS.Execute                   (CTX_500 transducer config, target mask, PlanTUS
                                            arguments, recalculate/reload prompt)
  * RunPlanTUSBackground                   (PlanTUS command line, success/error reporting)
  * RUN_PLAN_TUS.check_queue_TUSPlan       (PlanTUS trajectory -> _PlanTUS trajectory file and
                                            BabelBrain config, for Brainsight and Slicer)

PlanTUS itself isn't run (Integration/test_plantus.py does), so no SimNIBS, GPU or test data is
needed. Execute and check_queue_TUSPlan run on the synthetic head's T1 (synthetic_dataset.py) with
Brainsight and Slicer trajectories made for it (plantus_helpers.synthetic_plantus_inputs).
"""
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel
import numpy as np
import pytest
import yaml

from Tests.helpers.parameters import BABELBRAIN_SRC, requires_feature
from Tests.helpers import plantus_helpers as ph

pytestmark = requires_feature('plantus')

WRAPPER = ph.PLANTUS_BUNDLED_ROOT / 'PlanTUS_wrapper.py'
TX_NAME, FREQ = 'CTX_500', 500e3  # transducer RUN_PLAN_TUS runs with
SKIN_DISTANCE = 5.0  # mm, Options dialog skin distance used in the Execute tests


@pytest.fixture(scope='module')
def rp():
    """The module under test (imported here so collection works on versions without PlanTUS)."""
    import PlanTUSViewer.RunPlanTUS as RunPlanTUS
    return RunPlanTUS


@pytest.fixture(autouse=True)
def _reset_fake_processes():
    ph.FakeProcess.created.clear()


# ================================================================================================================================
# PLANTUS TRANSDUCER CONFIG
# ================================================================================================================================
def _wrapper_required_keys():
    """Transducer config keys PlanTUS_wrapper.py reads with config["key"] (required, no default)."""
    return set(re.findall(r'config\["(\w+)"\]', WRAPPER.read_text()))


def test_PlanTUSTxConfig_ExportYAML(rp, tmp_path):
    logging.info("Writing a PlanTUSTxConfig with known values to yaml")
    cfg = rp.PlanTUSTxConfig(max_distance=80.0, min_distance=30.0, optimal_distance=50.0,
                             transducer_diameter=64.0, max_angle=10.0, plane_offset=10.5,
                             additional_offset=2.0, focal_distance_list=[30.0, 50.0, 80.0],
                             flhm_list=[8.0, 15.0, 30.0], IDTarget='superficial_target_left',
                             bUseGenericTransducerModel=True)
    fname = tmp_path / 'PlanTUSTxConfig.yaml'
    cfg.ExportYAML(str(fname))

    logging.info("Reading the yaml back: values as given, plus the default (equal) score weights")
    with open(fname) as f:
        loaded = yaml.safe_load(f)
    assert loaded == {
        'max_distance': 80.0, 'min_distance': 30.0, 'optimal_distance': 50.0,
        'transducer_diameter': 64.0, 'max_angle': 10.0, 'plane_offset': 10.5,
        'additional_offset': 2.0, 'focal_distance_list': [30.0, 50.0, 80.0],
        'flhm_list': [8.0, 15.0, 30.0], 'IDTarget': 'superficial_target_left',
        'weight_skin_target_distances': 0.2, 'weight_skin_target_angles': 0.2,
        'weight_skin_target_intersections': 0.2, 'weight_skin_skull_angles': 0.2,
        'weight_skull_thickness': 0.2, 'bUseGenericTransducerModel': True,
    }


def test_PlanTUSTxConfig_has_wrapper_keys(rp, tmp_path):
    """Every key the bundled PlanTUS requires is in the yaml BabelBrain writes."""
    required = _wrapper_required_keys()
    logging.info(f"PlanTUS_wrapper.py requires config keys {sorted(required)}")
    assert required, f"No config[...] keys found in {WRAPPER}"

    logging.info("Checking the yaml BabelBrain writes has all of them")
    fname = tmp_path / 'PlanTUSTxConfig.yaml'
    rp.PlanTUSTxConfig(1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, [1.0], [1.0]).ExportYAML(str(fname))
    with open(fname) as f:
        missing = required - set(yaml.safe_load(f))
    assert not missing, f"PlanTUS_wrapper.py requires keys PlanTUSTxConfig doesn't write: {sorted(missing)}"


def test_PlanTUSTxConfig_weights_sum_to_one(rp):
    logging.info("Checking PlanTUS' five default score weights add up to 1")
    cfg = rp.PlanTUSTxConfig(1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, [1.0], [1.0])
    weights = [cfg.weight_skin_target_distances, cfg.weight_skin_target_angles,
               cfg.weight_skin_target_intersections, cfg.weight_skin_skull_angles, cfg.weight_skull_thickness]
    assert sum(weights) == pytest.approx(1.0)


# ================================================================================================================================
# EXECUTE: TRANSDUCER CONFIG
# ================================================================================================================================
@pytest.fixture
def synthetic_inputs(tmp_path):
    return ph.synthetic_plantus_inputs(tmp_path)


def test_Execute_transducer_config(qtbot, monkeypatch, synthetic_inputs):
    """PlanTUSTxConfig.yaml for CTX_500: distances from its TPO range, offset between focus and exit plane."""
    tx = ph.transducer_config(TX_NAME)
    entry = tx['PlanTUS'][FREQ]
    focal, flhm = entry['FocalDistanceList'], entry['FHMLList']

    logging.info(f"Running Execute for {TX_NAME} at {FREQ / 1e3:g} kHz with a {SKIN_DISTANCE} mm skin distance")
    runner = ph.make_run_plan_tus(qtbot, monkeypatch, TX_NAME, FREQ, *synthetic_inputs, skin_distance=SKIN_DISTANCE)
    cfg = ph.execute(runner)

    logging.info("Comparing the PlanTUSTxConfig.yaml it wrote with the transducer's BabelBrain config")
    assert _wrapper_required_keys() <= set(cfg)
    assert cfg['IDTarget'] == runner.ID == 'target_left'
    # BabelBrain config is in m, PlanTUS' in mm
    assert cfg['min_distance'] == pytest.approx(tx['MinimalTPODistance'] * 1e3)
    assert cfg['max_distance'] == pytest.approx(tx['MaximalTPODistance'] * 1e3)
    assert cfg['optimal_distance'] == pytest.approx(np.mean(focal))
    assert cfg['plane_offset'] == pytest.approx((tx['FocalLength'] - tx['NaturalOutPlaneDistance']) * 1e3)
    assert cfg['transducer_diameter'] == pytest.approx(tx['TxDiam'] * 1e3)
    assert cfg['additional_offset'] == pytest.approx(SKIN_DISTANCE)
    assert cfg['focal_distance_list'] == focal and cfg['flhm_list'] == flhm
    assert cfg['max_angle'] == 10.0
    assert cfg['bUseGenericTransducerModel'] is False


def test_Execute_negative_skin_distance_clipped(qtbot, monkeypatch, synthetic_inputs):
    """A negative skin distance (transducer pushed into the scalp) is passed to PlanTUS as 0."""
    logging.info("Running Execute with a -3 mm skin distance, PlanTUS should get 0")
    runner = ph.make_run_plan_tus(qtbot, monkeypatch, TX_NAME, FREQ, *synthetic_inputs, skin_distance=-3.0)
    assert ph.execute(runner)['additional_offset'] == 0.0


# ================================================================================================================================
# EXECUTE: TARGET MASK AND PLANTUS ARGUMENTS
# ================================================================================================================================
@pytest.mark.parametrize('trajectory_type,combined,id_index', [
    ('brainsight', False, 0),
    ('brainsight', True, 1),
    ('slicer', False, 0),
    ('slicer', True, 1),
])
def test_Execute_target_mask(qtbot, monkeypatch, tmp_path, trajectory_type, combined, id_index):
    """2 mm radius mask around the selected trajectory's target, named after the trajectory index."""
    logging.info(f"Running Execute on a {'combined' if combined else 'single'} {trajectory_type} "
                 f"trajectory file, trajectory {id_index} selected")
    t1, m2m, traj = ph.synthetic_plantus_inputs(tmp_path, trajectory_type, combined)
    runner = ph.make_run_plan_tus(qtbot, monkeypatch, TX_NAME, FREQ, t1, m2m, traj,
                                  trajectory_type=trajectory_type, id_index=id_index)
    ph.execute(runner)

    logging.info("Checking the target mask is written on the T1 grid")
    mask_path = traj.parent / f"{traj.stem}_PlanTUSMask_T{id_index}.nii.gz"
    assert mask_path.is_file()
    mask = nibabel.load(str(mask_path))
    t1_img = nibabel.load(str(t1))
    assert mask.shape == t1_img.shape[:3]
    np.testing.assert_allclose(mask.affine, t1_img.affine)

    logging.info("Checking the mask is centred on the selected trajectory's target")
    target_ras = ph.read_trajectory(trajectory_type, traj, id_index)[:3, 3]
    voxel = np.array(t1_img.header.get_zooms()[:3])
    np.testing.assert_allclose(ph.mask_center_ras(mask_path), target_ras, atol=voxel.max())
    # every mask voxel within the 2 mm radius (rounded up to whole voxels)
    ijk = np.argwhere(np.asarray(mask.dataobj) > 0)
    ras = nibabel.affines.apply_affine(mask.affine, ijk)
    assert np.linalg.norm(ras - target_ras, axis=1).max() <= 2 + 2 * voxel.max()
    assert runner._RMat[:3, 3] == pytest.approx(target_ras)


def test_Execute_user_mask(qtbot, monkeypatch, tmp_path):
    """A volume ROI selected in the Options dialog is used as the PlanTUS mask."""
    logging.info("Writing a box shaped ROI on the T1 grid")
    t1, m2m, traj = ph.synthetic_plantus_inputs(tmp_path)
    t1_img = nibabel.load(str(t1))
    roi = np.zeros(t1_img.shape[:3], np.float32)
    roi[18:22, 20:24, 24:28] = 1
    roi_path = tmp_path / 'user_roi.nii.gz'
    nibabel.save(nibabel.Nifti1Image(roi, t1_img.affine), str(roi_path))

    logging.info("Running Execute with the ROI selected, the target mask should be the ROI")
    runner = ph.make_run_plan_tus(qtbot, monkeypatch, TX_NAME, FREQ, t1, m2m, traj, volume_roi=str(roi_path))
    ph.execute(runner)
    mask = nibabel.load(str(traj.parent / f"{traj.stem}_PlanTUSMask_T0.nii.gz"))
    np.testing.assert_array_equal(np.asarray(mask.dataobj), roi)


def test_Execute_plantus_arguments(qtbot, monkeypatch, tmp_path):
    """RunPlanTUSBackground is started in its own process with the files Execute prepared."""
    logging.info("Leaving a BabelBrain_trajectory.txt from a previous PlanTUS selection")
    t1, m2m, traj = ph.synthetic_plantus_inputs(tmp_path)
    (m2m / 'PlanTUS').mkdir()
    stale = m2m / 'PlanTUS' / 'BabelBrain_trajectory.txt'
    stale.write_text('previous selection')

    logging.info("Running Execute (the PlanTUS process is replaced by a fake one)")
    runner = ph.make_run_plan_tus(qtbot, monkeypatch, TX_NAME, FREQ, t1, m2m, traj, simnibs_root='/simnibs')
    ph.execute(runner)

    logging.info("Checking one process was started with RunPlanTUSBackground and the prepared files")
    assert len(ph.FakeProcess.created) == 1
    process = ph.FakeProcess.created[0]
    assert process.started and runner.CalProcess is process
    assert process.target.__name__ == 'RunPlanTUSBackground'
    _, scriptbase, simnibs, plantus_root, t1_arg, msh, mask, tx_config, only_reload = process.args
    assert process.args == runner.planTUSargs
    assert scriptbase.rstrip('/\\').endswith(str(ph.PLANTUS_SCRIPTS_DIR.relative_to(BABELBRAIN_SRC)))
    assert simnibs == '/simnibs'
    assert plantus_root == str(ph.PLANTUS_BUNDLED_ROOT)  # empty PlanTUS root: built-in PlanTUS
    assert t1_arg == str(t1)
    assert msh == str(next(m2m.glob('*.msh')))
    assert mask == str(traj.parent / f"{traj.stem}_PlanTUSMask_T0.nii.gz")
    assert tx_config == str(t1.parent / 'PlanTUSTxConfig.yaml')
    assert only_reload is False

    logging.info("Checking the output folders, the stale selection is removed and the Options dialog is locked")
    plan_dir = str(m2m / 'PlanTUS') + '/'
    assert runner.PlanTUSRoot.replace('\\', '/') == plan_dir.replace('\\', '/')
    assert runner.PlanOutputPath.endswith(f"{traj.stem}_PlanTUSMask_T0")
    assert not stale.exists(), "previous BabelBrain_trajectory.txt must be removed before running"
    assert not runner.OptionsDlg.isEnabled()


@pytest.mark.parametrize('plantus_root', ['...', 'missing_folder', 'custom'])
def test_Execute_plantus_root(qtbot, monkeypatch, tmp_path, plantus_root):
    """A custom PlanTUS folder is used only when it has PlanTUS_wrapper.py, else the built-in one."""
    t1, m2m, traj = ph.synthetic_plantus_inputs(tmp_path)
    expected = str(ph.PLANTUS_BUNDLED_ROOT)
    if plantus_root == 'custom':
        # A folder with PlanTUS_wrapper.py counts as a PlanTUS installation
        custom = tmp_path / 'MyPlanTUS'
        custom.mkdir()
        (custom / 'PlanTUS_wrapper.py').touch()
        plantus_root = expected = str(custom)
    elif plantus_root == 'missing_folder':
        plantus_root = str(tmp_path / 'missing_folder')

    logging.info(f"Running Execute with PlanTUS folder {plantus_root!r}, expecting {expected}")
    runner = ph.make_run_plan_tus(qtbot, monkeypatch, TX_NAME, FREQ, t1, m2m, traj, plantus_root=plantus_root)
    ph.execute(runner)
    assert runner.planTUSargs[3] == expected


@pytest.mark.parametrize('answer,only_reload', [('Yes', False), ('No', True)])
def test_Execute_existing_results(rp, qtbot, monkeypatch, tmp_path, answer, only_reload):
    """With PlanTUS results for the target the user chooses between recalculating and reloading."""
    logging.info("Leaving PlanTUS results for the target from a previous run")
    t1, m2m, traj = ph.synthetic_plantus_inputs(tmp_path)
    out = m2m / 'PlanTUS' / f"{traj.stem}_PlanTUSMask_T0"
    out.mkdir(parents=True)
    (out / 'distances_skin.func.gii').touch()
    asked = []

    def question(*args, **kwargs):
        asked.append(args)
        return getattr(rp.QMessageBox, answer)

    logging.info(f"Running Execute and answering {answer!r} to recalculating, PlanTUS should "
                 f"{'reload' if only_reload else 'recalculate'}")
    monkeypatch.setattr(rp.QMessageBox, 'question', question)
    runner = ph.make_run_plan_tus(qtbot, monkeypatch, TX_NAME, FREQ, t1, m2m, traj)
    ph.execute(runner)
    assert len(asked) == 1
    assert runner.planTUSargs[-1] is only_reload


def test_GenerateTrajectory(qtbot, monkeypatch, synthetic_inputs):
    """Placement for a selected vertex reruns PlanTUS with the same arguments."""
    logging.info("Running Execute, then generating the placement for vertex 1234")
    runner = ph.make_run_plan_tus(qtbot, monkeypatch, TX_NAME, FREQ, *synthetic_inputs)
    ph.execute(runner)
    runner.GenerateTrajectory(1234)
    runner.CaltimerTUSPlan.stop()  # the timer waiting for PlanTUS isn't needed (no real process)

    logging.info("Checking PlanTUS was started again with the same arguments, for that vertex only")
    process = ph.FakeProcess.created[-1]
    assert process.started and process.args == runner.planTUSargs
    assert process.kwargs == {'runOnlyTrajectory': 1234}
    assert runner.RunningTUSPlan is False


# ================================================================================================================================
# RUN PLANTUS BACKGROUND
# ================================================================================================================================
class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def text(self):
        return ''.join(self.items)


def _run_background(rp, monkeypatch, run, bOnlyReload=False, runOnlyTrajectory=-1):
    """Run RunPlanTUSBackground with subprocess.run replaced by `run`. Returns the subprocess.run
    calls and what it wrote to its queue. (No logging in here: live logging would undo its
    stdout redirect, see plantus_helpers.run_plantus_headless.)"""
    calls = []

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return run(args, **kwargs)

    monkeypatch.setattr(rp.subprocess, 'run', fake_run)
    out = _Queue()
    stdout = sys.stdout
    try:
        rp.RunPlanTUSBackground(out, '/scripts/', '/simnibs', '/plantus', '/data/T1.nii.gz', '/data/m2m/synthetic.msh',
                                '/data/mask.nii.gz', '/data/PlanTUSTxConfig.yaml', bOnlyReload,
                                runOnlyTrajectory=runOnlyTrajectory)
    finally:
        sys.stdout = stdout  # it redirects stdout to the queue for good (BabelBrain runs it in its own process)
    return calls, out.text()


def _ok(args, **kwargs):
    return subprocess.CompletedProcess(args, 0, stdout='PlanTUS stdout', stderr='PlanTUS stderr')


@pytest.mark.parametrize('bOnlyReload,runOnlyTrajectory,extra', [
    (False, -1, ['--overwrite']),
    (True, -1, ['--reuse_existing']),
    (True, 4321, ['--reuse_existing', '--placement_only', '4321']),
])
def test_RunPlanTUSBackground_command(rp, monkeypatch, bOnlyReload, runOnlyTrajectory, extra):
    logging.info(f"Running RunPlanTUSBackground (reload={bOnlyReload}, vertex={runOnlyTrajectory}) "
                 "with PlanTUS replaced by a successful run")
    calls, log = _run_background(rp, monkeypatch, _ok, bOnlyReload, runOnlyTrajectory)

    logging.info(f"Checking the platform run script gets the six inputs followed by {extra}")
    assert len(calls) == 1
    args, kwargs = calls[0]
    inputs = ['/simnibs', '/plantus', '/data/T1.nii.gz', '/data/m2m/synthetic.msh', '/data/mask.nii.gz',
              '/data/PlanTUSTxConfig.yaml']
    if sys.platform == 'linux':
        assert args[:2] == ['bash', '/scripts/run_linux.sh']
    elif sys.platform == 'darwin':
        assert args[:2] == ['zsh', '/scripts/run_mac.sh']
    else:
        assert args[0].endswith('run_win.bat')
    assert args[-len(inputs) - len(extra):] == inputs + extra
    assert kwargs['check'] is True and kwargs['capture_output'] is True

    logging.info("Checking PlanTUS' output and the success marker reach BabelBrain's queue")
    assert 'PlanTUS stdout' in log and 'PlanTUS stderr' in log
    assert '--Babel-Brain-Success' in log
    assert '--Babel-Brain-Low-Error' not in log


def test_RunPlanTUSBackground_plantus_error(rp, monkeypatch):
    def fail(args, **kwargs):
        raise subprocess.CalledProcessError(1, args, output='partial output', stderr='Traceback: PlanTUS failed')

    logging.info("Running RunPlanTUSBackground with PlanTUS exiting with an error")
    _, log = _run_background(rp, monkeypatch, fail)

    logging.info("Checking the error marker and PlanTUS' output reach BabelBrain's queue")
    assert '--Babel-Brain-Low-Error' in log
    assert 'Traceback: PlanTUS failed' in log and 'partial output' in log
    assert '--Babel-Brain-Success' not in log


def test_RunPlanTUSBackground_launch_error(rp, monkeypatch):
    def fail(args, **kwargs):
        raise FileNotFoundError('zsh not found')

    logging.info("Running RunPlanTUSBackground with the run script failing to start")
    _, log = _run_background(rp, monkeypatch, fail)

    logging.info("Checking the error marker and the reason reach BabelBrain's queue")
    assert '--Babel-Brain-Low-Error' in log and 'zsh not found' in log
    assert '--Babel-Brain-Success' not in log


def test_run_scripts_forward_arguments():
    """The bundled scripts pass the six inputs and the optional arguments on to PlanTUS_wrapper.py."""
    for script, inputs, optional in [('run_mac.sh', '"$3" "$4" "$5" "$6"', '$7 $8 $9'),
                                     ('run_linux.sh', '"$3" "$4" "$5" "$6"', '$7 $8 $9'),
                                     ('run_win.bat', '"%~3" "%~4" "%~5" "%~6"', '%7 %8 %9')]:
        logging.info(f"Checking {script} calls PlanTUS_wrapper.py with the inputs and optional arguments")
        text = (ph.PLANTUS_SCRIPTS_DIR / script).read_text()
        assert f'PlanTUS_wrapper.py" {inputs} --babelbrain_integration {optional}' in text, script


def test_wrapper_accepts_babelbrain_arguments():
    """PlanTUS_wrapper.py has the options RunPlanTUSBackground and the run scripts pass."""
    text = WRAPPER.read_text()
    for option in ('--babelbrain_integration', '--overwrite', '--reuse_existing', '--placement_only', '--skip_viewer'):
        logging.info(f"Checking PlanTUS_wrapper.py defines {option}")
        assert f'"{option}"' in text, option


# ================================================================================================================================
# CHECK QUEUE: PLANTUS RESULT BACK INTO BABELBRAIN
# ================================================================================================================================
def _fake_plantus_result(runner, trajectory_type, traj, shift=(3.0, -2.0, 1.0)):
    """Write what PlanTUS leaves after the user accepts a placement: the trajectory in
    <output>/<ID>/ and its copy BabelBrain_trajectory.txt. Returns the trajectory matrix."""
    from ConvMatTransform import WriteTrajectoryBrainsight

    mat = ph.read_trajectory(trajectory_type, traj, runner.IDIndex).copy()
    mat[:3, 3] += shift
    out = Path(runner.PlanOutputPath) / runner.ID
    out.mkdir(parents=True, exist_ok=True)
    name = Path(runner.PlanOutputPath).name
    traj_file = out / f"{name}_{runner.ID}_Trajectory_BabelBrain.txt"
    WriteTrajectoryBrainsight(str(traj_file), [mat], [f"{name}_{runner.ID}"], 'NIfTI:S:Aligned')
    (out / f"{name}_{runner.ID}_TransducerModel.surf.gii").touch()
    shutil.copy(traj_file, runner.PlanTUSBabelBrainResult)
    return mat


def _prepared_runner(qtbot, monkeypatch, tmp_path, trajectory_type='brainsight', combined=False, id_index=0):
    t1, m2m, traj = ph.synthetic_plantus_inputs(tmp_path, trajectory_type, combined)
    runner = ph.make_run_plan_tus(qtbot, monkeypatch, TX_NAME, FREQ, t1, m2m, traj,
                                  trajectory_type=trajectory_type, id_index=id_index)
    ph.execute(runner)
    return runner, traj


@pytest.mark.parametrize('trajectory_type,combined,id_index', [
    ('brainsight', False, 0),
    ('brainsight', True, 1),
    ('slicer', False, 0),
    ('slicer', True, 1),
])
def test_check_queue_TUSPlan_accepts_trajectory(qtbot, monkeypatch, tmp_path, trajectory_type, combined, id_index):
    """The accepted PlanTUS placement replaces the selected trajectory in a new _PlanTUS file."""
    logging.info(f"Running Execute on a {'combined' if combined else 'single'} {trajectory_type} "
                 f"trajectory file, trajectory {id_index} selected")
    runner, traj = _prepared_runner(qtbot, monkeypatch, tmp_path, trajectory_type, combined, id_index)
    others = {i: ph.read_trajectory(trajectory_type, traj, i)
              for i in range(len(runner.MainApp.Config['ID'])) if i != id_index}

    logging.info("Writing a PlanTUS placement (target shifted a few mm) as if the user accepted it")
    expected = _fake_plantus_result(runner, trajectory_type, traj)

    logging.info("Reporting PlanTUS success to check_queue_TUSPlan and accepting the final results dialog")
    shown = ph.finish_plantus(runner, monkeypatch)
    assert runner.OptionsDlg.isEnabled()
    assert not shown['critical']
    assert len(shown['final_results']) == 1 and shown['final_results'][0].endswith('_TransducerModel.surf.gii')
    assert len(shown['information']) == 1  # reminder to run Step 1 again

    logging.info("Checking the new _PlanTUS trajectory file: selected trajectory replaced, others unchanged")
    final = Path(runner.MainApp.Config['Mat4Trajectory'])
    assert final.name == f"{traj.stem}_PlanTUS{traj.suffix}"
    assert final.parent == traj.parent and final.is_file()
    np.testing.assert_allclose(ph.read_trajectory(trajectory_type, final, id_index), expected, atol=1e-6)
    for i, mat in others.items():
        np.testing.assert_allclose(ph.read_trajectory(trajectory_type, final, i), mat, atol=1e-6)
    if trajectory_type == 'brainsight':
        assert ph.trajectory_ids(trajectory_type, final)[id_index] == runner.ID

    logging.info("Checking BabelBrain's config and window were updated")
    assert runner.MainApp.Config['ID'][id_index] == runner.ID
    assert runner.MainApp._bTrajAlreadyModified is False
    runner.MainApp.UpdateWindowTitle.assert_called_once()
    # Only Brainsight trajectories are synced back to Brainsight
    if trajectory_type == 'brainsight':
        index, mat = runner.MainApp.NotifyBrainsightTrajectoryReplaced.call_args.args
        assert index == id_index
        np.testing.assert_allclose(mat, expected, atol=1e-6)
    else:
        runner.MainApp.NotifyBrainsightTrajectoryReplaced.assert_not_called()


def test_check_queue_TUSPlan_trajectory_rejected(qtbot, monkeypatch, tmp_path):
    """Cancelling the final results dialog keeps the original trajectory."""
    logging.info("Running Execute and writing a PlanTUS placement")
    runner, traj = _prepared_runner(qtbot, monkeypatch, tmp_path)
    _fake_plantus_result(runner, 'brainsight', traj)

    logging.info("Reporting PlanTUS success and cancelling the final results dialog")
    shown = ph.finish_plantus(runner, monkeypatch, confirm=False)

    logging.info("Checking the original trajectory is kept and no _PlanTUS file was written")
    assert len(shown['final_results']) == 1
    assert runner.MainApp.Config['Mat4Trajectory'] == str(traj)
    assert not (traj.parent / f"{traj.stem}_PlanTUS.txt").exists()


def test_check_queue_TUSPlan_no_selection(qtbot, monkeypatch, tmp_path):
    """PlanTUS closed without choosing a placement: nothing changes and no error is shown."""
    logging.info("Running Execute, then reporting PlanTUS success without a placement")
    runner, traj = _prepared_runner(qtbot, monkeypatch, tmp_path)
    shown = ph.finish_plantus(runner, monkeypatch)

    logging.info("Checking nothing was shown, the trajectory is unchanged and the Options dialog is unlocked")
    assert shown == {'final_results': [], 'critical': [], 'information': []}
    assert runner.MainApp.Config['Mat4Trajectory'] == str(traj)
    assert runner.OptionsDlg.isEnabled()


def test_check_queue_TUSPlan_missing_trajectory(qtbot, monkeypatch, tmp_path):
    """A selection without the PlanTUS trajectory file is reported as an incomplete execution."""
    logging.info("Running Execute and writing BabelBrain_trajectory.txt without PlanTUS' trajectory files")
    runner, traj = _prepared_runner(qtbot, monkeypatch, tmp_path)
    Path(runner.PlanTUSRoot).mkdir(parents=True, exist_ok=True)
    Path(runner.PlanTUSBabelBrainResult).write_text('selected')

    logging.info("Reporting PlanTUS success, expecting an error and the trajectory unchanged")
    shown = ph.finish_plantus(runner, monkeypatch)
    assert shown['critical'] == ["We couldn't find any PlanTUS generated trajectory."]
    assert runner.MainApp.Config['Mat4Trajectory'] == str(traj)


def test_check_queue_TUSPlan_error(qtbot, monkeypatch, tmp_path):
    """A PlanTUS error is reported and the Options dialog is enabled again."""
    logging.info("Running Execute and writing a PlanTUS placement")
    runner, traj = _prepared_runner(qtbot, monkeypatch, tmp_path)
    _fake_plantus_result(runner, 'brainsight', traj)

    logging.info("Reporting a PlanTUS error, the placement must be ignored")
    shown = ph.finish_plantus(runner, monkeypatch, messages=('Starting PlanTUS\n', '--Babel-Brain-Low-Error\n'))
    assert shown['critical'] == ['An error occurred during execution of PlanTUS.']
    assert not shown['final_results']
    assert runner.MainApp.Config['Mat4Trajectory'] == str(traj)
    assert runner.OptionsDlg.isEnabled()


def test_check_queue_TUSPlan_still_running(qtbot, monkeypatch, tmp_path):
    """Progress messages alone don't finish the run."""
    logging.info("Running Execute, then passing only a progress message to check_queue_TUSPlan")
    runner, traj = _prepared_runner(qtbot, monkeypatch, tmp_path)
    shown = ph.finish_plantus(runner, monkeypatch, messages=('Starting PlanTUS\n',))

    logging.info("Checking nothing was shown and the Options dialog stays locked")
    assert shown == {'final_results': [], 'critical': [], 'information': []}
    assert not runner.OptionsDlg.isEnabled()
