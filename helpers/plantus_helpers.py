"""
Helpers for the PlanTUS tests (Unit/BabelBrain/PlanTUSViewer/test_RunPlanTUS.py and
Integration/test_plantus.py).

RUN_PLAN_TUS (PlanTUSViewer/RunPlanTUS.py) is normally created by the Options dialog with the
BabelBrain main window as MainApp. The tests build a stand-in for both (make_run_plan_tus) holding
only what RUN_PLAN_TUS reads, so Execute and check_queue_TUSPlan run unchanged: on the synthetic head
(synthetic_plantus_inputs) in the unit tests, on ernie (copy_ernie_inputs) in the integration test.

PlanTUS always opens its interactive viewer unless --skip_viewer is passed, which BabelBrain never
does. run_plantus_headless runs BabelBrain's RunPlanTUSBackground (and so the bundled run_*.sh /
run_win.bat scripts and the real PlanTUS) with --skip_viewer added to the command line.
"""
import logging
import os
import queue
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import nibabel
import numpy as np
import yaml

from Tests.helpers.parameters import (BABELBRAIN_SRC, DATASETS, SYNTHETIC_DATASET, TRANSDUCERS_BY_NAME,
                                      trajectory_file)

PLANTUS_SCRIPTS_DIR = BABELBRAIN_SRC / 'ExternalBin' / 'PlanTUS'
PLANTUS_BUNDLED_ROOT = PLANTUS_SCRIPTS_DIR / 'PlanTUS'

# Files PlanTUS writes in <m2m>/PlanTUS/<mask name>/ for the planning stage (its own
# _expected_output_files list), the ones BabelBrain's viewer shows are in RUN_PLAN_TUS.showTUSPlanViewer
PLANTUS_PLANNING_OUTPUTS = [
    'skin.surf.gii',
    'skull.surf.gii',
    'distances_skin.func.gii',
    'distances_skin_thresholded.func.gii',
    'target_intersection_skin.func.gii',
    'angles_skin.func.gii',
    'skin_skull_angles_skin.func.gii',
]

# Modules PlanTUS imports from simnibs_env (PlanTUS README, "Dependencies"), import name -> pip name.
# SimNIBS doesn't ship them, they have to be pip installed in simnibs_env (PlanTUS v2.0.0 added trimesh and vtk)
PLANTUS_REQUIRED_MODULES = {'nibabel': 'nibabel', 'nilearn': 'nilearn', 'yaml': 'pyyaml', 'h5py': 'h5py',
                            'trimesh': 'trimesh', 'vtk': 'vtk'}

def missing_plantus_modules(simnibs_root):
    """PLANTUS_REQUIRED_MODULES that don't import with the simnibs_env python, as (python, [pip names])."""
    env = Path(simnibs_root) / 'simnibs_env'
    python = next((p for p in (env / 'bin' / 'python', env / 'python.exe', env / 'Scripts' / 'python.exe')
                   if p.is_file()), None)
    if python is None:
        return None, list(PLANTUS_REQUIRED_MODULES.values())
    check = ("import importlib.util\n"
             f"print(' '.join(m for m in {list(PLANTUS_REQUIRED_MODULES)} if importlib.util.find_spec(m) is None))")
    result = subprocess.run([str(python), '-c', check], capture_output=True, text=True)
    return python, [PLANTUS_REQUIRED_MODULES[m] for m in result.stdout.split()]


def transducer_config(tx_name):
    """Transducer config as BabelBrain loads it into AcSim.Config (default.yaml of the transducer)."""
    tx = TRANSDUCERS_BY_NAME[tx_name]
    with open(BABELBRAIN_SRC / 'babel_transducers' / tx['transducer_type'] / tx['module_name'] / 'default.yaml') as f:
        return yaml.safe_load(f)


# Targets of the synthetic head (synthetic_dataset.py, about 40 mm across, centred on the origin):
# name -> (RAS position mm, tilt of the trajectory about the y axis in degrees)
SYNTHETIC_TARGETS = {'target_left': ((-6.0, 2.0, 6.0), -20.0), 'target_right': ((6.0, -2.0, 6.0), 20.0)}


def _synthetic_trajectory(position, tilt):
    """4x4 trajectory (Brainsight convention) at `position`, tilted about the y axis."""
    c, s = np.cos(np.radians(tilt)), np.sin(np.radians(tilt))
    mat = np.eye(4)
    mat[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    mat[:3, 3] = position
    return mat


def _write_slicer_trajectory(fname, mat):
    """3D Slicer linear transform file for `mat`, written the way RUN_PLAN_TUS writes them."""
    from ConvMatTransform import BSight_to_itk, templateSlicer
    t = BSight_to_itk(mat)
    t[:3, :3] = t[:3, :3].T
    with open(fname, 'w') as f:
        f.write(templateSlicer.format(m0n0=t[0, 0], m0n1=t[1, 0], m0n2=t[2, 0], m1n0=t[0, 1], m1n1=t[1, 1],
                                      m1n2=t[2, 1], m2n0=t[0, 2], m2n1=t[1, 2], m2n2=t[2, 2],
                                      X=t[0, 3], Y=t[1, 3], Z=t[2, 3]))


def synthetic_plantus_inputs(dest, trajectory_type='brainsight', combined=False):
    """Inputs RUN_PLAN_TUS writes next to, made in `dest` from the synthetic head: its T1
    (PlanTUSTxConfig.yaml goes next to it), an empty placeholder .msh in the charm folder (Execute
    only looks it up, PlanTUS writes its outputs next to it) and a Brainsight or Slicer trajectory
    file (the target mask and _PlanTUS trajectory go next to it) with target_left, or with both
    SYNTHETIC_TARGETS when combined.

    Returns (t1 path, m2m folder, trajectory path)."""
    from ConvMatTransform import WriteTrajectoryBrainsight

    m2m = Path(dest) / 'm2m_synthetic'
    m2m.mkdir(parents=True, exist_ok=True)
    t1 = m2m / 'T1W.nii.gz'
    shutil.copy(SYNTHETIC_DATASET['T1_path'], t1)
    (m2m / 'synthetic.msh').touch()

    traj_dir = Path(dest) / 'Trajectories'
    traj_dir.mkdir(exist_ok=True)
    names = list(SYNTHETIC_TARGETS) if combined else ['target_left']
    mats = [_synthetic_trajectory(*SYNTHETIC_TARGETS[n]) for n in names]
    stem = f"{trajectory_type}_synthetic_{'targets' if combined else names[0]}"
    if trajectory_type == 'brainsight':
        traj = traj_dir / f"{stem}.txt"
        WriteTrajectoryBrainsight(str(traj), mats, names, 'NIfTI:S:Aligned')
    else:
        files = {}
        for name, mat in zip(names, mats):
            files[name] = str(traj_dir / f"slicer_synthetic_{name}.txt")
            _write_slicer_trajectory(files[name], mat)
        if combined:  # combined Slicer targets: yaml listing the individual files
            traj = traj_dir / f"{stem}.yaml"
            with open(traj, 'w') as f:
                yaml.dump(files, f)
        else:
            traj = Path(files[names[0]])
    return t1, m2m, traj


def copy_ernie_inputs(dest):
    """Copy the ernie inputs of a real PlanTUS run into `dest`: the T1, the Brainsight superficial
    left target, and the charm folder's mesh with the segmentation and EEG fiducials PlanTUS reads
    next to it (it also writes its outputs there).

    Returns (t1 path, m2m folder, trajectory path)."""
    ds = DATASETS['ernie']
    t1_src = Path(ds['T1_path'])
    m2m_src = Path(ds['m2m_folder_path'])
    traj_src = trajectory_file('brainsight', 'ernie', 'Superficial_Target_Left')
    msh_src = next(iter(m2m_src.glob('*.msh')), None)
    for path in (t1_src, traj_src, msh_src, m2m_src / 'final_tissues.nii.gz'):
        if path is None or not path.exists():
            import pytest
            pytest.skip(f"Missing PlanTUS input {path or m2m_src / '*.msh'}")

    m2m = Path(dest) / m2m_src.name
    m2m.mkdir(parents=True, exist_ok=True)
    t1 = m2m / t1_src.name
    shutil.copy(t1_src, t1)
    shutil.copy(msh_src, m2m / msh_src.name)
    shutil.copy(m2m_src / 'final_tissues.nii.gz', m2m / 'final_tissues.nii.gz')
    if (m2m_src / 'eeg_positions').is_dir():
        shutil.copytree(m2m_src / 'eeg_positions', m2m / 'eeg_positions', dirs_exist_ok=True)

    traj_dir = Path(dest) / 'Trajectories'
    traj_dir.mkdir(exist_ok=True)
    traj = traj_dir / traj_src.name
    shutil.copy(traj_src, traj)
    return t1, m2m, traj


def _read(trajectory_type, traj_path):
    from ConvMatTransform import ReadTrajectoryBrainsight, read_converted_itk_affine_transform
    reader = ReadTrajectoryBrainsight if trajectory_type == 'brainsight' else read_converted_itk_affine_transform
    return reader(str(traj_path), bGetID=True)


def trajectory_ids(trajectory_type, traj_path):
    """Target names in a trajectory file, as BabelBrain puts them in Config['ID']."""
    ids = _read(trajectory_type, traj_path)[1]
    return list(ids) if isinstance(ids, (list, tuple)) else [ids]


def read_trajectory(trajectory_type, traj_path, index=0):
    """4x4 trajectory matrix (Brainsight convention) as RUN_PLAN_TUS.Execute reads it."""
    mat = _read(trajectory_type, traj_path)[0]
    return mat[:, :, index] if mat.ndim == 3 else mat


class FakeProcess:
    """Stands in for multiprocessing.Process in RunPlanTUS, recording what would be run."""
    created = []

    def __init__(self, target=None, args=(), kwargs=None):
        self.target, self.args, self.kwargs = target, args, kwargs or {}
        self.started = False
        FakeProcess.created.append(self)

    def start(self):
        self.started = True

    def join(self):
        pass


def _text(value):
    return SimpleNamespace(text=lambda: value)


def _value(value):
    return SimpleNamespace(value=lambda: value)


def make_run_plan_tus(qtbot, monkeypatch, tx_name, freq, t1, m2m, traj, trajectory_type='brainsight',
                      id_index=0, skin_distance=0.0, plantus_root='', simnibs_root='', volume_roi=''):
    """RUN_PLAN_TUS for `tx_name` at `freq` on the given inputs, with stand-ins for the
    BabelBrain main window (MainApp) and the Options dialog (OptionsDlg). The PlanTUS process is
    replaced by FakeProcess (RUN_PLAN_TUS.CalProcess), so Execute only prepares the run."""
    from PySide6.QtWidgets import QWidget
    import PlanTUSViewer.RunPlanTUS as RunPlanTUS

    monkeypatch.setattr(RunPlanTUS, 'Process', FakeProcess)

    ids = trajectory_ids(trajectory_type, traj)
    main_app = MagicMock(name='BabelBrain')
    main_app.Config = {
        'TxSystem': tx_name,
        'TrajectoryType': trajectory_type,
        'OrigMat4Trajectory': str(traj),
        'Mat4Trajectory': str(traj),
        'T1W': str(t1),
        'simbnibs_path': str(m2m),
        'ID': list(ids),
    }
    main_app.AcSim.Config = transducer_config(tx_name)
    main_app.AcSim.GetTxModel.return_value = 'BSonix35mm'
    main_app.AcSim.Widget.SkinDistanceSpinBox.value.return_value = skin_distance
    main_app.Widget.USMaskkHzDropDown.property.return_value = freq
    main_app._bTrajAlreadyModified = True

    options = QWidget()
    qtbot.addWidget(options)
    options.ui = SimpleNamespace(
        PlanTUSRootlineEdit=_text(plantus_root),
        SimbNINBSRootlineEdit=_text(simnibs_root),
        VolumeROIPlanTUSlineEdit=_text(volume_roi),
        SkinDistanceSpinBox=_value(skin_distance),
    )

    runner = RunPlanTUS.RUN_PLAN_TUS(main_app, options, ids[id_index], id_index)
    return runner


def execute(runner):
    """RUN_PLAN_TUS.Execute, stopping the queue timer it starts (the process is a FakeProcess).
    Returns the PlanTUSTxConfig.yaml it wrote."""
    runner.Execute()
    runner.CaltimerTUSPlan.stop()
    with open(Path(runner.MainApp.Config['T1W']).parent / 'PlanTUSTxConfig.yaml') as f:
        return yaml.safe_load(f)


def run_plantus_headless(monkeypatch, plantus_args, runOnlyTrajectory=-1):
    """Run BabelBrain's RunPlanTUSBackground in this process with the arguments RUN_PLAN_TUS
    prepared (runner.planTUSargs), adding --skip_viewer to the PlanTUS command line so it doesn't
    block on its viewer. The run_*.sh / run_win.bat scripts forward only three optional arguments,
    so --placement_only N is passed as --placement_only=N to make room for --skip_viewer.

    Returns RunPlanTUSBackground's output (what BabelBrain reads from its queue)."""
    import PlanTUSViewer.RunPlanTUS as RunPlanTUS

    real_run = subprocess.run

    def run_without_viewer(args, **kwargs):
        args = list(args)
        if '--placement_only' in args:
            i = args.index('--placement_only')
            args[i:i + 2] = [f'--placement_only={args[i + 1]}']
        args.append('--skip_viewer')
        # Live logging (log_cli in pytest.ini) suspends and resumes pytest's output capture, which
        # puts pytest's own sys.stdout back over BabelBrain's redirect to the queue
        redirect = sys.stdout
        logging.info(f"Running {' '.join(map(str, args))}")
        sys.stdout = redirect
        return real_run(args, **kwargs)

    out = queue.Queue()
    stdout = sys.stdout
    try:
        with monkeypatch.context() as m:
            m.setattr(RunPlanTUS.subprocess, 'run', run_without_viewer)
            RunPlanTUS.RunPlanTUSBackground(out, *plantus_args[1:], runOnlyTrajectory=runOnlyTrajectory)
    finally:
        sys.stdout = stdout  # it redirects stdout to the queue for good (BabelBrain runs it in its own process)
    messages = []
    while not out.empty():
        messages.append(out.get())
    log = ''.join(messages)
    for line in log.splitlines():
        if line.strip():
            logging.debug(f"[PlanTUS] {line.rstrip()}")
    return log


def finish_plantus(runner, monkeypatch, messages=('--Babel-Brain-Success\n',), confirm=True):
    """Feed RUN_PLAN_TUS.check_queue_TUSPlan the end of a PlanTUS run, as its timer does once the
    process reports back. The final results dialog is replaced by returning `confirm`, and message
    boxes don't open. Returns what was shown: {'final_results': [surface], 'critical': [text],
    'information': [text]}."""
    import PlanTUSViewer.RunPlanTUS as RunPlanTUS

    shown = {'final_results': [], 'critical': [], 'information': []}

    def show_final_results(surface):
        shown['final_results'].append(surface)
        return confirm

    monkeypatch.setattr(runner, 'showFinalResults', show_final_results)
    monkeypatch.setattr(RunPlanTUS.QMessageBox, 'information',
                        lambda parent, title, text, *a, **k: shown['information'].append(text))
    monkeypatch.setattr(RunPlanTUS.QMessageBox, 'exec',
                        lambda box: shown['critical'].append(box.text()))

    runner.CalQueue = queue.Queue()
    for msg in messages:
        runner.CalQueue.put(msg)
    if runner.CalProcess is None:
        runner.CalProcess = FakeProcess()
    runner.check_queue_TUSPlan()
    return shown


def mask_center_ras(mask_path):
    """Centre of mass (RAS, mm) of a mask."""
    img = nibabel.load(str(mask_path))
    ijk = np.argwhere(np.asarray(img.dataobj) > 0).mean(axis=0)
    return nibabel.affines.apply_affine(img.affine, ijk)


def distance_point_to_line(point, line_point, direction):
    direction = np.asarray(direction) / np.linalg.norm(direction)
    v = np.asarray(point) - np.asarray(line_point)
    return float(np.linalg.norm(v - np.dot(v, direction) * direction))
