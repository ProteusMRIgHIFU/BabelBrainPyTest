### Tests Ran

- unit
- integration
- e2e tests
- regression tests comparing
  - v0.8.8
    - branch `UserTxServerMode` commit `54370e90ed7a678485ff038e918247ecbf920c3c`
  - v0.8.1
    - branch `main` commit `ca32b99dd3b7be725442989bad47cead11a6f8be`

### Test Notes

- Between these two BabelBrain versions, the TPO min value for CTX500 was changed. To make comparisons easier, the TPO min value in v0.8.8 was changed to match v0.8.1 just for the tests
- In BabelBrain v0.8.1, there was a bug where deprecated trimesh functions were called to help clean up meshes which resulted in errors. These functions were not called when using ID0082 dataset, but tests now incorporated ernie dataset which now triggers their call. To get outputs for v0.8.1, the environment used to run the tests downgraded trimesh to a compatible version to make those deprecated calls.

### Test Failures

#### Major Release Unit Tests

- Resample tests → DICE score is ~0.9999,  but not quite 1.0 so they fail due to strict tolerance
- Voxelization tests → DICE score is about 0.93 while passing is 0.95, It seems CPU actually picks up extra voxels on the surface whereas GPU seems to produce a cleaner result, this is more evident on the tests that passed using synthetic dataset
- DoIntersect → This always seems to fail when trying to do an intersect between two spheres USING blender. It’s usually fine for manifold method (except a failure on Linux), but we don’t see this issue for real datasets.

#### Major Release Integration Tests

- Nothing to report

#### Major Release Generate Regression Outputs

- Intensive sim tests failed due to resource limitations on mac and windows
- Soft tissue segmentation tests
  - On linux, in version v0.8.1, issue with call that is resolved in v0.8.8

#### Major Release E2E Tests

- Custom Transducer vs Corresponding Default Transducer tests
  - For CTX500, issue in pytest setup where in some instances, the TPO value in the custom tx file was not updated to match old value
  - For REMOPD, even if we use slicer for trajectory to avoid the additional flip done to the transducer, REMOPD still had a z rotation applied to it as well that the custom tx never gets so differences are inherent

#### Major Release Regression Tests

- Tests involving TheraPhas (formerly REMOPD), errors to be expected because of new orientation. TherPhas has been validated with real world experiments
- Normal pipeline for H317 and DomeTx, differences are negligable (<1%)
- Intensive sims fail if their files couldnt be generated
- GetSkullMaskFrom SimbNIBS
  - For CUDA, there were major differences resulting from enforcing dtype to be float32 in commit `92b8387c104057c79702b015ff4a38b7de94c751` for maps. Before CUDA allowed float64 explainging the large differences. Shift to float32 is to save memory.