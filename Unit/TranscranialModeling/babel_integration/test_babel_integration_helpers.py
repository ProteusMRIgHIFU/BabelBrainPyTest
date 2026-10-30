import logging
import os

import nibabel
import pytest

# Module added after v0.8.1, skip on versions that don't have it
BIHelpers = pytest.importorskip('TranscranialModeling.babel_integration.babel_integration_helpers')

def test_SaveNiftiEnforcedISO(dataset,load_files,check_data,tmp_path):

    # Load inputs
    logging.info("Loading T1W")
    input_files = {'T1W': dataset['T1_path']}
    input_data = load_files(input_files)

    # Run SaveNiftiEnforcedISO
    output_fname_initial = str(tmp_path) + os.sep + 'T1W-isotropic__.nii.gz'
    output_fname_final = str(tmp_path) + os.sep + 'T1W-isotropic.nii.gz'
    logging.info(f"Input T1W: shape {input_data['T1W'].shape}, voxel size {input_data['T1W'].header.get_zooms()[:3]}")
    logging.info("Running SaveNiftiEnforcedISO")
    BIHelpers.SaveNiftiEnforcedISO(input_data['T1W'], output_fname_initial)

    # Check intermediate files are deleted (only warns, leftovers don't fail the test)
    logging.info("Checking uncompressed intermediate files were cleaned up")
    intermediate_files = [output_fname_initial.split('.gz')[0], output_fname_final.split('.gz')[0]]
    for fname in intermediate_files:
        if os.path.exists(fname):
            logging.warning(f"temporary intermediate file ({fname}) was not deleted")

    # Load output
    enforced_iso_nifti = nibabel.load(output_fname_final)
    logging.info(f"Output: shape {enforced_iso_nifti.shape}, voxel size {enforced_iso_nifti.header.get_zooms()[:3]}")

    # Check that data is isometric
    logging.info("Checking output voxels are isometric")
    isometric = check_data['isometric'](enforced_iso_nifti)

    assert isometric, "Data is not isometric"