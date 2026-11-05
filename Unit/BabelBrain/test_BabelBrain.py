import logging

import nibabel
import numpy as np
import pytest

from BabelBrain import BabelBrain

@pytest.mark.step1
def test_save_T1W_iso(dataset,load_files,check_data,get_mpl_plot,compare_data,tmp_path,request):

    # Load inputs
    logging.info("Loading T1W")
    input_files = {'T1W': dataset['T1_path']}
    input_data = load_files(input_files)

    # Run save_T1W_iso function
    output_fname = str(tmp_path / 'T1W-isotropic.nii.gz')
    logging.info(f"Input T1W: shape {input_data['T1W'].shape}, voxel size {input_data['T1W'].header.get_zooms()[:3]}")
    logging.info("Running save_T1W_iso")
    BabelBrain.save_T1W_iso(input_files['T1W'], output_fname)

    # Load output
    T1W_iso = nibabel.load(output_fname)
    logging.info(f"Output: shape {T1W_iso.shape}, voxel size {T1W_iso.header.get_zooms()[:3]}")

    # Check that data is isometric
    logging.info("Checking output voxels are isometric")
    isometric = check_data['isometric'](T1W_iso)
    if not isometric:
        pytest.fail("T1W data is not isometric")

    # Check origin has not changed
    logging.info(f"Checking origin is unchanged: {input_data['T1W'].affine[:3,3]} vs {T1W_iso.affine[:3,3]}")
    same_origin = input_data['T1W'].affine[:3,3] == T1W_iso.affine[:3,3]
    same_origin = np.prod(same_origin)
    if not same_origin:
        pytest.fail("T1W data has changed origin")

    # Load data
    T1W_data = input_data['T1W'].get_fdata()
    T1W_iso_data = T1W_iso.get_fdata()
    
    # Save plot screenshot to be added to html report later
    logging.info("Saving input vs isotropic T1W plot for the report")
    request.node.screenshots = []
    plots = [T1W_data, T1W_iso_data]
    plot_names = ['T1W data', 'T1W Iso Data']
    screenshot = get_mpl_plot(plots, axes_num=2,titles=plot_names,color_map='gray')
    request.node.screenshots.append(screenshot)

    # Calculate Bhattacharyya Coefficient, output is between 0 and 1
    bhatt_coefficient = compare_data['bhatt_coeff'](T1W_data, T1W_iso_data)
    logging.info(f"Checking intensity histograms match, Bhattacharyya coefficient == 1 (rel 1e-9): {bhatt_coefficient}")

    assert bhatt_coefficient == pytest.approx(1.0, rel=1e-9), f"Bhattacharyya Coefficient is not 1.0 ({bhatt_coefficient})"