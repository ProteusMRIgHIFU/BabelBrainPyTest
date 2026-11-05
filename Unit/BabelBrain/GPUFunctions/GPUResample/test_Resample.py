import logging
import re

from nibabel import processing, nifti1
import numpy as np
import pytest

from BabelBrain.GPUFunctions.GPUResample import Resample
from Tests.helpers.logs import describe_array, log_duration

@pytest.mark.step1
def test_Resample_vs_CPU(computing_backend,dataset,spatial_step,check_os,get_gpu_device,load_files,resample_grid,cached_cpu_truth,gpu_vs_cpu_plot,compare_data,request):
        
    # Parameters
    input_folder = dataset['folder_path']
    order = 3
    file_type = 'T1W'
    input_files = {file_type: input_folder + 'T1W.nii.gz',}
    spatial_step_text = re.sub("\\.","_",str(spatial_step))
    output_fnames = {
        'Output_Truth': input_folder + f"{file_type}_cpu_resampled_order_{order}_spatial_step_{spatial_step_text}.nii.gz",
    }

    logging.info(f"Testing GPU resampling of {file_type} (order {order}) against nibabel")

    # Initialize GPU Backend
    gpu_device = get_gpu_device
    logging.info(f"Initializing Resample on {computing_backend['type']} (device {gpu_device})")
    Resample.InitResample(gpu_device,computing_backend['type'])

    # Load inputs
    logging.info(f"Loading {file_type} and building the output grid")
    input_data = load_files(input_files)

    # Output grid
    new_affine, new_shape = resample_grid(input_data[file_type],spatial_step)
    output_data = np.zeros(new_shape,dtype=np.uint8)
    output_nifti = nifti1.Nifti1Image(output_data,new_affine)
    logging.info(f"Output Dimensions: {output_data.shape}")
    logging.info(f"Output Size: {output_data.size}")

    # Run resample step
    with log_duration(f"GPU resampling ({computing_backend['type']})"):
        nifti_resampled_gpu = Resample.ResampleFromTo(input_data[file_type],output_nifti,mode='constant',order=order,cval=input_data[file_type].get_fdata().min(),GPUBackend=computing_backend['type'])
    data_resampled_gpu = nifti_resampled_gpu.get_fdata()
    describe_array('GPU output', data_resampled_gpu)
    logging.info("Getting CPU truth (nibabel resample_from_to)")
    data_resampled_cpu = cached_cpu_truth(
        output_fnames['Output_Truth'],
        lambda: processing.resample_from_to(input_data[file_type],output_nifti,mode='constant',order=order,cval=input_data[file_type].get_fdata().min()).get_fdata(),
        affine=new_affine,dtype=None)

    # Save plot screenshot to be added to html report later
    logging.info("Saving GPU vs CPU comparison plot for the report")
    request.node.screenshots = []
    plots = [input_data[file_type].get_fdata(),data_resampled_cpu,data_resampled_gpu]
    plot_names = ['Original Data', 'CPU Resample\nOutput (Truth)', 'GPU Resample\nOutput']
    screenshot = gpu_vs_cpu_plot(plots,plot_names,data_resampled_cpu,data_resampled_gpu)
    request.node.screenshots.append(screenshot)
    
    # Calculate Bhatt Coefficient
    calc_bc_func = compare_data['bhatt_coeff']
    bhatt_coeff = calc_bc_func(data_resampled_gpu,data_resampled_cpu)
    logging.info(f"Checking GPU vs CPU Bhattacharyya coefficient == 1 (rel 1e-9): {bhatt_coeff}")

    assert bhatt_coeff == pytest.approx(1.0, rel=1e-9), f"Bhattacharyya Coefficient is not 1.0 ({bhatt_coeff})"