import logging
import re

import numpy as np
import pytest
from scipy import ndimage

from BabelBrain.GPUFunctions.GPUMedianFilter import MedianFilter
from Tests.helpers.logs import describe_array, log_duration

@pytest.mark.step1
def test_MedianFilter_vs_CPU(computing_backend,dataset,spatial_step,check_os,get_gpu_device,load_files,get_resampled_input,cached_cpu_truth,gpu_vs_cpu_plot,compare_data,request):
        
     # Parameters
     input_folder = dataset['folder_path']
     input_files = {'CT': input_folder + 'CT.nii.gz',}
     spatial_step_text = re.sub("\\.","_",str(spatial_step))
     output_fnames = {
          'Resampled_Input': input_folder + f"CT_cpu_resampled_spatial_step_{spatial_step_text}.nii.gz",
          'Output_Truth': input_folder + f"CT_cpu_median_filter_spatial_step_{spatial_step_text}.nii.gz"
     }

     logging.info("Testing GPU median filter of the CT against scipy")

     # Initialize GPU Backend
     gpu_device = get_gpu_device
     logging.info(f"Initializing MedianFilter on {computing_backend['type']} (device {gpu_device})")
     MedianFilter.InitMedianFilter(gpu_device,computing_backend['type'])

     # Load inputs
     logging.info("Loading CT and resampling it to the spatial step")
     input_data = load_files(input_files)

     # Get resampled input for specified spatial step
     resampled_nifti, resampled_data = get_resampled_input(input_data['CT'],spatial_step,output_fnames['Resampled_Input'])

     # Determine median filter size (2 mm kernel in voxels)
     filter_size = np.round((np.ones(3)*2)/resampled_nifti.header.get_zooms()).astype(int)
     logging.info(f"Median Filter Size: {filter_size}")
     
     if np.any(filter_size[filter_size > 7]):
          logging.warning(f"Median filter was capped at 7")
          filter_size = 7 

     # Run median filter step
     CT_data = np.ascontiguousarray(input_data['CT'].get_fdata().astype(np.uint8))
     describe_array('Input CT', CT_data)
     with log_duration(f"GPU median filter ({computing_backend['type']})"):
         data_median_filter_gpu = MedianFilter.MedianFilter(CT_data,filter_size,GPUBackend=computing_backend['type'])
     describe_array('GPU output', data_median_filter_gpu)
     logging.info("Getting CPU truth (scipy ndimage.median_filter)")
     data_median_filter_cpu = cached_cpu_truth(output_fnames['Output_Truth'],
                                               lambda: ndimage.median_filter(CT_data,filter_size),
                                               affine=resampled_nifti.affine,dtype=None)

     # Save plot screenshot to be added to html report later
     logging.info("Saving GPU vs CPU comparison plot for the report")
     request.node.screenshots = []
     plots = [resampled_data,data_median_filter_cpu,data_median_filter_gpu]
     plot_names = ['Resampled Data','CPU Median Filter\nOutput (Truth)', 'GPU Median\n Filter Output']
     screenshot = gpu_vs_cpu_plot(plots,plot_names,data_median_filter_cpu,data_median_filter_gpu)
     request.node.screenshots.append(screenshot)
     
     # Calculate dice coefficient
     calc_dice_func = compare_data['dice_coefficient']
     dice_coeff = calc_dice_func(data_median_filter_gpu,data_median_filter_cpu)
     logging.info(f"Checking GPU vs CPU dice coefficient > 0.99: {dice_coeff}")

     assert dice_coeff > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff})"