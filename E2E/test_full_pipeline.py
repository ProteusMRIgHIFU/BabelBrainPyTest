import logging
import os

def test_full_pipeline_normal(babelbrain_widget,pipeline,check_os,transducer):
    logging.info(f"Running the full pipeline (steps 1 to 3) with transducer {transducer['name']}")
    pipeline.run_widget(babelbrain_widget())

def test_full_pipeline_regression_normal(babelbrain_widget,pipeline,check_os,compare_BabelBrain_Outputs,request,tmp_path,get_config_dirs):
    logging.info(f"Running the full pipeline (steps 1 to 3) into {tmp_path}")
    pipeline.run_widget(babelbrain_widget())

    # Compare output against reference outputs
    logging.info(f"Comparing the outputs in {tmp_path} with the reference outputs in {get_config_dirs['ref_dir_1']}")
    bb_outputs_match = compare_BabelBrain_Outputs(ref_folder = get_config_dirs['ref_dir_1'], test_folder = tmp_path, node_screenshots=request.node.screenshots)

    logging.info("Checking the h5 outputs match the reference")
    assert bb_outputs_match == True, "There were differences between the h5 files, see logged warnings for more details"

def test_full_pipeline_two_outputs(compare_BabelBrain_Outputs,get_config_dirs,request,output_subfolder):
    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    config_dirs = get_config_dirs
    ref_output_dir = config_dirs['ref_dir_1']
    ref_output_dir_2 = config_dirs['ref_dir_2']

    # output_subfolder is None when ref_dir_1 has the h5 files directly in it
    # (single-folder layout); otherwise each subfolder is its own parametrized
    # test, so it shows up as its own row in the html report.
    if output_subfolder is not None:
        ref_output_dir = os.path.join(ref_output_dir, output_subfolder)
        ref_output_dir_2 = os.path.join(ref_output_dir_2, output_subfolder)

    logging.info(f"Comparing the outputs in {ref_output_dir} with those in {ref_output_dir_2}")
    bb_outputs_match = compare_BabelBrain_Outputs(ref_folder = ref_output_dir, test_folder = ref_output_dir_2,node_screenshots=request.node.screenshots)

    logging.info("Checking the h5 outputs of both folders match")
    assert bb_outputs_match == True, "There were differences between the h5 files, see logged warnings for more details"
