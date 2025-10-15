#!/bin/zsh
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
cd BabelBrain
git checkout main
conda activate BabelBrain #activating 1st reference 
sed -i '' "/gen_output_folder/c\\
gen_output_folder = /Volumes/Samsung_T5/BabelBrainRegression/Generated_Outputs/v0.4.3
" Tests/config.ini
pytest -k "test_generate_valid_outputs and Metal" -m "basic_babelbrain_params"
# #########################
git checkout rc0.8.0
conda activate BabelMLX
# # setting output directory
sed -i '' "/gen_output_folder/c\\
gen_output_folder = /Volumes/Samsung_T5/BabelBrainRegression/Generated_Outputs/v0.8.0
" Tests/config.ini
pytest -k "test_generate_valid_outputs and MLX" -m "basic_babelbrain_params"
# ###########################
#setting comparison directories
conda deactivate
conda activate BabelBrain 
git checkout main
sed -i '' "/ref_output_folder_1/c\\
ref_output_folder_1 = /Volumes/Samsung_T5/BabelBrainRegression/Generated_Outputs/v0.4.3
" Tests/config.ini
sed -i '' "/ref_output_folder_2/c\\
ref_output_folder_2 = /Volumes/Samsung_T5/BabelBrainRegression/Generated_Outputs/v0.8.0
" Tests/config.ini
# run comparison test
pytest -k "test_full_pipeline_two_outputs"
conda deactivate
cd ..


