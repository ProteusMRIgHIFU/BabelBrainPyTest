import hashlib
import logging
from unittest import mock

import pytest

@pytest.mark.step1
class TestFileManager:
    def test_wait_for_file(self,set_up_file_manager,get_example_data,tmpdir,caplog):

        # Set up file manager
        logging.info("Setting up a blank FileManager")
        fm,_,_ = set_up_file_manager['blank']()

        # Save random numpy array in temp file
        random_array = get_example_data['numpy']()
        temp_file_path = tmpdir.join("random_array.npy")

        # Start saving file
        # No logging between save_file and wait_for_file: the save must still be running when waiting starts
        logging.info(f"Saving a {random_array.shape} array to {temp_file_path} in the background, then waiting for it")
        fm.save_file(random_array,temp_file_path)

        # Call wait_for_file method
        with caplog.at_level(logging.INFO):
            fm.wait_for_file(temp_file_path)

        logging.info("Checking wait_for_file logged that the file still needed saving")
        assert f"{temp_file_path} needs to save first" in caplog.text

    def test_generate_hash(self,set_up_file_manager, tmpdir):

            # Set up file manager
            logging.info("Setting up a blank FileManager")
            fm,_,_ = set_up_file_manager['blank']()

            # Create a temporary file for testing
            temp_file = tmpdir.join("test_file.txt")
            msg = "Sample content for hashing"
            logging.info(f"Writing '{msg}' to {temp_file}")
            with open(temp_file,'w') as file:
                file.write(msg)

            # Calculate the hash using generate_hash method
            logging.info("Hashing the file with generate_hash")
            result = fm.generate_hash(str(temp_file))

            # Calculate the hash manually for comparison (FileManager uses a 4-byte blake2s digest)
            blake2s_hash = hashlib.blake2s(digest_size=4)
            blake2s_hash.update(msg.encode('utf-8'))
            expected_hash = blake2s_hash.hexdigest()

            logging.info(f"Checking hash {result} matches the manual blake2s hash {expected_hash}")
            assert result == expected_hash, "Generated hash does not match expected hash"

    def test_generate_hash_file_not_found(self,set_up_file_manager):

        # Set up file manager
        logging.info("Setting up a blank FileManager")
        fm,_,_ = set_up_file_manager['blank']()

        # Test if the function raises FileNotFoundError when file does not exist
        logging.info("Checking generate_hash raises FileNotFoundError for a missing file")
        with pytest.raises(FileNotFoundError, match="No such file: 'non_existent_file.txt'"):
            fm.generate_hash("non_existent_file.txt")

    def test_generate_hash_read_error(self,set_up_file_manager, tmpdir):

            # Set up file manager
            logging.info("Setting up a blank FileManager")
            fm,_,_ = set_up_file_manager['blank']()

            # Create a temporary file for testing
            temp_file = tmpdir.join("test_file.txt")
            msg = "Sample content for hashing"
            logging.info(f"Writing '{msg}' to {temp_file}")
            with open(temp_file,'w') as file:
                file.write(msg)

            # Simulate an OSError when trying to read the file
            logging.info("Checking a simulated OSError on read is raised as RuntimeError")
            with mock.patch("builtins.open", side_effect=OSError("Simulated read error")):
                with pytest.raises(RuntimeError, match="An error occurred while reading the file"):
                    fm.generate_hash(str(temp_file))

    def test_nibabel_to_sitk(self,dataset,set_up_file_manager,extract_sitk_info,compare_data):

        # Set up file manager
        logging.info(f"Setting up a FileManager with the {dataset['id']} T1")
        fm = set_up_file_manager['existing_dataset'](dataset)

        # Run nibabel_to_sitk function
        logging.info("Converting the nibabel T1 to SimpleITK")
        output_data = fm.nibabel_to_sitk(fm.saved_objects['T1_nib'])

        # Extract info
        output_spacing,output_direction,output_origin,output_array = extract_sitk_info(output_data)
        truth_spacing,truth_direction,truth_origin,truth_array = extract_sitk_info(fm.saved_objects['T1_sitk'])

        # Compare info
        logging.info("Comparing spacing, direction, origin and voxels with the T1 loaded by SimpleITK")
        dice_coeff_spacing = compare_data['dice_coefficient'](output_spacing, truth_spacing)
        dice_coeff_direction = compare_data['dice_coefficient'](output_direction, truth_direction)
        dice_coeff_origin = compare_data['dice_coefficient'](output_origin, truth_origin)
        dice_coeff_array = compare_data['dice_coefficient'](output_array, truth_array)

        dice_coeff_overall = (dice_coeff_spacing + dice_coeff_direction + dice_coeff_origin + dice_coeff_array)/4

        logging.info(f"Checking mean dice coefficient {dice_coeff_overall:.4f} > 0.99")
        assert dice_coeff_overall > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff_overall})"

    def test_sitk_to_nibabel(self,dataset,set_up_file_manager,extract_nib_info,compare_data):

        # Set up file manager
        logging.info(f"Setting up a FileManager with the {dataset['id']} T1")
        fm = set_up_file_manager['existing_dataset'](dataset)

        # Run sitk_to_nibabel function
        logging.info("Converting the SimpleITK T1 to nibabel")
        output_data = fm.sitk_to_nibabel(fm.saved_objects['T1_sitk'])

        # Extract info
        output_zooms,output_affine,output_array = extract_nib_info(output_data)
        truth_zooms,truth_affine,truth_array = extract_nib_info(fm.saved_objects['T1_nib'])

        # Compare info
        logging.info("Comparing zooms, affine and voxels with the T1 loaded by nibabel")
        dice_coeff_spacing = compare_data['dice_coefficient'](output_zooms, truth_zooms)
        dice_coeff_direction = compare_data['dice_coefficient'](output_affine, truth_affine)
        dice_coeff_array = compare_data['dice_coefficient'](output_array, truth_array)

        dice_coeff_overall = (dice_coeff_spacing + dice_coeff_direction + dice_coeff_array)/3

        logging.info(f"Checking mean dice coefficient {dice_coeff_overall:.4f} > 0.99")
        assert dice_coeff_overall > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff_overall})"

    def test_nibabel_to_sitk_and_back(self,dataset,set_up_file_manager,extract_nib_info,compare_data):

        # Set up file manager
        logging.info(f"Setting up a FileManager with the {dataset['id']} T1")
        fm = set_up_file_manager['existing_dataset'](dataset)

        # Run nibabel_to_sitk function
        logging.info("Converting the nibabel T1 to SimpleITK and back to nibabel")
        output_data = fm.nibabel_to_sitk(fm.saved_objects['T1_nib'])

        # Run sitk_to_nibabel function
        output_data = fm.sitk_to_nibabel(output_data)

        # Extract info
        output_zooms,output_affine,output_array = extract_nib_info(output_data)
        truth_zooms,truth_affine,truth_array = extract_nib_info(fm.saved_objects['T1_nib'])

        # Compare info
        logging.info("Comparing zooms, affine and voxels with the original nibabel T1")
        dice_coeff_spacing = compare_data['dice_coefficient'](output_zooms, truth_zooms)
        dice_coeff_direction = compare_data['dice_coefficient'](output_affine, truth_affine)
        dice_coeff_array = compare_data['dice_coefficient'](output_array, truth_array)

        dice_coeff_overall = (dice_coeff_spacing + dice_coeff_direction + dice_coeff_array)/3

        logging.info(f"Checking mean dice coefficient {dice_coeff_overall:.4f} > 0.99")
        assert dice_coeff_overall > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff_overall})"

    def test_sitk_to_nibabel_and_back(self,dataset,set_up_file_manager,extract_sitk_info,compare_data):

        # Set up file manager
        logging.info(f"Setting up a FileManager with the {dataset['id']} T1")
        fm = set_up_file_manager['existing_dataset'](dataset)

        # Run sitk_to_nibabel function
        logging.info("Converting the SimpleITK T1 to nibabel and back to SimpleITK")
        output_data = fm.sitk_to_nibabel(fm.saved_objects['T1_sitk'])

        # Run nibabel_to_sitk function
        output_data = fm.nibabel_to_sitk(output_data)

        # Extract info
        output_spacing,output_direction,output_origin,output_array = extract_sitk_info(output_data)
        truth_spacing,truth_direction,truth_origin,truth_array = extract_sitk_info(fm.saved_objects['T1_sitk'])

        # Compare info
        logging.info("Comparing spacing, direction, origin and voxels with the original SimpleITK T1")
        dice_coeff_spacing = compare_data['dice_coefficient'](output_spacing, truth_spacing)
        dice_coeff_direction = compare_data['dice_coefficient'](output_direction, truth_direction)
        dice_coeff_origin = compare_data['dice_coefficient'](output_origin, truth_origin)
        dice_coeff_array = compare_data['dice_coefficient'](output_array, truth_array)

        dice_coeff_overall = (dice_coeff_spacing + dice_coeff_direction + dice_coeff_origin + dice_coeff_array)/4

        logging.info(f"Checking mean dice coefficient {dice_coeff_overall:.4f} > 0.99")
        assert dice_coeff_overall > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff_overall})"

    def test_check_reuse_files_no_files(self,scan_type,set_up_file_manager,capfd):

        # Set up file manager for 'previous run'
        logging.info(f"Setting up run 1 ({scan_type} scan), saving no outputs")
        fm1,input_fnames_run_1,output_fnames_run_1 = set_up_file_manager['blank'](scan_type)
        logging.info(f"Run 1 Input files:\n{list(input_fnames_run_1.values())}")
        logging.info(f"\nRun 1 Output files:\n{list(output_fnames_run_1.values())}")

        # Set up file manager for 'second run'
        logging.info(f"Setting up run 2 with the same {scan_type} settings")
        fm2,input_fnames_run_2,output_fnames_run_2 = set_up_file_manager['blank'](scan_type)
        logging.info(f"Run 2 Input files:\n{list(input_fnames_run_2.values())} (Should be identically to run 1)")
        logging.info(f"\nRun 2 Output files:\n{list(output_fnames_run_2.values())}")

        # Check if files can be reused
        logging.info("Running check_reuse_files for run 2")
        reuse,_ = fm2.check_reuse_files(input_fnames_run_2,output_fnames_run_2)
        print_statements, _ = capfd.readouterr()

        logging.info(f"Checking files aren't reused (reuse = {reuse}) because they don't exist")
        assert reuse == False, "Files were reused even though they don't exist"
        assert "Previous files don't exist" in print_statements, "Reason for not reusing files was incorrect"

    def test_check_reuse_files_valid(self,scan_type,set_up_file_manager,get_example_data):

        # Set up file manager for 'previous run'
        logging.info(f"Setting up run 1 ({scan_type} scan)")
        fm1,input_fnames_run_1,output_fnames_run_1 = set_up_file_manager['blank'](scan_type)
        logging.info(f"Run 1 Input files:\n{list(input_fnames_run_1.values())}")
        logging.info(f"\nRun 1 Output files:\n{list(output_fnames_run_1.values())}")

        # Get random example data
        example_data_1 = get_example_data['nifti_nib']()
        example_data_2 = get_example_data['nifti_nib']()

        # Save 'previous run' files (output2 depends on output1, which depends on input1)
        logging.info("Saving run 1 outputs with their precursor file hashes")
        fm1.save_file(example_data_1,output_fnames_run_1['output1'],precursor_files=input_fnames_run_1['input1'])
        fm1.save_file(example_data_2,output_fnames_run_1['output2'],precursor_files=output_fnames_run_1['output1'])

        # Wait for files to finish saving
        fm1.wait_for_file(output_fnames_run_1['output1'])
        fm1.wait_for_file(output_fnames_run_1['output2'])

        # Set up file manager for 'second run'
        logging.info(f"Setting up run 2 with the same {scan_type} settings")
        fm2,input_fnames_run_2,output_fnames_run_2 = set_up_file_manager['blank'](scan_type)
        logging.info(f"Run 2 Input files:\n{list(input_fnames_run_2.values())} (Should be identically to run 1)")
        logging.info(f"\nRun 2 Output files:\n{list(output_fnames_run_2.values())}")

        # Check if files can be reused
        logging.info("Running check_reuse_files for run 2")
        reuse,_ = fm2.check_reuse_files(input_fnames_run_2,output_fnames_run_2)

        logging.info(f"Checking run 1 files are reused (reuse = {reuse})")
        assert reuse == True, f"Files aren't reused even though they can be (reuse = {reuse})"

    def test_check_reuse_files_different_CT_type(self,scan_type,second_scan_type,set_up_file_manager,get_example_data,capfd):

        if scan_type == second_scan_type:
            pytest.skip("Not a different scan type")

        # Set up file manager for 'previous run'
        logging.info(f"Setting up run 1 ({scan_type} scan)")
        fm1,input_fnames_run_1,output_fnames_run_1 = set_up_file_manager['blank'](scan_type)
        logging.info(f"Run 1 Input files:\n{list(input_fnames_run_1.values())}")
        logging.info(f"\nRun 1 Output files:\n{list(output_fnames_run_1.values())}")

        # Get random example data
        example_data_1 = get_example_data['nifti_nib']()
        example_data_2 = get_example_data['nifti_nib']()

        # Save 'previous run' files
        logging.info("Saving run 1 outputs with their precursor file hashes")
        fm1.save_file(example_data_1,output_fnames_run_1['output1'],precursor_files=input_fnames_run_1['input1'])
        fm1.save_file(example_data_2,output_fnames_run_1['output2'],precursor_files=output_fnames_run_1['output1'])

        # Wait for files to finish saving
        fm1.wait_for_file(output_fnames_run_1['output1'])
        fm1.wait_for_file(output_fnames_run_1['output2'])

        # Set up file manager for 'second run'
        logging.info(f"Setting up run 2 with a different scan type ({second_scan_type})")
        fm2,input_fnames_run_2,output_fnames_run_2 = set_up_file_manager['blank'](second_scan_type)
        logging.info(f"Run 2 Input files:\n{list(input_fnames_run_2.values())} (Should be identically to run 1)")
        logging.info(f"\nRun 2 Output files:\n{list(output_fnames_run_2.values())}")

        # Check if files can be reused
        logging.info("Running check_reuse_files for run 2")
        reuse,_ = fm2.check_reuse_files(input_fnames_run_2,output_fnames_run_2)
        print_statements,_ = capfd.readouterr()

        logging.info(f"Checking files aren't reused (reuse = {reuse}) because the CT type changed")
        assert reuse == False, f"Files are reused even though they shouldn't be (reuse = {reuse})"
        assert any(msg in print_statements for msg in ["Previous files used different CT Type", "Previous files didn't use CT"]), "Reason for not reusing files was incorrect"

    def test_check_reuse_files_different_HU_threshold(self,scan_type,set_up_file_manager,get_example_data,capfd):

        if scan_type == 'NONE':
            pytest.skip("No CT therefore no HUT value")

        # Set up file manager for 'previous run' (default HU threshold, 300)
        logging.info(f"Setting up run 1 ({scan_type} scan, default HU threshold)")
        fm1,input_fnames_run_1,output_fnames_run_1 = set_up_file_manager['blank'](scan_type)
        logging.info(f"Run 1 Input files:\n{list(input_fnames_run_1.values())}")
        logging.info(f"\nRun 1 Output files:\n{list(output_fnames_run_1.values())}")

        # Get random example data
        example_data_1 = get_example_data['nifti_nib']()
        example_data_2 = get_example_data['nifti_nib']()

        # Save 'previous run' files
        logging.info("Saving run 1 outputs with their precursor file hashes")
        fm1.save_file(example_data_1,output_fnames_run_1['output1'],precursor_files=input_fnames_run_1['input1'])
        fm1.save_file(example_data_2,output_fnames_run_1['output2'],precursor_files=output_fnames_run_1['output1'])

        # Wait for files to finish saving
        fm1.wait_for_file(output_fnames_run_1['output1'])
        fm1.wait_for_file(output_fnames_run_1['output2'])

        # Set up file manager for 'second run'
        logging.info("Setting up run 2 with HU threshold 400")
        fm2,input_fnames_run_2,output_fnames_run_2 = set_up_file_manager['blank'](scan_type,HUT=400.0)
        logging.info(f"Run 2 Input files:\n{list(input_fnames_run_2.values())} (Should be identically to run 1)")
        logging.info(f"\nRun 2 Output files:\n{list(output_fnames_run_2.values())}")

        # Check if files can be reused
        logging.info("Running check_reuse_files for run 2")
        reuse,_ = fm2.check_reuse_files(input_fnames_run_2,output_fnames_run_2)
        print_statements,_ = capfd.readouterr()

        logging.info(f"Checking files aren't reused (reuse = {reuse}) because the HU threshold changed")
        assert reuse == False, f"Files are reused even though they shouldn't be (reuse = {reuse})"
        assert any(msg in print_statements for msg in ["Previous files used different HUThreshold","Previous files didn't have HU Threshold value"]), "Reason for not reusing files was incorrect"

    def test_check_reuse_files_different_pCT_range(self,scan_type,set_up_file_manager,get_example_data,capfd):
        if scan_type == 'NONE' or scan_type == 'CT':
            pytest.skip("No CT or real CT therefore no pseudo CT range value")

        # Set up file manager for 'previous run' (default pseudo CT range, (0.1, 0.6))
        logging.info(f"Setting up run 1 ({scan_type} scan, default pseudo CT range)")
        fm1,input_fnames_run_1,output_fnames_run_1 = set_up_file_manager['blank'](scan_type)
        logging.info(f"Run 1 Input files:\n{list(input_fnames_run_1.values())}")
        logging.info(f"\nRun 1 Output files:\n{list(output_fnames_run_1.values())}")

        # Get random example data
        example_data_1 = get_example_data['nifti_nib']()
        example_data_2 = get_example_data['nifti_nib']()

        # Save 'previous run' files
        logging.info("Saving run 1 outputs with their precursor file hashes")
        fm1.save_file(example_data_1,output_fnames_run_1['output1'],precursor_files=input_fnames_run_1['input1'])
        fm1.save_file(example_data_2,output_fnames_run_1['output2'],precursor_files=output_fnames_run_1['output1'])

        # Wait for files to finish saving
        fm1.wait_for_file(output_fnames_run_1['output1'])
        fm1.wait_for_file(output_fnames_run_1['output2'])

        # Set up file manager for 'second run'
        logging.info("Setting up run 2 with pseudo CT range (0.1, 0.5)")
        fm2,input_fnames_run_2,output_fnames_run_2 = set_up_file_manager['blank'](scan_type,pCT_range=(0.1,0.5))
        logging.info(f"Run 2 Input files:\n{list(input_fnames_run_2.values())} (Should be identically to run 1)")
        logging.info(f"\nRun 2 Output files:\n{list(output_fnames_run_2.values())}")

        # Check if files can be reused
        logging.info("Running check_reuse_files for run 2")
        reuse,_ = fm2.check_reuse_files(input_fnames_run_2,output_fnames_run_2)
        print_statements,_ = capfd.readouterr()

        logging.info(f"Checking files aren't reused (reuse = {reuse}) because the pseudo CT range changed")
        assert reuse == False, f"Files are reused even though they shouldn't be (reuse = {reuse})"
        assert any(msg in print_statements for msg in ["Previous files used different ZTE/PETRA Range","Previous files didn't have ZTE/PETRA Range values"]), "Reason for not reusing files was incorrect"
