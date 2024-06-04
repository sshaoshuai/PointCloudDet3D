 
#!/bin/bash

# Define the directory where the commands will be executed
TARGET_DIRECTORY="/app/OpenPCDet/tools"
CONFIG_DIR="/app/OpenPCDet/tools/cfgs/custom_models/prod/"

# Change to the target directory
cd "$TARGET_DIRECTORY"

# Array of configuration files
CONFIG_FILES=(
    "pointpillar_20ppv.yaml"
    "pointpillar_32ppv.yaml"
    "pointpillar_40ppv.yaml"
    "pointpillar_64ppv.yaml"

    "pv_rcnn_3grid.yaml"
    "pv_rcnn_6grid.yaml"
    "pv_rcnn_8grid.yaml"
    "pv_rcnn_12grid.yaml"

    "centerpoint_voxel2-1875.yaml"
    "centerpoint_voxel5-1875.yaml"
    "centerpoint_voxel5-375.yaml"

    "second_2lay.yaml"
    "second_4lay.yaml"
    "second_5layers.yaml"
    "second_6lay.yaml"
)

# Loop through each configuration file and execute the command
for CFG in "${CONFIG_FILES[@]}"
do
    echo "Starting training with configuration: $CFG"
    # Execute the python training command with the current configuration file and capture the output
    python3 train.py --cfg_file "${CONFIG_DIR}${CFG}" &> output.log

    BACK_PID=$!
    wait $BACK_PID

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for configuration: $CFG"
    else
        echo "Training failed for configuration: $CFG. Check logs for details."
        # Optional: Exit on failure
        #exit 1
    fi

    # Capture the last 100 lines from the output and save it to a file
    tail -n 100 output.log > "${CONFIG_DIR}/out/${CFG%.*}_output.txt"

    # Optional: Remove the full log if it's no longer needed
    # rm output.log

    # Optional: Sleep for a certain amount of time before starting the next command
    # sleep 10
done

echo "All training commands have been executed."
