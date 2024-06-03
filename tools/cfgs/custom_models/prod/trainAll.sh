 
#!/bin/bash

# Define the directory where the commands will be executed
TARGET_DIRECTORY="/app/OpenPCDet/tools"

# Change to the target directory
cd "$TARGET_DIRECTORY"
CONFIG_DIR = "/app/OpenPCDet/tools/cfgs/custom_models/prod/"
# Array of configuration files
CONFIG_FILES=(
    "pointpillar_10ep.yaml"
    "pointpillar_25ep.yaml"
    "pointpillar_50ep.yaml"
    "pointpillar_100ep.yaml"
    "pv_rcnn_10ep.yaml"
    "pv_rcnn_25ep.yaml"
    "pv_rcnn_50ep.yaml"
    "pv_rcnn_100ep.yaml"
    "centerpoint_10ep.yaml"
    "centerpoint_25ep.yaml"
    "centerpoint_50ep.yaml"
    "centerpoint_100ep.yaml"
    "second_10ep.yaml"
    "second_25ep.yaml"
    "second_50ep.yaml"
    "second_100ep.yaml"
)

# Loop through each configuration file and execute the command
for CFG in "${CONFIG_FILES[@]}"
do
    echo "Starting training with configuration: $CFG"
    # Execute the python training command with the current configuration file and capture the output
    python3 train.py --cfg_file $CONFIG_DIR$CFG &> output.log

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for configuration: $CFG"
    else
        echo "Training failed for configuration: $CFG. Check logs for details."
        # Optional: Exit on failure
        exit 1
    fi

    # Capture the last 100 lines from the output and save it to a file
    tail -n 100 output.log > "${CONFIG_DIR}${CFG%.*}_output.txt"

    # Optional: Remove the full log if it's no longer needed
    # rm output.log

    # Optional: Sleep for a certain amount of time before starting the next command
    # sleep 10
done

echo "All training commands have been executed."
