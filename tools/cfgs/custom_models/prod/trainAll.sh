 
#!/bin/bash

# Define the directory where the commands will be executed
TARGET_DIRECTORY="/app/OpenPCDet/tools"
CONFIG_DIR="/app/OpenPCDet/tools/cfgs/custom_models/prod/"
OUTPUT_DIR="/app/OpenPCDet/output$CONFIG_DIR"

# Change to the target directory
cd "$TARGET_DIRECTORY"

# Array of configuration files
CONFIG_FILES=(
    "centerpoint_voxel2-1875_nms1d.yaml"
    
    "pointpillar_20ppv_lr001.yaml"
    "pointpillar_20ppv.yaml"
    "pointpillar_32ppv_lr001.yaml"

    "pv_rcnn_3grid_2048key.yaml"

    "second_2lay_nms1.yaml"
    "second_5layers_nms1.yaml"
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

    # Get size of latest checkpoint file
    CHECKPOINT_FILE=$(ls -t "${OUTPUT_DIR}${CFG%.*}/default/ckpt/" | head -1)
    #echo "FILE $CHECKPOINT_FILE"
    CHECKPOINT_SIZE=$(stat -c %s "${OUTPUT_DIR}${CFG%.*}/default/ckpt/${CHECKPOINT_FILE}")
    #echo "SIZE $CHECKPOINT_SIZE"
    echo "Size of latest Checkpoint $CHECKPOINT_FILE is $CHECKPOINT_SIZE" >> output.log

    # Start Test and get Inference time
    echo "Starting test with configuration: $CFG"
    python3 test.py --cfg_file "${CONFIG_DIR}${CFG}" --ckpt "${OUTPUT_DIR}${CFG%.*}/default/ckpt/${CHECKPOINT_FILE}" &> test-output.log

    BACK_PID=$!
    wait $BACK_PID

    # get inference time "infer_time=5.6(114.0)"  from test-output.log
    grep -oP 'infer_time=\K[0-9.]+\(.*\)' test-output.log >> output.log

    # Capture the last 100 lines from the output and save it to a file
    tail -n 100 output.log > "${CONFIG_DIR}out/${CFG%.*}_output.txt"

    # Optional: Remove the full log if it's no longer needed
    # rm output.log

    # Optional: Sleep for a certain amount of time before starting the next command
    # sleep 10
done

echo "All training commands have been executed."
