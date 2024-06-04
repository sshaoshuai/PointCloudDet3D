 
#!/bin/bash

# Define the directory where the commands will be executed
TARGET_DIRECTORY="/app/OpenPCDet/tools"
CONFIG_DIR="/app/OpenPCDet/tools/cfgs/custom_models/prod/"
OUTPUT_DIR="/app/OpenPCDet/output$CONFIG_DIR"

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
#    "pv_rcnn_8grid.yaml"
#    "pv_rcnn_12grid.yaml"

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

    # Get size of latest checkpoint file
    CHECKPOINT_FILE=$(ls -t "${OUTPUT_DIR}${CFG%.*}/default/ckpt/" | head -1)
    #echo "FILE $CHECKPOINT_FILE"
    CHECKPOINT_SIZE=$(stat -c %s "${OUTPUT_DIR}${CFG%.*}/default/ckpt/${CHECKPOINT_FILE}")
    #echo "SIZE $CHECKPOINT_SIZE"
    echo "\nSize of latest Checkpoint $CHECKPOINT_FILE is $CHECKPOINT_SIZE\n\n" >> output.log

    # Start Test and get Inference time
    echo "Starting test with configuration: $CFG"
    python3 test.py --cfg_file "${CONFIG_DIR}${CFG}" --ckpt "${CHECKPOINT_FILE}" &> test-output.log

    BACK_PID=$!
    wait $BACK_PID

    # get inference time "infer_time=5.6(114.0)"  from test-output.log
    (grep -oP 'infer_time=\K[0-9.]+\(.*\)' test-output.log | tail -n 2) >> output.log

    # Capture the last 100 lines from the output and save it to a file
    tail -n 100 output.log > "${CONFIG_DIR}out/${CFG%.*}_output.txt"

    # Optional: Remove the full log if it's no longer needed
    # rm output.log

    # Optional: Sleep for a certain amount of time before starting the next command
    # sleep 10
done

echo "All training commands have been executed."
