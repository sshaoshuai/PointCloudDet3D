# Modal setup

This is a guide to setting up Modal for the first time.

Most complexity is in composing the `pcdet_cuda_image` in `shared.py`.

## Install modal

* pip install modal
* modal setup

See https://modal.com/docs/guide for more information.

## Prepare data

* Download the nuScenes mini dataset
* Create shared Modal volume: `modal volume create nuscenes`.
* (Optionally: delete all image and radar folders to speed up the next step.)
* Upload data to the volume: `modal volume put nuscenes v1.0-mini`. (Navigate to the folder containing the nuScenes data and run this command)

## Format the data

This takes about an hour. I'm not sure what they are doing, but somehow normalizing the data for opendet

* `modal run build_nuscenes.py`

## Train the model

This is currently exiting with floating point error. I'm not sure why, but I don't think it's related to Modal.
See https://github.com/open-mmlab/OpenPCDet/issues/1642

* `modal run train_pp.py`
