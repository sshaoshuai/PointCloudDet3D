import os
from shared import pcdet_cuda_image, volumes
import modal

app = modal.App("pcdet")

@app.function(image=pcdet_cuda_image, gpu="T4", timeout=3600*2, volumes=volumes)
def train_pointpillars():
    print("-> Training PP")

    import subprocess
    subprocess.run(["pip", "freeze"])
    # os.chdir("/root/tools")
    # subprocess.run(["python", "train.py", "--cfg_file", "cfgs/nuscenes_models/cbgs_pp_multihead.yaml"])
    from tools.check_dataloader import main
    main()
    
    print("-> Done training PP")


@app.local_entrypoint()
def main():
    train_pointpillars.remote()