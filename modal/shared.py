import modal

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
op_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{op_sys}"

pcdet_cuda_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")  # for cloning OpenPCDet
    .apt_install(["clang", "python3-clang", "ffmpeg", "libsm6", "libxext6"])  # Reqs for building OpenPCDet
    .pip_install(["spconv", "nuscenes-devkit"])  # Some reqs not in requirements.txt
    .pip_install_from_requirements("../requirements.txt")  # Remining reqs for OpenPCDet
    .run_commands("git clone https://github.com/beijbom/OpenPCDet.git /root/OpenPCDet" )  # Clone the code. 
    .run_commands("cd /root/OpenPCDet && git reset --hard 1918f7f66defea995c0b79a26a0f51289677b466")  # Checkout a specific commit for reproducability.
    .run_commands("cp -r /root/OpenPCDet/pcdet /root/pcdet") # Copy the pcdet folder to the root
    .run_commands("cp /root/OpenPCDet/setup.py /root/setup.py") # Copy the setup.py file to the root
    .run_commands("export TORCH_CUDA_ARCH_LIST='6.0 6.1 7.0 7.5 8.0' && cd /root/ && python setup.py develop")  # Install OpenPCDet
    .copy_local_dir("../tools", "/root/tools") # Copy the tools folder to the container. This means you can change configs files and run them without having to commit the changes.
)

volume = modal.Volume.from_name("nuscenes")
VOL_MOUNT_PATH = "/root/data/nuscenes"

volumes={VOL_MOUNT_PATH: volume}