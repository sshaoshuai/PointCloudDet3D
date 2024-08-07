import modal

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
op_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{op_sys}"

pcdet_cuda_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .apt_install(["clang", "python3-clang", "ffmpeg", "libsm6", "libxext6"])
    .pip_install(["spconv", "nuscenes-devkit"])
    .pip_install_from_requirements("../requirements.txt")
    .run_commands("git clone https://github.com/beijbom/OpenPCDet.git /root/OpenPCDet" )
    .run_commands("cd /root/OpenPCDet && git reset --hard 1918f7f66defea995c0b79a26a0f51289677b466")
    .run_commands("cp -r /root/OpenPCDet/pcdet /root/pcdet")
    .run_commands("cp /root/OpenPCDet/setup.py /root/setup.py")
    .run_commands("export TORCH_CUDA_ARCH_LIST='6.0 6.1 7.0 7.5 8.0' && cd /root/ && python setup.py develop")
    .copy_local_dir("../tools", "/root/tools")
)

volume = modal.Volume.from_name("nuscenes")
VOL_MOUNT_PATH = "/root/data/nuscenes"

volumes={VOL_MOUNT_PATH: volume}