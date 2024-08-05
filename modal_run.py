import modal
import os

app = modal.App("pcdet")

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
op_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{op_sys}"

pcdet_cuda_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .apt_install(["clang", "python3-clang"])
    .pip_install(["spconv"])
    .pip_install_from_requirements("requirements.txt")
    .copy_local_dir("pcdet", "/root/pcdet")
    .copy_local_file("setup.py", "/root/setup.py")
    .run_commands("export TORCH_CUDA_ARCH_LIST='6.0 6.1 7.0 7.5 8.0' && cd /root/ && python setup.py develop")
)

pcdet_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(["spconv"])
    .pip_install_from_requirements("requirements.txt")
    .copy_local_dir("pcdet", "/root/pcdet")
    .copy_local_file("setup.py", "/root/setup.py")
)

@app.function(image=pcdet_cuda_image, memory=8000, gpu="T4", timeout=600)
def build_pcdet(version='v1.0-mini'):
    print("Hello from PCDet!")
    
    # from pcdet.datasets.nuscenes import create_nuscenes_info
    # create_nuscenes_info(version=version)

@app.local_entrypoint()
def main():
    print(build_pcdet.remote(version='v1.0-mini'))