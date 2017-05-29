#!/bin/bash

# Copyright 2017 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is for dynamically installing nvidia kernel drivers in Container Optimized OS

set -o errexit
set -o pipefail
set -x

# The script must be run as a root.
# Prerequisites:
#
# LAKITU_KERNEL_SHA1 - The env variable is expected to be set to HEAD of the kernel version used on the host.
# BASE_DIR - Directory that is mapped to a stateful partition on host. Defaults to `/rootfs/nvidia`.
#
# The script will output the following artifacts:
# ${BASE_DIR}/lib* --> Nvidia CUDA libraries
# ${BASE_DIR}/bin/* --> Nvidia debug utilities
# ${BASE_DIR}/.cache/* --> Nvidia driver artifacts cached for idempotency.
#

BASE_DIR=${BASE_DIR:-"/rootfs/nvidia"}
CACHE_DIR="${BASE_DIR}/.cache"
USR_WORK_DIR="${CACHE_DIR}/usr-work"
USR_WRITABLE_DIR="${CACHE_DIR}/usr-writable"
LIB_WORK_DIR="${CACHE_DIR}/lib-work"
LIB_WRITABLE_DIR="${CACHE_DIR}/lib-writable"

LIB_OUTPUT_DIR="${BASE_DIR}/lib"
BIN_OUTPUT_DIR="${BASE_DIR}/bin"

KERNEL_SRC_DIR="/lakitu-kernel"
NVIDIA_DRIVER_DIR="/nvidia"
NVIDIA_DRIVER_VERSION="375.26"

# Source: https://developer.nvidia.com/cuda-downloads
NVIDIA_CUDA_URL="https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run"
NVIDIA_CUDA_MD5SUM="33e1bd980e91af4e55f3ef835c103f9b"
NVIDIA_CUDA_PKG_NAME="cuda_8.0.61_375.26_linux.run"
NVIDIA_DRIVER_PKG_NAME="NVIDIA-Linux-x86_64-375.26.run"

check_nvidia_device() {
    lspci
    if ! lspci | grep -i -q NVIDIA; then
        echo "No NVIDIA devices attached to this instance."
        exit 0
    fi
    echo "Found NVIDIA device on this instance."
}

prepare_kernel_source() {
    local kernel_git_repo="https://chromium.googlesource.com/chromiumos/third_party/kernel"
    local kernel_version="$(uname -r)"
    local kernel_version_stripped="$(echo ${kernel_version} | sed 's/\+//')"

    # Checkout the correct tag.
    echo "Downloading kernel source at tag ${kernel_version_stripped} ..."
    pushd "${KERNEL_SRC_DIR}"
    # TODO: Consume KERNEL SHA1 from COS image directly.
    # git checkout "tags/v${kernel_version_stripped}"
    git checkout ${LAKITU_KERNEL_SHA1}

    # Prepare kernel configu and source for modules.
    echo "Preparing kernel sources ..."
    zcat "/proc/config.gz" > ".config"
    make olddefconfig
    make modules_prepare
    # Done.
    popd
}

download_install_nvidia() {
    local pkg_name="${NVIDIA_CUDA_PKG_NAME}"
    local url="${NVIDIA_CUDA_URL}"
    local log_file_name="${NVIDIA_DRIVER_DIR}/nvidia-installer.log"

    mkdir -p "${NVIDIA_DRIVER_DIR}"
    pushd "${NVIDIA_DRIVER_DIR}"

    echo "Downloading Nvidia CUDA package from ${url} ..."
    curl -L -s "${url}" -o "${pkg_name}"
    echo "${NVIDIA_CUDA_MD5SUM} ${pkg_name}" | md5sum --check

    echo "Extracting Nvidia CUDA package ..."
    sh ${pkg_name} --extract="$(pwd)"

    echo "Running the Nvidia driver installer ..."
    if ! sh "${NVIDIA_DRIVER_PKG_NAME}" --kernel-source-path="${KERNEL_SRC_DIR}" --silent --accept-license --keep --log-file-name="${log_file_name}"; then
        echo "Nvidia installer failed, log below:"
        echo "==================================="
        tail -50 "${log_file_name}"
        echo "==================================="
        exit 1
    fi
    # Create unified memory device file.
    nvidia-modprobe -c0 -u
    popd
}

unlock_loadpin_and_reboot_if_needed() {
    kernel_cmdline="$(cat /proc/cmdline)"
    if echo "${kernel_cmdline}" | grep -q -v "lsm.module_locking=0"; then
        local -r esp_partition="/dev/sda12"
        local -r mount_path="/tmp/esp"
        local -r grub_cfg="efi/boot/grub.cfg"

        mkdir -p "${mount_path}"
        mount "${esp_partition}" "${mount_path}"

        pushd "${mount_path}"
        cp "${grub_cfg}" "${grub_cfg}.orig"
        sed 's/cros_efi/cros_efi lsm.module_locking=0/g' -i "efi/boot/grub.cfg"
        cat "${grub_cfg}"
        popd
        sync
        umount "${mount_path}"
        # Restart the node for loadpin to be disabled.
        echo b > /sysrq
    fi
}

create_uvm_device() {
    # Create unified memory device file.
    nvidia-modprobe -c0 -u
}

verify_base_image() {
    mount --bind /rootfs/etc/os-release /etc/os-release
    local id="$(grep "^ID=" /etc/os-release)"
    if [[ "${id#*=}" != "cos" ]]; then
        echo "This installer is designed to run on Container-Optimized OS only"
        exit 1
    fi
}

setup_overlay_mounts() {
    mkdir -p ${USR_WRITABLE_DIR} ${USR_WORK_DIR} ${LIB_WRITABLE_DIR} ${LIB_WORK_DIR} 
    mount -t overlay -o lowerdir=/usr,upperdir=${USR_WRITABLE_DIR},workdir=${USR_WORK_DIR} none /usr
    mount -t overlay -o lowerdir=/lib,upperdir=${LIB_WRITABLE_DIR},workdir=${LIB_WORK_DIR} none /lib
}

exit_if_install_not_needed() {
    if nvidia-smi; then
	echo "nvidia drivers already installed. Skipping installation"
        post_installation_sequence
	exit 0
    fi
}

restart_kubelet() {
    echo "Sending SIGTERM to kubelet"
    pkill -SIGTERM kubelet
}

# Copy user space libraries and debug utilities to a special output directory on the host.
# Make these artifacts world readable and executable.
copy_files_to_host() {
    mkdir -p ${LIB_OUTPUT_DIR} ${BIN_OUTPUT_DIR}
    cp -r ${USR_WRITABLE_DIR}/lib/x86_64-linux-gnu/* ${LIB_OUTPUT_DIR}/
    cp -r ${USR_WRITABLE_DIR}/bin/* ${BIN_OUTPUT_DIR}/
    chmod -R a+rx ${LIB_OUTPUT_DIR}
    chmod -R a+rx ${BIN_OUTPUT_DIR}
}

post_installation_sequence() {
    create_uvm_device
    # Copy nvidia user space libraries and debug tools to the host for use from other containers.
    copy_files_to_host
    # Restart the kubelet for it to pick up the GPU devices.
    restart_kubelet
}

main() {
    # Do not run the installer unless the base image is Container Optimized OS (COS)
    verify_base_image
    # Do not run the installer unless a Nvidia device is found on the PCI bus
    check_nvidia_device
    # Setup overlay mounts to capture nvidia driver artificats in a more permanent storage on the host.
    setup_overlay_mounts
    # Disable a critical security feature in COS that will allow for dynamically loading Nvidia drivers 
    unlock_loadpin_and_reboot_if_needed
    # Exit if installation is not required (for idempotency)
    exit_if_install_not_needed
    # Checkout kernel sources appropriate for the base image.
    prepare_kernel_source
    # Download, compile and install nvidia drivers.
    download_install_nvidia
    # Verify that the Nvidia drivers have been successfully installed.
    nvidia-smi
    # Perform post installation steps - copying artifacts, restarting kubelet, etc.
    post_installation_sequence
}

main "$@"
