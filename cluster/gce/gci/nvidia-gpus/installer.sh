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

# The script must be run as a root.
# Prerequisites:
#
# LAKITU_KERNEL_SHA1 - The env variable is expected to be set to HEAD of the kernel version used on the host.
#
# The following directories are expected to exist
# /rootfs/usr - This is expected to map to a writable directory on the host, ideally `/usr` on the host.
# /rootfs/nvidia-overlay - This is expected to map to a stateful writable directory on the host.
#
# The script will output the following artifacts:
# /rootfs/usr/lib/nvidia/* --> Nvidia CUDA libraries
# /rootfs/usr/lib/nvidia/bin/* --> Nvidia debug utilities
# /rootfs/nvidia-overlay/usr-work --> User space libraries and debug utilities for caching purposes
# /rootfs/nvidia-overlay/lib-work --> Kernel modules for caching purposes
#
# Containers on the host are expected to consume nvidia libraries and debug utilities from `/usr/lib/nvidia` on the host.

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
    # git checkout "tags/v${kernel_version_stripped}"
    git checkout ${LAKITU_KERNEL_SHA1}

    # Prepare kernel configu and source for modules.
    echo "Preparing kernel sources ..."
    zcat "/proc/config.gz" > ".config"
    make olddefconfig
    make modules_prepare

    # Check if the magic version hack is needed.
    if ! grep -q "${kernel_version}" "include/generated/utsrelease.h"; then
        # Change the LOCALVERSION for magic version hack.
        echo "Applying magic version hack"
        cp ".config" ".config.orig"
        sed -i "s/CONFIG_LOCALVERSION=\"\"/CONFIG_LOCALVERSION=\"+\"/g" ".config"
        make modules_prepare
    fi
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
    if echo "${kernel_cmdline}" | grep -q "root=/dev/dm-0"; then
        local -r esp_partition="/dev/sda12"
        local -r mount_path="/tmp/esp"
        local -r grub_cfg="efi/boot/grub.cfg"

        mkdir -p "${mount_path}"
        mount "${esp_partition}" "${mount_path}"

        pushd "${mount_path}"
        cp "${grub_cfg}" "${grub_cfg}.orig"
        sed 's/defaultA=2/defaultA=0/g' -i "efi/boot/grub.cfg"
        sed 's/defaultB=3/defaultB=1/g' -i "efi/boot/grub.cfg"
        cat "${grub_cfg}"
        popd
        sync
        umount "${mount_path}"
        echo b > /sysrq
    fi
    sysctl -w kernel.chromiumos.module_locking=0
}

verify_nvidia() {
    # If the installation was successful, nvidia tools should be installed
    # under /usr/bin.
    nvidia-smi
    if [[ ! -e /dev/nvidia-uvm ]]; then
        # Create unified memory device file.
        nvidia-modprobe -c0 -u
    fi
}

verify_base_image() {
    mount --bind /rootfs/etc/os-release /etc/os-release
    os_release=$(cat /etc/os-release)
    if [[ ${os_release} != *"Container-Optimized OS"* ]]; then
        echo "This installer is designed to run on Container Optimized OS only"
        exit 1
    fi  
}

setup_overlay_mounts() {
    mkdir -p /rootfs/nvidia-overlay/usr /rootfs/nvidia-overlay/usr-work /rootfs/nvidia-overlay/lib /rootfs/nvidia-overlay/lib-work
    mount -t overlay -o lowerdir=/usr,upperdir=/rootfs/nvidia-overlay/usr,workdir=/rootfs/nvidia-overlay/usr-work none /usr
    mount -t overlay -o lowerdir=/lib,upperdir=/rootfs/nvidia-overlay/lib,workdir=/rootfs/nvidia-overlay/lib-work none /lib
}

exit_if_install_not_needed() {
    if verify_nvidia; then
	echo "nvidia drivers already installed. Skipping installation"
        # Kubelet will not pick up nvidia GPUs unless /dev/nvidia-uvm and /dev/nvidiactl files exist when it boots up.
        # Restart kubelet just in case it did not pick up in the last iteration.
        restart_kubelet
	exit 0
    fi
}

restart_kubelet() {
    echo "Sending SIGTERM to kubelet"
    pkill -SIGTERM kubelet
}

copy_files_to_host() {
    mkdir -p /rootfs/usr/lib/nvidia /rootfs/usr/lib/nvidia/bin
    cp -r /rootfs/nvidia-overlay/usr/lib/x86_64-linux-gnu/* /rootfs/usr/lib/nvidia/
    cp -r /rootfs/nvidia-overlay/usr/bin/* /rootfs/usr/lib/nvidia/bin/
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
    verify_nvidia
    # Copy nvidia user space libraries and debug tools to the host for use from other containers.
    copy_files_to_host
    # Restart the kubelet for it to pick up the GPU devices.
    restart_kubelet
}

main "$@"
