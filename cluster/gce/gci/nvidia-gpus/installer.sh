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

should_install() {
    if verify_nvidia; then
	echo "nvidia drivers already installed. Skipping installation"
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
    cp -r /rootfs/nvidia-overlay/lib/modules/* /rootfs/lib/modules/
}

main() {
    verify_base_image
    check_nvidia_device
    setup_overlay_mounts
    should_install
    unlock_loadpin_and_reboot_if_needed
    prepare_kernel_source
    download_install_nvidia
    verify_nvidia
    copy_files_to_host
    restart_kubelet
}

main "$@"
