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

# Sets up FlexVolume drivers on GCE COS instances using mounting utilities packaged in a Google
# Container Registry image.
# The user-provided FlexVolume driver(s) must be under /flexvolume of the image filesystem.
# For example, the driver k8s/nfs must be located at /flexvolume/k8s~nfs/nfs .
#
# This script should be used on a clean instance, with no FlexVolume installed.
# Should not be run on instances with an existing full or partial installation.
# Upon failure, the script will clean up the partial installation automatically.
#
# Must be executed under /home/kubernetes/bin with sudo.
# Warning: kubelet will be restarted upon successful execution.

set -o errexit
set -o nounset
set -o pipefail

MOUNTER_IMAGE=${1:-}
MOUNTER_PATH=/home/kubernetes/flexvolume_mounter
VOLUME_PLUGIN_DIR=/etc/srv/kubernetes/kubelet-plugins/volume/exec

usage() {
  echo "usage: $0 imagename[:tag]"
  echo "    imagename  Name of a Container Registry image. By default the latest image is used."
  echo "    :tag       Container Registry image tag."
  exit 1
}

if [ -z ${MOUNTER_IMAGE} ]; then
  echo "ERROR: No Container Registry mounter image is specified."
  echo
  usage
fi

# Unmounts a mount point lazily. If a mount point does not exist, continue silently,
# and without error.
umount_silent() {
  umount -l $1 &> /dev/null || /bin/true
}

# Waits for kubelet to restart for 1 minute.
kubelet_wait() {
  timeout=60
  kubelet_readonly_port=10255
  until [[ $timeout -eq 0 ]]; do
    printf "."
    if [[ $( curl -s http://localhost:${kubelet_readonly_port}/healthz ) == "ok" ]]; then
      return 0
    fi
    sleep 1
    timeout=$(( timeout-1 ))
  done

  # Timed out waiting for kubelet to become healthy.
  return 1
}

flex_clean() {
  echo
  echo "An error has occurred. Cleaning up..."
  echo

  umount_silent ${VOLUME_PLUGIN_DIR}
  rm -rf ${VOLUME_PLUGIN_DIR}
  umount_silent ${MOUNTER_PATH}/var/lib/kubelet
  umount_silent ${MOUNTER_PATH}
  rm -rf ${MOUNTER_PATH}
  
  if [ -n ${IMAGE_URL:-} ]; then
    docker rmi -f ${IMAGE_URL} &> /dev/null || /bin/true
  fi
  if [ -n ${MOUNTER_DEFAULT_NAME:-} ]; then
    docker rm -f ${MOUNTER_DEFAULT_NAME} &> /dev/null || /bin/true
  fi
}

trap flex_clean ERR

# Generates a bash script that wraps all calls to the actual driver inside mount utilities
# in the chroot environment. Kubelet sees this script as the FlexVolume driver.
generate_chroot_wrapper() {
  if [ ! -d ${MOUNTER_PATH}/flexvolume ]; then
    echo "Failed to set up FlexVolume driver: cannot find directory '/flexvolume' in the mount utility image."
    exit 1
  fi
  
  for driver_dir in ${MOUNTER_PATH}/flexvolume/*; do
    if [ -d "$driver_dir" ]; then

      filecount=$(ls -1 $driver_dir | wc -l)
      if [ $filecount -gt 1 ]; then
        echo "ERROR: Expected 1 file in the FlexVolume directory but found $filecount."
        exit 1
      fi
      
      driver_file=$( ls $driver_dir | head -n 1 )

      # driver_path points to the actual driver inside the mount utility image,
      # relative to image root.
      # wrapper_path is the wrapper script location, which is known to kubelet.
      driver_path=flexvolume/$( basename $driver_dir )/${driver_file}
      wrapper_dir=${VOLUME_PLUGIN_DIR}/$( basename $driver_dir )
      wrapper_path=${wrapper_dir}/${driver_file}

      mkdir -p $wrapper_dir
      cat >$wrapper_path <<EOF
#!/bin/bash
chroot ${MOUNTER_PATH} ${driver_path} "\$@"
EOF

      chmod 755 $wrapper_path
      echo "FlexVolume driver installed at ${wrapper_path}"
    fi
  done
}

echo
echo "Importing mount utility image from Container Registry..."
echo

METADATA=http://metadata.google.internal/computeMetadata/v1
SVC_ACCT_ENDPOINT=$METADATA/instance/service-accounts/default
ACCESS_TOKEN=$(curl -s -H 'Metadata-Flavor: Google' $SVC_ACCT_ENDPOINT/token | cut -d'"' -f 4)
PROJECT_ID=$(curl -s -H 'Metadata-Flavor: Google' $METADATA/project/project-id)
IMAGE_URL=gcr.io/${PROJECT_ID}/${MOUNTER_IMAGE}
MOUNTER_DEFAULT_NAME=flexvolume_mounter
sudo -u ${SUDO_USER} docker login -u _token -p $ACCESS_TOKEN https://gcr.io > /dev/null
sudo -u ${SUDO_USER} docker run --name=${MOUNTER_DEFAULT_NAME} ${IMAGE_URL}
docker export ${MOUNTER_DEFAULT_NAME} > /tmp/${MOUNTER_DEFAULT_NAME}.tar
docker rm ${MOUNTER_DEFAULT_NAME} > /dev/null
docker rmi ${IMAGE_URL} > /dev/null

echo
echo "Loading mount utilities onto this instance..."
echo

mkdir ${MOUNTER_PATH}
tar xf /tmp/${MOUNTER_DEFAULT_NAME}.tar -C ${MOUNTER_PATH}

# Bind the kubelet directory to one under flexvolume_mounter
mkdir ${MOUNTER_PATH}/var/lib/kubelet
mount --rbind /var/lib/kubelet/ ${MOUNTER_PATH}/var/lib/kubelet
mount --make-rshared ${MOUNTER_PATH}/var/lib/kubelet

# Remount the flexvolume_mounter environment with /dev enabled.
mount --bind ${MOUNTER_PATH} ${MOUNTER_PATH}
mount -o remount,dev,exec ${MOUNTER_PATH}

echo
echo "Setting up FlexVolume driver..."
echo

mkdir -p ${VOLUME_PLUGIN_DIR}
mount --bind ${VOLUME_PLUGIN_DIR} ${VOLUME_PLUGIN_DIR}
mount -o remount,exec ${VOLUME_PLUGIN_DIR}
generate_chroot_wrapper

echo
echo "Restarting Kubelet..."
echo

systemctl restart kubelet.service
kubelet_wait
if [ $? -eq 0 ]; then
  echo
  echo "FlexVolume is ready."
else
  echo "ERROR: Timed out after 1 minute waiting for kubelet restart."
fi
