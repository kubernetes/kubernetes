#!/bin/bash

set -ex

# This script downloads a Kubernetes release and creates a tar file with only
# the files that are needed for this charm.

# Usage: create_kubernetes_tar.sh VERSION ARCHITECTURE

usage() {
  echo "Build a tar file with only the files needed for the kubernetes charm."
  echo "The script accepts two arguments version and desired architecture."
  echo "$0 version architecture"
}

download_kubernetes() {
  local VERSION=$1
  URL_PREFIX="https://github.com/GoogleCloudPlatform/kubernetes"
  KUBERNETES_URL="${URL_PREFIX}/releases/download/${VERSION}/kubernetes.tar.gz"
  # Remove the previous temporary files to remain idempotent.
  if [ -f /tmp/kubernetes.tar.gz ]; then
    rm /tmp/kubernetes.tar.gz
  fi
  # Download the kubernetes release from the Internet.
  wget --no-verbose --tries 2 -O /tmp/kubernetes.tar.gz $KUBERNETES_URL
}

extract_kubernetes() {
  local ARCH=$1
  # Untar the kubernetes release file.
  tar -xvzf /tmp/kubernetes.tar.gz -C /tmp
  # Untar the server linux amd64 package.
  tar -xvzf /tmp/kubernetes/server/kubernetes-server-linux-$ARCH.tar.gz -C /tmp
}

create_charm_tar() {
  local OUTPUT_FILE=${1:-"$PWD/kubernetes.tar.gz"}
  local OUTPUT_DIR=`dirname $OUTPUT_FILE`
  if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT
  fi

  # Change to the directory the binaries are.
  cd /tmp/kubernetes/server/bin/

  # Create a tar file with the binaries that are needed for kubernetes master.
  tar -cvzf $OUTPUT_FILE kube-apiserver kube-controller-manager kubectl kube-scheduler
}

if [ $# -gt 2 ]; then
  usage
  exit 1
fi
VERSION=${1:-"v0.8.1"}
ARCH=${2:-"amd64"}
download_kubernetes $VERSION
extract_kubernetes $ARCH
TAR_FILE="$PWD/kubernetes-master-$VERSION-$ARCH.tar.gz"
create_charm_tar $TAR_FILE
