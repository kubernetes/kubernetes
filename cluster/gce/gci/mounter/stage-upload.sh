#!/bin/sh

# Copyright 2016 The Kubernetes Authors.
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

# Due to the GCE custom metadata size limit, we split the entire script into two
# files configure.sh and configure-helper.sh. The functionality of downloading
# kubernetes configuration, manifests, docker images, and binary files are
# put in configure.sh, which is uploaded via GCE custom metadata.

set -o errexit
set -o pipefail
set -o nounset

RKT_VERSION="v1.18.0"
DOCKER2ACI_VERSION="v0.13.0"
MOUNTER_VERSION=$1
DOCKER_IMAGE=docker://$2
MOUNTER_ACI_IMAGE=gci-mounter-${MOUNTER_VERSION}.aci
RKT_GCS_DIR=gs://kubernetes-release/rkt/
MOUNTER_GCS_DIR=gs://kubernetes-release/gci-mounter/

TMPDIR=/tmp
# Setup a working directory
DOWNLOAD_DIR=$(mktemp --tmpdir=${TMPDIR} -d gci-mounter-build.XXXXXXXXXX)

# Setup a staging directory
STAGING_DIR=$(mktemp --tmpdir=${TMPDIR} -d gci-mounter-staging.XXXXXXXXXX)
RKT_DIR=${STAGING_DIR}/${RKT_VERSION}
ACI_DIR=${STAGING_DIR}/gci-mounter
CWD=${PWD}

# Cleanup the temporary directories
function cleanup {
    rm -rf ${DOWNLOAD_DIR}
    rm -rf ${STAGING_DIR}
    cd ${CWD}
}

# Delete temporary directories on exit
trap cleanup EXIT

mkdir ${RKT_DIR}
mkdir ${ACI_DIR}

# Download rkt
cd ${DOWNLOAD_DIR}
echo "Downloading rkt ${RKT_VERSION}"
wget "https://github.com/coreos/rkt/releases/download/${RKT_VERSION}/rkt-${RKT_VERSION}.tar.gz" &> /dev/null
echo "Extracting rkt ${RKT_VERSION}"
tar xzf rkt-${RKT_VERSION}.tar.gz

# Stage rkt into working directory
cp rkt-${RKT_VERSION}/rkt ${RKT_DIR}/rkt
cp rkt-${RKT_VERSION}/stage1-fly.aci ${RKT_DIR}/

# Convert docker image to aci and stage it
echo "Downloading docker2aci ${DOCKER2ACI_VERSION}"
wget "https://github.com/appc/docker2aci/releases/download/${DOCKER2ACI_VERSION}/docker2aci-${DOCKER2ACI_VERSION}.tar.gz" &> /dev/null
echo "Extracting docker2aci ${DOCKER2ACI_VERSION}"
tar xzf docker2aci-${DOCKER2ACI_VERSION}.tar.gz
ACI_IMAGE=$(${DOWNLOAD_DIR}/docker2aci-${DOCKER2ACI_VERSION}/docker2aci ${DOCKER_IMAGE} 2>/dev/null | tail -n 1)
cp ${ACI_IMAGE} ${ACI_DIR}/${MOUNTER_ACI_IMAGE}

# Upload the contents to gcs
echo "Uploading rkt artifacts in ${RKT_DIR} to ${RKT_GCS_DIR}"
gsutil cp -R ${RKT_DIR} ${RKT_GCS_DIR}
echo "Uploading gci mounter ACI in ${ACI_DIR} to ${MOUNTER_GCS_DIR}"
gsutil cp ${ACI_DIR}/${MOUNTER_ACI_IMAGE} ${MOUNTER_GCS_DIR}

echo "Upload completed"
echo "Update rkt, stag1-fly.aci & gci-mounter ACI versions and SHA1 in cluster/gce/gci/configure.sh"
echo "${RKT_VERSION}/rkt sha1: $(sha1sum ${RKT_DIR}/rkt)"
echo "${RKT_VERSION}/stage1-fly.aci sha1: $(sha1sum ${RKT_DIR}/stage1-fly.aci)"
echo "${MOUNTER_ACI_IMAGE} hash: $(sha1sum ${ACI_DIR}/${MOUNTER_ACI_IMAGE})"
