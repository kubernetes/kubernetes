#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# This script will set up the salt directory on the target server.  It takes one
# argument that is a tarball with the pre-compiled kubernetes server binaries.

set -o errexit
set -o nounset
set -o pipefail

function get-tokens() {
  KUBELET_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  KUBE_PROXY_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  MASTER_USER="root"
  MASTER_PASSWD=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=12 count=1 2>/dev/null)
}
get-tokens

SALT_ROOT=$(dirname "${BASH_SOURCE}")
readonly SALT_ROOT

readonly KUBE_DOCKER_WRAPPED_BINARIES=(
  kube-apiserver
  kube-controller-manager
  kube-scheduler
  kube-proxy
)

readonly SERVER_BIN_TAR=${1-}
if [[ -z "$SERVER_BIN_TAR" ]]; then
  echo "!!! No binaries specified"
  exit 1
fi

# Create a temp dir for untaring
KUBE_TEMP=$(mktemp --tmpdir=/srv -d -t kubernetes.XXXXXX)
trap 'rm -rf "${KUBE_TEMP}"' EXIT

# This file is meant to run on the master.  It will install the salt configs
# into the appropriate place on the master.  We do this by creating a new set of
# salt trees and then quickly mv'ing them where the old ones were.

readonly SALTDIRS=(salt pillar reactor)

echo "+++ Installing salt files into new trees"
rm -rf /srv/salt-new
mkdir -p /srv/salt-new

# This bash voodoo will prepend $SALT_ROOT to the start of each item in the
# $SALTDIRS array
cp -v -R --preserve=mode "${SALTDIRS[@]/#/${SALT_ROOT}/}" /srv/salt-new

echo "+++ Installing salt overlay files"
for dir in "${SALTDIRS[@]}"; do
  if [[ -d "/srv/salt-overlay/$dir" ]]; then
    cp -v -R --preserve=mode "/srv/salt-overlay/$dir" "/srv/salt-new/"
  fi
done

echo "+++ Install binaries from tar: $1"
tar -xz -C "${KUBE_TEMP}" -f "$1"
mkdir -p /srv/salt-new/salt/kube-bins
mkdir -p /srv/salt-new/salt/kube-addons-images
cp -v "${KUBE_TEMP}/kubernetes/server/bin/"* /srv/salt-new/salt/kube-bins/
cp -v "${KUBE_TEMP}/kubernetes/addons/"* /srv/salt-new/salt/kube-addons-images/

kube_bin_dir="/srv/salt-new/salt/kube-bins";
docker_images_sls_file="/srv/salt-new/pillar/docker-images.sls";
for docker_file in "${KUBE_DOCKER_WRAPPED_BINARIES[@]}"; do
  docker_tag=$(cat ${kube_bin_dir}/${docker_file}.docker_tag);
  if [[ ! -z "${KUBE_IMAGE_TAG:-}" ]]; then
    docker_tag="${KUBE_IMAGE_TAG}"
  fi
  sed -i "s/#${docker_file}_docker_tag_value#/${docker_tag}/" "${docker_images_sls_file}";
done

cat <<EOF >>"${docker_images_sls_file}"
kube_docker_registry: '$(echo ${KUBE_DOCKER_REGISTRY:-gcr.io/google_containers})'
EOF

# Generate and distribute a shared secret (bearer token) to
# apiserver and kubelet so that kubelet can authenticate to
# apiserver to send events.
echo "+++ Creating kube-apiserver token files"
readonly known_tokens_file="/srv/salt-new/salt/kube-apiserver/known_tokens.csv"
if [[ ! -f "${known_tokens_file}" ]]; then
  mkdir -p /srv/salt-new/salt/kube-apiserver
  (umask u=rw,go= ;
    echo "$KUBELET_TOKEN,kubelet,kubelet" > $known_tokens_file;
    echo "$KUBE_PROXY_TOKEN,kube_proxy,kube_proxy" >> $known_tokens_file)
fi

readonly BASIC_AUTH_FILE="/srv/salt-new/salt/kube-apiserver/basic_auth.csv"
if [ ! -e "${BASIC_AUTH_FILE}" ]; then
  mkdir -p /srv/salt-new/salt/kube-apiserver
  (umask u=rwx,go=;
    echo "${MASTER_USER},${MASTER_PASSWD},admin" > "${BASIC_AUTH_FILE}")
fi

echo "+++ Swapping in new configs"
for dir in "${SALTDIRS[@]}"; do
  if [[ -d "/srv/$dir" ]]; then
    rm -rf "/srv/$dir"
  fi
  mv -v "/srv/salt-new/$dir" "/srv/$dir"
done

rm -rf /srv/salt-new
