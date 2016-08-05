#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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
mkdir -p /srv/salt-new/salt/kube-docs
cp -v "${KUBE_TEMP}/kubernetes/server/bin/"* /srv/salt-new/salt/kube-bins/
cp -v "${KUBE_TEMP}/kubernetes/LICENSES" /srv/salt-new/salt/kube-docs/
cp -v "${KUBE_TEMP}/kubernetes/kubernetes-src.tar.gz" /srv/salt-new/salt/kube-docs/

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

# TODO(zmerlynn): Forgive me, this is really gross. But in order to
# avoid breaking the non-Salt deployments, which already painfully
# have to templatize a couple of the add-ons anyways, manually
# templatize the addon registry for regional support. When we get
# better templating, we can fix this.
readonly kube_addon_registry="${KUBE_ADDON_REGISTRY:-gcr.io/google_containers}"
if [[ "${kube_addon_registry}" != "gcr.io/google_containers" ]]; then
  find /srv/salt-new -name \*.yaml -or -name \*.yaml.in | \
    xargs sed -ri "s@(image:\s.*)gcr.io/google_containers@\1${kube_addon_registry}@"
  # All the legacy .manifest files with hardcoded gcr.io are JSON.
  find /srv/salt-new -name \*.manifest -or -name \*.json | \
    xargs sed -ri "s@(image\":\s+\")gcr.io/google_containers@\1${kube_addon_registry}@"
fi

echo "+++ Swapping in new configs"
for dir in "${SALTDIRS[@]}"; do
  if [[ -d "/srv/$dir" ]]; then
    rm -rf "/srv/$dir"
  fi
  mv -v "/srv/salt-new/$dir" "/srv/$dir"
done

rm -rf /srv/salt-new
