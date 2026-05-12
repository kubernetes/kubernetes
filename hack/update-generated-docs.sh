#!/usr/bin/env bash

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

# This file is not intended to be run automatically. It is meant to be run
# immediately before exporting docs. We do not want to check these documents in
# by default.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env
kube::util::ensure-temp-dir

BINS=(
	./cmd/gendocs
	./cmd/genkubedocs
	./cmd/genman
	./cmd/genyaml
)
GOPROXY=off go install "${BINS[@]}"

# Run all doc generators.
# $1 is the directory to put those generated documents
generate_docs() {
  local dest="$1"

  mkdir -p "${dest}/docs/user-guide/kubectl/"
  gendocs "${dest}/docs/user-guide/kubectl/"

  mkdir -p "${dest}/docs/admin/"
  genkubedocs "${dest}/docs/admin/" "kube-apiserver"
  genkubedocs "${dest}/docs/admin/" "kube-controller-manager"
  genkubedocs "${dest}/docs/admin/" "kube-proxy"
  genkubedocs "${dest}/docs/admin/" "kube-scheduler"
  genkubedocs "${dest}/docs/admin/" "kubelet"
  genkubedocs "${dest}/docs/admin/" "kubeadm"

  mkdir -p "${dest}/docs/man/man1/"
  genman "${dest}/docs/man/man1/" "kube-apiserver"
  genman "${dest}/docs/man/man1/" "kube-controller-manager"
  genman "${dest}/docs/man/man1/" "kube-proxy"
  genman "${dest}/docs/man/man1/" "kube-scheduler"
  genman "${dest}/docs/man/man1/" "kubelet"
  genman "${dest}/docs/man/man1/" "kubectl"
  genman "${dest}/docs/man/man1/" "kubeadm"

  mkdir -p "${dest}/docs/yaml/kubectl/"
  genyaml "${dest}/docs/yaml/kubectl/"

  # create the list of generated files
  pushd "${dest}" > /dev/null || return 1
  touch docs/.generated_docs
  find . -type f | cut -sd / -f 2- | LC_ALL=C sort > docs/.generated_docs
  popd > /dev/null || return 1
}

# Removes previously generated docs-- we don't want to check them in. $KUBE_ROOT
# must be set.
remove_generated_docs() {
  if [ -e "${KUBE_ROOT}/docs/.generated_docs" ]; then
    # remove all of the old docs; we don't want to check them in.
    while read -r file; do
      rm "${KUBE_ROOT}/${file}" 2>/dev/null || true
    done <"${KUBE_ROOT}/docs/.generated_docs"
    # The docs/.generated_docs file lists itself, so we don't need to explicitly
    # delete it.
  fi
}

# generate into KUBE_TMP
generate_docs "${KUBE_TEMP}"

# remove all of the existing docs in KUBE_ROOT
remove_generated_docs

# Copy fresh docs into the repo.
# the shopt is so that we get docs/.generated_docs from the glob.
shopt -s dotglob
cp -af "${KUBE_TEMP}"/* "${KUBE_ROOT}"
shopt -u dotglob
