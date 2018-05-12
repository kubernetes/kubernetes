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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

BINS=(
	cmd/gendocs
	cmd/genkubedocs
	cmd/genman
	cmd/genyaml
)
make -C "${KUBE_ROOT}" WHAT="${BINS[*]}"

kube::util::ensure-temp-dir

kube::util::gen-docs "${KUBE_TEMP}"

# remove all of the old docs
kube::util::remove-gen-docs

# Copy fresh docs into the repo.
# the shopt is so that we get docs/.generated_docs from the glob.
shopt -s dotglob
cp -af "${KUBE_TEMP}"/* "${KUBE_ROOT}"
shopt -u dotglob

# Replace with placeholder docs
kube::util::set-placeholder-gen-docs
