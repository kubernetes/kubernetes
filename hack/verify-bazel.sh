#!/usr/bin/env bash
# Copyright 2015 The Kubernetes Authors.
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

# Update the Godeps/LICENSES document.
# Generates a table of Godep dependencies and their license.
#
# Usage:
#    $0 [--create-missing] [/path/to/licenses]
#
#    --create-missing will write the files that only exist upstream, locally.
#    This option is mostly used for testing as we cannot check-in any of the
#    additionally created files into the godep auto-generated tree.
#
#    Run every time a license file is added/modified within /Godeps to
#    update /Godeps/LICENSES

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

go get github.com/mikedanese/gazel
if [[ $("${GOPATH}/bin/gazel" -dry-run |& tee /dev/stderr | wc -l) != 0 ]]; then
  echo
  echo "BUILD files are not up to date"
  echo "Run ./hack/update-bazel.sh"
  exit 1
fi
