#!/bin/bash

# Copyright 2018 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::log::status "Ensuring prereqs"
kube::util::ensure-gnu-sed

cd "${KUBE_ROOT}"
go run ./cmd/staginghub/main.go &>/tmp/staginghub.log &
STAGINGHUB_PID=$!

function kube::dep::cleanup() {
    kill ${STAGINGHUB_PID} &>/dev/null || true

    # fixup Gopkg.toml
    if [ -f Gopkg.toml ]; then
      ${SED} -i 's/.*\(source = .*localhost:12345.*\)/#  \1/' Gopkg.toml
    fi

    # fixup Gopkg.lock
    # TODO: remove staging repos

    # restore staging symlinks
    for d in $(cd staging/src/k8s.io; ls -1); do
        rm -rf vendor/k8s.io/${d}
        ln -s ../../staging/src/k8s.io/${d} vendor/k8s.io/${d}
    done
}
kube::util::trap_add kube::dep::cleanup EXIT

if [ -f Gopkg.toml ]; then
  ${SED} -i 's/#.*\(source = .*localhost:12345.*\)/  \1/' Gopkg.toml
fi

dep "$@"
