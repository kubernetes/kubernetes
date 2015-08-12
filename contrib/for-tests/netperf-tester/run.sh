#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# This script finds kubectl and then calls netperf_tester.
# It is essentially a copy of cluster/kubectl.sh

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../../..

# Detect the OS name/arch so that we can find our binary
case "$(uname -s)" in
  Darwin)
    host_os=darwin
    ;;
  Linux)
    host_os=linux
    ;;
  *)
    echo "Unsupported host OS.  Must be Linux or Mac OS X." >&2
    exit 1
    ;;
esac

case "$(uname -m)" in
  x86_64*)
    host_arch=amd64
    ;;
  i?86_64*)
    host_arch=amd64
    ;;
  amd64*)
    host_arch=amd64
    ;;
  arm*)
    host_arch=arm
    ;;
  i?86*)
    host_arch=386
    ;;
  *)
    echo "Unsupported host arch. Must be x86_64, 386 or arm." >&2
    exit 1
    ;;
esac

# If KUBECTL_PATH isn't set, gather up the list of likely places and use ls
# to find the latest one.
if [[ -z "${KUBECTL_PATH:-}" ]]; then
  locations=(
    "${KUBE_ROOT}/_output/dockerized/bin/${host_os}/${host_arch}/kubectl"
    "${KUBE_ROOT}/_output/local/bin/${host_os}/${host_arch}/kubectl"
    "${KUBE_ROOT}/platforms/${host_os}/${host_arch}/kubectl"
  )
  kubectl=$( (ls -t "${locations[@]}" 2>/dev/null || true) | head -1 )

  if [[ ! -x "$kubectl" ]]; then
    {
      echo "It looks as if you don't have a compiled kubectl binary"
      echo
      echo "If you are running from a clone of the git repo, please run"
      echo "'./build/run.sh hack/build-cross.sh'. Note that this requires having"
      echo "Docker installed."
      echo
      echo "If you are running from a binary release tarball, something is wrong. "
      echo "Look at http://kubernetes.io/ for information on how to contact the "
      echo "development team for help."
    } >&2
    exit 1
  fi
elif [[ ! -x "${KUBECTL_PATH}" ]]; then
  {
    echo "KUBECTL_PATH environment variable set to '${KUBECTL_PATH}', but "
    echo "this doesn't seem to be a valid executable."
  } >&2
  exit 1
fi
KUBECTL="${KUBECTL_PATH:-${kubectl}}"

export KUBECTL
export KUBE_ROOT
go run $(dirname "${BASH_SOURCE}")/netperf_tester.go --kubectl="${KUBECTL}" "$@"
