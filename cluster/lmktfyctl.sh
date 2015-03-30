#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

LMKTFY_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${LMKTFY_ROOT}/cluster/lmktfy-env.sh"
UTILS=${LMKTFY_ROOT}/cluster/${LMKTFYRNETES_PROVIDER}/util.sh
if [ -f ${UTILS} ]; then
    source "${UTILS}"
fi

# Get the absolute path of the directory component of a file, i.e. the
# absolute path of the dirname of $1.
get_absolute_dirname() {
  echo "$(cd "$(dirname "$1")" && pwd)"
}

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
    host_arch=x86
    ;;
  *)
    echo "Unsupported host arch. Must be x86_64, 386 or arm." >&2
    exit 1
    ;;
esac

# If LMKTFYCTL_PATH isn't set, gather up the list of likely places and use ls
# to find the latest one.
if [[ -z "${LMKTFYCTL_PATH:-}" ]]; then
  locations=(
    "${LMKTFY_ROOT}/_output/dockerized/bin/${host_os}/${host_arch}/lmktfyctl"
    "${LMKTFY_ROOT}/_output/local/bin/${host_os}/${host_arch}/lmktfyctl"
    "${LMKTFY_ROOT}/platforms/${host_os}/${host_arch}/lmktfyctl"
  )
  lmktfyctl=$( (ls -t "${locations[@]}" 2>/dev/null || true) | head -1 )

  if [[ ! -x "$lmktfyctl" ]]; then
    {
      echo "It looks as if you don't have a compiled lmktfyctl binary"
      echo
      echo "If you are running from a clone of the git repo, please run"
      echo "'./build/run.sh hack/build-cross.sh'. Note that this requires having"
      echo "Docker installed."
      echo
      echo "If you are running from a binary release tarball, something is wrong. "
      echo "Look at http://lmktfy.io/ for information on how to contact the "
      echo "development team for help."
    } >&2
    exit 1
  fi
elif [[ ! -x "${LMKTFYCTL_PATH}" ]]; then
  {
    echo "LMKTFYCTL_PATH environment variable set to '${LMKTFYCTL_PATH}', but "
    echo "this doesn't seem to be a valid executable."
  } >&2
  exit 1
fi
lmktfyctl="${LMKTFYCTL_PATH:-${lmktfyctl}}"

# While GKE requires the lmktfyctl binary, it's actually called through
# gcloud. But we need to adjust the PATH so gcloud gets the right one.
if [[ "$LMKTFYRNETES_PROVIDER" == "gke" ]]; then
  detect-project &> /dev/null
  export PATH=$(get_absolute_dirname $lmktfyctl):$PATH
  lmktfyctl="${GCLOUD}"
  # GKE runs lmktfyctl through gcloud.
  config=(
    "preview"
    "container"
    "lmktfyctl"
    "--project=${PROJECT}"
    "--zone=${ZONE}"
    "--cluster=${CLUSTER_NAME}"
  )
elif [[ "$LMKTFYRNETES_PROVIDER" == "vagrant" ]]; then
  # When we are using vagrant it has hard coded lmktfyconfig, and do not clobber public endpoints
  config=(
    "--lmktfyconfig=$HOME/.lmktfy_vagrant_lmktfyconfig"
  )
elif [[ "$LMKTFYRNETES_PROVIDER" == "libvirt-coreos" ]]; then
  detect-master > /dev/null
  config=(
    "--server=http://${LMKTFY_MASTER_IP}:8080"
  )
fi

echo "current-context: \"$(${lmktfyctl} "${config[@]:+${config[@]}}" config view -o template --template='{{index . "current-context"}}')\"" >&2

echo "Running:" "${lmktfyctl}" "${config[@]:+${config[@]}}" "${@+$@}" >&2
"${lmktfyctl}" "${config[@]:+${config[@]}}" "${@+$@}"
