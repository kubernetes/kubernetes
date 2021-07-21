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

set -o errexit
set -o nounset
set -o pipefail

# Stop the bleeding, turn off the warning until we fix token gen.
# echo "-=-=-=-=-=-=-=-=-=-="
# echo "NOTE:"
# echo "kubectl.sh is deprecated and will be removed soon."
# echo "please replace all usage with calls to the kubectl"
# echo "binary and ensure that it is in your PATH."
# echo ""
# echo "Please see 'kubectl help config' for more details"
# echo "about configuring kubectl for your cluster."
# echo "-=-=-=-=-=-=-=-=-=-="


KUBE_ROOT=${KUBE_ROOT:-$(dirname "${BASH_SOURCE[0]}")/..}
source "${KUBE_ROOT}/cluster/kube-util.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

# If KUBECTL_PATH isn't set, gather up the list of likely places and use ls
# to find the latest one.
if [[ -z "${KUBECTL_PATH:-}" ]]; then
  kubectl=$( kube::util::find-binary "kubectl" )

  if [[ ! -x "$kubectl" ]]; then
    {
      echo "It looks as if you don't have a compiled kubectl binary"
      echo
      echo "If you are running from a clone of the git repo, please run"
      echo "'./build/run.sh make cross'. Note that this requires having"
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
kubectl="${KUBECTL_PATH:-${kubectl}}"

if [[ "$KUBERNETES_PROVIDER" == "gke" ]]; then
  detect-project &> /dev/null
elif [[ "$KUBERNETES_PROVIDER" == "ubuntu" ]]; then
  detect-master > /dev/null
  config=(
    "--server=http://${KUBE_MASTER_IP}:8080"
  )
fi


if false; then
  # disable these debugging messages by default
  echo "current-context: \"$(${kubectl} "${config[@]:+${config[@]}}" config view -o template --template='{{index . "current-context"}}')\"" >&2
  echo "Running:" "${kubectl}" "${config[@]:+${config[@]}}" "${@+$@}" >&2
fi

if [[ "${1:-}" =~ ^(path)$ ]]; then
  echo "${kubectl}"
  exit 0
fi

"${kubectl}" "${config[@]:+${config[@]}}" "${@+$@}"

