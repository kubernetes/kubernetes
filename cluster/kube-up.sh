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

# Bring up a Kubernetes cluster.
#
# If the full release name (gs://<bucket>/<release>) is passed in then we take
# that directly.  If not then we assume we are doing development stuff and take
# the defaults in the release config.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

if [ -f "${KUBE_ROOT}/cluster/env.sh" ]; then
    source "${KUBE_ROOT}/cluster/env.sh"
fi

source "${KUBE_ROOT}/cluster/kube-util.sh"

DEPRECATED_PROVIDERS=(
  "centos"
  "local"
)

for provider in "${DEPRECATED_PROVIDERS[@]}"; do
  if [[ "${KUBERNETES_PROVIDER}" == "${provider}" ]]; then
    cat <<EOF 1>&2

!!! DEPRECATION NOTICE !!!

The '${provider}' kube-up provider is deprecated and will be removed in a future
release of kubernetes. Deprecated providers will be removed within 2 releases.

See https://github.com/kubernetes/kubernetes/issues/49213 for more info.

EOF
    break
  fi
done

if [ -z "${ZONE-}" ]; then
  echo "... Starting cluster using provider: ${KUBERNETES_PROVIDER}" >&2
else
  echo "... Starting cluster in ${ZONE} using provider ${KUBERNETES_PROVIDER}" >&2
fi

echo "... calling verify-prereqs" >&2
verify-prereqs
echo "... calling verify-kube-binaries" >&2
verify-kube-binaries
echo "... calling verify-release-tars" >&2
verify-release-tars

echo "... calling kube-up" >&2
kube-up

echo "... calling validate-cluster" >&2
# Override errexit
(validate-cluster) && validate_result="$?" || validate_result="$?"

# We have two different failure modes from validate cluster:
# - 1: fatal error - cluster won't be working correctly
# - 2: weak error - something went wrong, but cluster probably will be working correctly
# We just print an error message in case 2).
if [[ "${validate_result}" == "1" ]]; then
	exit 1
elif [[ "${validate_result}" == "2" ]]; then
	echo "...ignoring non-fatal errors in validate-cluster" >&2
fi

if [[ "${ENABLE_PROXY:-}" == "true" ]]; then
  . /tmp/kube-proxy-env
  echo ""
  echo "*** Please run the following to add the kube-apiserver endpoint to your proxy white-list ***"
  cat /tmp/kube-proxy-env
  echo "***                                                                                      ***"
  echo ""
fi

echo -e "Done, listing cluster services:\n" >&2
"${KUBE_ROOT}/cluster/kubectl.sh" cluster-info
echo

exit 0
