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

# This script will source the default skeleton helper functions, then sources
# cluster/${KUBERNETES_PROVIDER}/util.sh where KUBERNETES_PROVIDER, if unset,
# will use its default value (gce).

# TODO(b/197113765): Remove this script and use binary directly.
if [[ -e "$(dirname "${BASH_SOURCE[0]}")/../../hack/lib/util.sh" ]]; then
  # When kubectl.sh is used directly from the repo, it's under gke/cluster.
  KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
else
  # When kubectl.sh is used from unpacked tarball, it's under cluster.
  KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
fi

source "$(dirname "${BASH_SOURCE[0]}")/skeleton/util.sh"

if [[ -n "${KUBERNETES_CONFORMANCE_TEST:-}" ]]; then
    KUBERNETES_PROVIDER=""
else
    KUBERNETES_PROVIDER="${KUBERNETES_PROVIDER:-gce}"
fi

# PROVIDER_VARS is a list of cloud provider specific variables. Note:
# this is a list of the _names_ of the variables, not the value of the
# variables. Providers can add variables to be appended to kube-env.
# (see `build-kube-env`).

PROVIDER_UTILS="$(dirname "${BASH_SOURCE[0]}")/${KUBERNETES_PROVIDER}/util.sh"
if [ -f "${PROVIDER_UTILS}" ]; then
    source "${PROVIDER_UTILS}"
fi
