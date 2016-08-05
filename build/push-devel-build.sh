#!/bin/bash

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

# Pushes a development build to a directory in your current project,
# pushing to something like:
# gs://kubernetes-releases-3fda2/devel/v0.8.0-437-g7f147ed/

set -o errexit
set -o nounset
set -o pipefail

LATEST=$(git describe)
KUBE_GCS_NO_CACHING=n
KUBE_GCS_MAKE_PUBLIC=y
KUBE_GCS_UPLOAD_RELEASE=y
KUBE_GCS_RELEASE_PREFIX="devel/${LATEST}"

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/build/common.sh"

kube::release::gcs::release
