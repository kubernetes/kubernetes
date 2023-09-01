#!/usr/bin/env bash

# Copyright 2021 The Kubernetes Authors.
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

# This script checks whether the OWNERS files need to be formatted or not by
# `yamlfmt`. Run `hack/update-yamlfmt.sh` to actually format sources.
#
# Usage: `hack/verify-yamlfmt.sh`.


set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/verify-generated.sh"

kube::verify::generated "YAML files need to be formatted" "Please run 'hack/update-yamlfmt.sh'" hack/update-yamlfmt.sh
