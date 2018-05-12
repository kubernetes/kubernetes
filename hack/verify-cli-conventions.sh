#!/usr/bin/env bash

# Copyright 2016 The Kubernetes Authors.
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

kube::golang::setup_env

BINS=(
	cmd/clicheck
)
make -C "${KUBE_ROOT}" WHAT="${BINS[*]}"

clicheck=$(kube::util::find-binary "clicheck")

if ! output=$($clicheck 2>&1)
then
	echo "$output"
	echo
	echo "FAILURE: CLI is not following one or more required conventions."
	exit 1
else
	echo "$output"
	echo
  echo "SUCCESS: CLI is following all tested conventions."
fi
