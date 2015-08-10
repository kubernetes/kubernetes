#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# Run a command in the docker build container.  Typically this will be one of
# the commands in `hack/`.  When running in the build container the user is sure
# to have a consistent reproducible build environment.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE}")/.." && pwd)

GOPATH=$(cd "${KUBE_ROOT}/../../.." && pwd)
GOPATH="${GOPATH}:${KUBE_ROOT}/Godeps/_workspace"

# TODO: install counterfeiter? use container? https://github.com/maxbrunsfeld/counterfeiter
# go install github.com/maxbrunsfeld/counterfeiter

GOPATH="${GOPATH}" counterfeiter -o pkg/registry/component/mocks/mocks.go --fake-name MockRegistry pkg/registry/component Registry
GOPATH="${GOPATH}" counterfeiter -o pkg/probe/http/mocks/mocks.go --fake-name MockHTTPGetter pkg/probe/http HTTPGetter
GOPATH="${GOPATH}" counterfeiter -o pkg/storage/mocks/mocks.go --fake-name MockInterface pkg/storage Interface

# TODO: fix pkg/probe/http/mocks/mocks.go:8: http redeclared as imported package name
# TODO: add boilerplate