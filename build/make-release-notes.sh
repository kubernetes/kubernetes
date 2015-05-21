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

# Clean out the output directory on the docker host.
set -o errexit
set -o nounset
set -o pipefail

function pop_dir {
  popd > /dev/null
}

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

if [[ -z "${1:-}" ]]; then
  echo "Usage: ${0} <pr-number>"
  exit 1
fi

pushd . > /dev/null
trap 'pop_dir' INT TERM EXIT

cd ${KUBE_ROOT}/contrib/release-notes
# TODO: vendor these dependencies, but using godep again will be annoying...
GOPATH=$PWD go get github.com/google/go-github/github
GOPATH=$PWD go get github.com/google/go-querystring/query
GOPATH=$PWD go build release-notes.go
./release-notes --last-release-pr=${1}

