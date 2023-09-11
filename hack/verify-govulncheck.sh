#!/usr/bin/env bash
# Copyright 2023 The Kubernetes Authors.
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

set -euo pipefail

export WORKDIR=${ARTIFACTS:-$TMPDIR}
export PATH=$PATH:$GOPATH/bin
mkdir -p "${WORKDIR}"
go install golang.org/x/vuln/cmd/govulncheck@v1.0.1

govulncheck -scan module ./... > "${WORKDIR}/head.txt"

git reset --hard HEAD
git checkout -b base "${PULL_BASE_SHA}"
govulncheck -scan module ./... > "${WORKDIR}/pr-base.txt"

diff -s -u --ignore-all-space "${WORKDIR}"/pr-base.txt "${WORKDIR}"/head.txt || true
