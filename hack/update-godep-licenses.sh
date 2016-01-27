#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# Update the Godeps/LICENSES.md document.
# Generates a table of Godep dependencies and their license.
# Requires:
#    docker
#    mesosphere/godep-licenses (docker image) - source: https://github.com/mesosphere/godep-licenses
# Usage:
#    Run every time a license file is added/modified within /Godeps to update /Godeps/LICENSES.md.
#    Add exceptions (-e <repo>:<license>) for any dependency (project) vendored by Godep
#      that has a known license that isn't vendored by Godep or can't be found by godep-licenses.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT="${KUBE_ROOT:-$(cd "$(dirname "${BASH_SOURCE}")/.." && pwd -P)}"

cd "${KUBE_ROOT}"

exec docker run --rm -i -v "${KUBE_ROOT}:/repo" mesosphere/godep-licenses:latest -p /repo \
  -e github.com/abbot/go-http-auth:Apache-2 \
  -e github.com/beorn7/perks/quantile:MIT? \
  -e github.com/daviddengcn/go-colortext:BSD? \
  -e github.com/docker/docker/pkg/symlink:spdxBSD3 \
  -e github.com/shurcooL/sanitized_anchor_name:MIT? \
  -e github.com/spf13/cobra:Apache-2 \
  -e github.com/stretchr/objx:MIT? \
  -e github.com/docker/spdystream:SeeFile \
  -e gopkg.in/yaml.v2:LesserExceptionGPLVer3-TOOLONG \
  -o md > Godeps/LICENSES.md
