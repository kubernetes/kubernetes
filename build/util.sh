#!/bin/bash

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

# Common utility functions for build scripts

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

function kube::release::semantic_version() {
  # This takes:
  # Client Version: version.Info{Major:"1", Minor:"1+", GitVersion:"v1.1.0-alpha.0.2328+3c0a05de4a38e3", GitCommit:"3c0a05de4a38e355d147dbfb4d85bad6d2d73bb9", GitTreeState:"clean"}
  # and spits back the GitVersion piece in a way that is somewhat
  # resilient to the other fields changing (we hope)
  ${KUBE_ROOT}/cluster/kubectl.sh version -c | sed "s/, */\\
/g" | egrep "^GitVersion:" | cut -f2 -d: | cut -f2 -d\"
}

function kube::release::semantic_image_tag_version() {
    printf "$(kube::release::semantic_version)" | tr + _
}
