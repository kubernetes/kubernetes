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

if [ "$#" -lt 1 ]; then
    echo "Usage: run-gen-swagger-docs.sh <API version> <absolute output path, default to PWD>"
    exit
fi
OUTPUT=${2:-${PWD}}

KUBE_ROOT=$(realpath $(dirname "${BASH_SOURCE}")/../..)

docker run \
    -v ${OUTPUT}:/output \
    -v ${KUBE_ROOT}:/kube \
    gcr.io/google_containers/gen-swagger-docs:v1 /kube/api/swagger-spec/$1.json /kube/pkg/api/$1/register.go

