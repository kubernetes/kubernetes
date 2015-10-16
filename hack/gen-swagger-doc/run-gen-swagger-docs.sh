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

# Usage: run-gen-swagger-docs.sh <absolute output path, default to PWD>

OUTPUT=${1:-${PWD}}

docker run -v ${OUTPUT}:/output gcr.io/google_containers/gen-swagger-docs:v1.1 https://raw.githubusercontent.com/kubernetes/kubernetes/master/api/swagger-spec/v1.json https://raw.githubusercontent.com/kubernetes/kubernetes/master/pkg/api/v1/register.go
docker run -v ${OUTPUT}:/output gcr.io/google_containers/gen-swagger-docs:v1.1 https://raw.githubusercontent.com/kubernetes/kubernetes/master/api/swagger-spec/v1beta1.json https://raw.githubusercontent.com/kubernetes/kubernetes/master/pkg/apis/extensions/v1beta1/register.go
