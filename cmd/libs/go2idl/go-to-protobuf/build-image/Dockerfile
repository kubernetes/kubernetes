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

# This file creates a standard build environment for building Kubernetes
FROM gcr.io/google_containers/kube-cross:KUBE_BUILD_IMAGE_CROSS_TAG

# Mark this as a kube-build container
RUN touch /kube-build-image

WORKDIR /go/src/k8s.io/kubernetes

# Install goimports tool
RUN go get golang.org/x/tools/cmd/goimports
