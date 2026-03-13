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

# The line below points to the latest go-runner image, a wrapper around
# gcr.io/distroless/static which adds the go-runner command that we need
# for redirecting log output.
#
# See https://console.cloud.google.com/gcr/images/k8s-staging-build-image/global/go-runner
# for a list of available versions. This base image should be updated
# periodically.
FROM registry.k8s.io/build-image/go-runner:v2.3.1-go1.17.2-bullseye.0

COPY kubemark /kubemark
