# Copyright 2021 The Kubernetes Authors.
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

# This file create the kube-apiserver image.
ARG BASEIMAGE

FROM k8s.gcr.io/build-image/setcap:buster-v1.4.0
ARG BINARY
COPY ${BINARY} /${BINARY}
# We apply cap_net_bind_service so that kube-apiserver can be run as
# non-root and still listen on port less than 1024
RUN setcap cap_net_bind_service=+ep /${BINARY}

FROM ${BASEIMAGE}
ARG BINARY
COPY --from=0 /${BINARY} /usr/local/bin/${BINARY}
