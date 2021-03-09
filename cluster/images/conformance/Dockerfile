# Copyright 2018 The Kubernetes Authors.
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

ARG BASEIMAGE

FROM gcr.io/k8s-staging-build-image/debian-base:v2.1.3 as base

FROM ${BASEIMAGE}

# This is a dependency for `kubectl diff` tests
COPY --from=base /usr/bin/diff /usr/local/bin/

COPY ginkgo /usr/local/bin/
COPY e2e.test /usr/local/bin/
COPY kubectl /usr/local/bin/
COPY gorunner /usr/local/bin/
COPY gorunner /gorunner
COPY cluster /kubernetes/cluster

ENV E2E_FOCUS="\[Conformance\]"
ENV E2E_SKIP=""
ENV E2E_PROVIDER="local"
ENV E2E_PARALLEL="1"
ENV E2E_VERBOSITY="4"
ENV RESULTS_DIR="/tmp/results"
ENV KUBECONFIG=""

ENTRYPOINT [ "/gorunner" ]
