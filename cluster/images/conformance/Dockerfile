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

FROM BASEIMAGE

COPY ginkgo /usr/local/bin/
COPY e2e.test /usr/local/bin/
COPY kubectl /usr/local/bin/
COPY run_e2e.sh /run_e2e.sh
COPY gorunner /gorunner
COPY cluster /kubernetes/cluster
WORKDIR /usr/local/bin

ENV E2E_FOCUS="\[Conformance\]"
ENV E2E_SKIP=""
ENV E2E_PROVIDER="local"
ENV E2E_PARALLEL="1"
ENV E2E_VERBOSITY="4"
ENV RESULTS_DIR="/tmp/results"
ENV KUBECONFIG=""

CMD [ "/bin/bash", "-c", "/run_e2e.sh" ]
