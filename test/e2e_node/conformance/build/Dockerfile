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

FROM BASEIMAGE
# If we're building for another architecture than amd64, the CROSS_BUILD_ placeholder is removed so e.g. CROSS_BUILD_COPY turns into COPY
# If we're building normally, for amd64, CROSS_BUILD lines are removed
CROSS_BUILD_COPY qemu-ARCH-static /usr/bin/

COPY ginkgo /usr/local/bin/
COPY e2e_node.test /opt/

# The following environment variables can be override when starting the container.
# FOCUS is regex matching test to run. By default run all conformance test.
ENV FOCUS="\[Conformance\]"
# SKIP is regex matching test to skip. By default empty.
ENV SKIP=""
# PARALLELISM is the number of processes the test will run in parallel.
ENV PARALLELISM=8
# REPORT_PATH is the path in the container to save test result and logs.
ENV REPORT_PATH="/opt/result"
# MANIFEST_PATH is the kubelet manifest path in the container.
ENV MANIFEST_PATH="/opt/config"

ENTRYPOINT ginkgo --focus="$FOCUS" --skip="$SKIP" --nodes=$PARALLELISM /opt/e2e_node.test -- --containerize=true --prepull-images=false --manifest-path=$MANIFEST_PATH --report-dir=$REPORT_PATH
