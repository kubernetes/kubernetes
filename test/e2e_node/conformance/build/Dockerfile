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

COPY ginkgo /usr/local/bin/
COPY e2e_node.test /usr/local/bin

# The following environment variables can be override when starting the container.
# FOCUS is regex matching test to run. By default run all conformance test.
# SKIP is regex matching test to skip. By default skip flaky and serial test.
# PARALLELISM is the number of processes the test will run in parallel.
# REPORT_PATH is the path in the container to save test result and logs.
# FLAKE_ATTEMPTS is the time to retry when there is a test failure. By default 2.
# TEST_ARGS is the test arguments passed into the test.
ENV FOCUS="\[Conformance\]" \
	   SKIP="\[Flaky\]|\[Serial\]" \
	   PARALLELISM=8 \
	   REPORT_PATH="/var/result" \
	   FLAKE_ATTEMPTS=2 \
	   TEST_ARGS=""

ENTRYPOINT ginkgo --focus="$FOCUS" \
	--skip="$SKIP" \
	--nodes=$PARALLELISM \
	--flakeAttempts=$FLAKE_ATTEMPTS \
	/usr/local/bin/e2e_node.test \
	-- --conformance=true \
	--prepull-images=false \
	--report-dir="$REPORT_PATH" \
	$TEST_ARGS
