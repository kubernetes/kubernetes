#!/usr/bin/env bash
# Copyright 2019 The Kubernetes Authors.
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


set -o errexit
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
BAZEL_OUT_DIR="$KUBE_ROOT/bazel-bin"
BAZEL_GEN_DIR="$KUBE_ROOT/bazel-genfiles"
METRICS_LIST_PATH="test/instrumentation/stable-metrics-list.yaml"

# detect if run from bazel
if [ -z "${TEST_BINARY}" ]; then
  bazel build //test/instrumentation:list_stable_metrics

  if [ -d "$BAZEL_OUT_DIR" ]; then
    OUTPUT_FILE="$BAZEL_OUT_DIR/$METRICS_LIST_PATH"
  else
    # Handle bazel < 0.25
    # https://github.com/bazelbuild/bazel/issues/6761
    OUTPUT_FILE="$BAZEL_GEN_DIR/$METRICS_LIST_PATH"
  fi
else
  OUTPUT_FILE="$KUBE_ROOT/$METRICS_LIST_PATH"
fi

if diff -u "$KUBE_ROOT/test/instrumentation/testdata/stable-metrics-list.yaml" "$OUTPUT_FILE"; then
  echo PASS
  exit 0
fi
echo 'Diffs in stable metrics detected, please run "test/instrumentation/update-stable-metrics.sh"'
exit 1
