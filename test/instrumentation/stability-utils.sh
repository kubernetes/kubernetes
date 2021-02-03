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
set -o nounset
set -o pipefail


function kube::update::stablemetrics {
  BAZEL_OUT_DIR="$KUBE_ROOT/bazel-bin"
  BAZEL_GEN_DIR="$KUBE_ROOT/bazel-genfiles"
  METRICS_LIST_PATH="test/instrumentation/stable-metrics-list.yaml"

  bazel build //test/instrumentation:list_stable_metrics
  if [ -d "$BAZEL_OUT_DIR" ]; then
    cp "$BAZEL_OUT_DIR/$METRICS_LIST_PATH" "$KUBE_ROOT/test/instrumentation/testdata/stable-metrics-list.yaml"
  else
    # Handle bazel < 0.25
    # https://github.com/bazelbuild/bazel/issues/6761
    echo "$BAZEL_OUT_DIR not found trying $BAZEL_GEN_DIR"
    cp "$BAZEL_GEN_DIR/$METRICS_LIST_PATH" "$KUBE_ROOT/test/instrumentation/testdata/stable-metrics-list.yaml"
  fi
}


function kube::verify::metrics {
  BAZEL_OUT_DIR="$KUBE_ROOT/bazel-bin"
  BAZEL_GEN_DIR="$KUBE_ROOT/bazel-genfiles"
  METRICS_LIST_PATH="test/instrumentation/stable-metrics-list.yaml"

  if [ -z "${TEST_BINARY:-}" ]; then

    bazel build //test/instrumentation:list_stable_metrics --verbose_failures

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
    echo "PASS metrics stability check"
  else
    echo 'Diffs in stable metrics detected, please run "test/instrumentation/update-stable-metrics.sh"'
    exit 1
  fi
}
