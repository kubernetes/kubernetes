#!/usr/bin/env bash

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

# This script validates that only a restricted set of packages are importing
# github.com/prometheus/*

# NOTE: this is not the same as verify-imports which can only verify
# that within a particular package the imports made are allowed.
#
# This is also not the same thing as verify-import-boss, which is pretty
# powerful for specifying restricted imports but does not scale to checking
# the entire source tree well and is only enabled for specific packages.
#
# See: https://github.com/kubernetes/kubernetes/issues/99876

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

# See: https://github.com/kubernetes/kubernetes/issues/89267
allowed_prometheus_importers=(
  ./cluster/images/etcd-version-monitor/etcd-version-monitor.go
  ./staging/src/k8s.io/component-base/metrics/prometheusextension/timing_histogram.go
  ./staging/src/k8s.io/component-base/metrics/prometheusextension/timing_histogram_test.go
  ./staging/src/k8s.io/component-base/metrics/prometheusextension/timing_histogram_vec.go
  ./staging/src/k8s.io/component-base/metrics/prometheusextension/weighted_histogram.go
  ./staging/src/k8s.io/component-base/metrics/prometheusextension/weighted_histogram_test.go
  ./staging/src/k8s.io/component-base/metrics/prometheusextension/weighted_histogram_vec.go
  ./staging/src/k8s.io/component-base/metrics/collector.go
  ./staging/src/k8s.io/component-base/metrics/collector_test.go
  ./staging/src/k8s.io/component-base/metrics/counter.go
  ./staging/src/k8s.io/component-base/metrics/counter_test.go
  ./staging/src/k8s.io/component-base/metrics/desc.go
  ./staging/src/k8s.io/component-base/metrics/gauge.go
  ./staging/src/k8s.io/component-base/metrics/gauge_test.go
  ./staging/src/k8s.io/component-base/metrics/histogram.go
  ./staging/src/k8s.io/component-base/metrics/histogram_test.go
  ./staging/src/k8s.io/component-base/metrics/http.go
  ./staging/src/k8s.io/component-base/metrics/labels.go
  ./staging/src/k8s.io/component-base/metrics/legacyregistry/registry.go
  ./staging/src/k8s.io/component-base/metrics/metric.go
  ./staging/src/k8s.io/component-base/metrics/opts.go
  ./staging/src/k8s.io/component-base/metrics/processstarttime_others.go
  ./staging/src/k8s.io/component-base/metrics/registry.go
  ./staging/src/k8s.io/component-base/metrics/registry_test.go
  ./staging/src/k8s.io/component-base/metrics/summary.go
  ./staging/src/k8s.io/component-base/metrics/testutil/metrics.go
  ./staging/src/k8s.io/component-base/metrics/testutil/metrics_test.go
  ./staging/src/k8s.io/component-base/metrics/testutil/promlint.go
  ./staging/src/k8s.io/component-base/metrics/testutil/testutil.go
  ./staging/src/k8s.io/component-base/metrics/timing_histogram_test.go
  ./staging/src/k8s.io/component-base/metrics/value.go
  ./staging/src/k8s.io/component-base/metrics/wrappers.go
  ./test/e2e/apimachinery/flowcontrol.go
  ./test/e2e/node/pods.go
  ./test/e2e_node/resource_metrics_test.go
  ./test/instrumentation/main_test.go
  ./test/integration/apiserver/flowcontrol/concurrency_test.go
  ./test/integration/apiserver/flowcontrol/concurrency_util_test.go
  ./test/integration/metrics/metrics_test.go
)

# Go imports always involve a double quoted string of the package path
# https://golang.org/ref/spec#Import_declarations
#
# If you *really* need a string literal that looks like "github.com/prometheus/.*"
# somewhere else that actually isn't an import, you can use backticks / a raw
# string literal instead (which cannot be used in imports, only double quotes).
#
# NOTE: we previously had an implementation that checked for an actual import
# as a post-processing step on the matching files, which is cheap enough and
# accurate, except that it's difficult to guarantee we check for all supported
# GOOS, GOARCH, and other build tags, and we want to prevent all imports.
# So we dropped this, in favor of only the grep call.
# See: https://github.com/kubernetes/kubernetes/pull/100552
really_failing_files=()
all_failing_files=()
while IFS='' read -r filepath; do
  # convert from file to package, and only insert unique results
  # we want to minimize the amount of `go list` calls we need to make
  if ! kube::util::array_contains "$filepath" "${allowed_prometheus_importers[@]}"; then
    # record a failure if not
    really_failing_files+=("$filepath")
  fi
  all_failing_files+=("$filepath")
done < <(cd "${KUBE_ROOT}" && grep \
  --exclude-dir={_output,vendor} \
  --include='*.go' \
  -R . \
  -l \
  -Ee '"github.com/prometheus/.*"' \
| LC_ALL=C sort -u)

# check for any files we're allowing to fail that are no longer failing, so we
# can enforce that the list shrinks
allowed_but_not_failing=()
for allowed_file in "${allowed_prometheus_importers[@]}"; do
  if ! kube::util::array_contains "$allowed_file" "${all_failing_files[@]}"; then
    allowed_but_not_failing+=("$allowed_file")
  fi
done

# we will exit with this at the end of the script depending on the checks below
exit_code=0

# check for files we've allow-listed that no longer need to be
if [ -n "${allowed_but_not_failing[*]}" ]; then
  {
    echo "ERROR: Some files allow-listed to import prometheus are no longer failing and should be removed."
    echo "Please remove these files from allowed_prometheus_importers in hack/verify-prometheus-imports.sh"
    echo ""
    echo "Non-failing but allow-listed files:"
    for non_failing_file in "${allowed_but_not_failing[@]}"; do
      echo "  ${non_failing_file}"
    done
  } >&2
  exit_code=1
fi
# check for files that fail but are not allow-listed
if [ -n "${really_failing_files[*]}" ]; then
  {
    echo "ERROR: Some files are importing packages under github.com/prometheus/* but are not allow-listed to do so."
    echo ""
    echo "See: https://github.com/kubernetes/kubernetes/issues/89267"
    echo ""
    echo "Failing files:"
    for failing_file in "${really_failing_files[@]}"; do
      echo "  ${failing_file}"
    done
  } >&2
  exit_code=2
fi

exit "$exit_code"
