/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// BenchmarkPerfScheduling is implemented in benchmark_test
// to ensure that scheduler_perf can be run from outside kubernetes.
package benchmark_test

import (
	"flag"
	"strings"
	"testing"

	benchmark "k8s.io/kubernetes/test/integration/scheduler_perf"
)

var perfSchedulingLabelFilter = flag.String("perf-scheduling-label-filter", "performance", "comma-separated list of labels which a testcase must have (no prefix or +) or must not have (-), used by BenchmarkPerfScheduling")

func BenchmarkPerfScheduling(b *testing.B) {
	if testing.Short() {
		*perfSchedulingLabelFilter += ",+short"
	}

	benchmark.RunBenchmarkPerfScheduling(b, nil, strings.Split(*perfSchedulingLabelFilter, ","))
}
