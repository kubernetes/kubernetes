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
	"testing"

	benchmark "k8s.io/kubernetes/test/integration/scheduler_perf"
)

func BenchmarkPerfScheduling(b *testing.B) {
	benchmark.RunBenchmarkPerfScheduling(b, nil)
}
