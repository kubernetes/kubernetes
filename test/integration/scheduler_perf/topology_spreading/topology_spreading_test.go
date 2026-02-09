/*
Copyright 2025 The Kubernetes Authors.

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

package topologyspreading

import (
	"fmt"
	"os"
	"testing"

	_ "k8s.io/component-base/logs/json/register"
	perf "k8s.io/kubernetes/test/integration/scheduler_perf"
)

func TestMain(m *testing.M) {
	if err := perf.InitTests(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	m.Run()
}

func TestSchedulerPerf(t *testing.T) {
	perf.RunIntegrationPerfScheduling(t, "performance-config.yaml")
}

func BenchmarkPerfScheduling(b *testing.B) {
	perf.RunBenchmarkPerfScheduling(b, "performance-config.yaml", "topologyspreading", nil)
}
