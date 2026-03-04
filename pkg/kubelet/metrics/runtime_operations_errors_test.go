/*
Copyright 2024 The Kubernetes Authors.

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

package metrics

import (
	"testing"
)

// TestRuntimeOperationsErrorsInitialization verifies that the
// kubelet_runtime_operations_errors_total metric is initialized to 0
// for all operation types at startup, ensuring the metric is present
// even when no errors have occurred.
func TestRuntimeOperationsErrorsInitialization(t *testing.T) {
	Register()
	defer clearRuntimeMetrics()

	gatherer := GetGather()
	metricFamilies, err := gatherer.Gather()
	if err != nil {
		t.Fatalf("Failed to gather metrics: %v", err)
	}

	// Find the runtime_operations_errors_total metric
	var found bool
	var metricCount int
	for _, mf := range metricFamilies {
		if mf.GetName() == "kubelet_runtime_operations_errors_total" {
			found = true
			metricCount = len(mf.GetMetric())

			// Verify all metrics are initialized to 0
			for _, m := range mf.GetMetric() {
				if m.GetCounter().GetValue() != 0 {
					t.Errorf("Expected counter value 0, got %v for labels %v",
						m.GetCounter().GetValue(), m.GetLabel())
				}
			}
			break
		}
	}

	if !found {
		t.Fatal("kubelet_runtime_operations_errors_total metric not found in /metrics endpoint")
	}

	// Verify we have metrics for all expected operation types (35 total)
	expectedOperationCount := 35
	if metricCount != expectedOperationCount {
		t.Errorf("Expected %d operation types, got %d", expectedOperationCount, metricCount)
	}
}

func clearRuntimeMetrics() {
	RuntimeOperations.Reset()
	RuntimeOperationsDuration.Reset()
	RuntimeOperationsErrors.Reset()
}
