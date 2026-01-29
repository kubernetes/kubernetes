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

package metrics

import (
	"testing"
)

func TestSchedulerMetricsRegistration(t *testing.T) {
	Register()

	// Some metrics (esp. with labels) don't show up in Gather() until at least one
	// labeled child is created. Create one child per metric family so we can verify they are registered and gatherable.
	Goroutines.WithLabelValues(Binding)
	PermitWaitDuration.WithLabelValues("success")
	PluginEvaluationTotal.WithLabelValues("testPlugin", Filter, "default")
	UnschedulableReason("testPlugin", "default")

	mfs, err := GetGather().Gather()
	if err != nil {
		t.Fatalf("gather metrics: %v", err)
	}

	want := map[string]bool{
		"scheduler_goroutines":                   false,
		"scheduler_permit_wait_duration_seconds": false,
		"scheduler_plugin_evaluation_total":      false,
		"scheduler_unschedulable_pods":           false,
	}

	for _, mf := range mfs {
		if _, ok := want[mf.GetName()]; ok {
			want[mf.GetName()] = true
		}
	}

	for name, found := range want {
		if !found {
			t.Fatalf("metric %q not registered", name)
		}
	}
}
