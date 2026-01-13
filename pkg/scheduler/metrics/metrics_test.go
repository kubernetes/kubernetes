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
	"strings"
	"testing"

	"k8s.io/component-base/metrics/testutil"
)

// normalizeExpected trims the overall string and removes leading spaces/tabs from each line
func normalizeExpected(s string) string {
	s = strings.TrimSpace(s)
	lines := strings.Split(s, "\n")
	normalized := make([]string, 0, len(lines))
	for _, line := range lines {
		normalized = append(normalized, strings.TrimLeft(line, " \t"))
	}
	result := strings.Join(normalized, "\n")
	// Ensure result ends with a newline for Prometheus text format
	if result != "" && !strings.HasSuffix(result, "\n") {
		result += "\n"
	}
	return result
}

func TestSchedulerMetricsRegistrationAndEmission(t *testing.T) {
	// Register metrics
	Register()

	tests := []struct {
		name       string
		metricName string
		update     func()
		want       string
	}{
		{
			name:       "scheduler_goroutines",
			metricName: "scheduler_goroutines",
			update: func() {
				Goroutines.Reset()
				Goroutines.WithLabelValues(Binding).Set(1.0)
			},
			want: `
				# HELP scheduler_goroutines [BETA] Number of running goroutines split by the work they do such as binding.
				# TYPE scheduler_goroutines gauge
				scheduler_goroutines{operation="binding"} 1
			`,
		},
		{
			name:       "scheduler_permit_wait_duration_seconds",
			metricName: "scheduler_permit_wait_duration_seconds",
			update: func() {
				PermitWaitDuration.Reset()
				PermitWaitDuration.WithLabelValues("success").Observe(0.1)
			},
			want: `
				# HELP scheduler_permit_wait_duration_seconds [BETA] Duration of waiting on permit.
				# TYPE scheduler_permit_wait_duration_seconds histogram
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="0.001"} 0
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="0.002"} 0
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="0.004"} 0
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="0.008"} 0
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="0.016"} 0
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="0.032"} 0
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="0.064"} 0
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="0.128"} 1
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="0.256"} 1
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="0.512"} 1
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="1.024"} 1
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="2.048"} 1
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="4.096"} 1
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="8.192"} 1
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="16.384"} 1
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="32.768"} 1
				scheduler_permit_wait_duration_seconds_bucket{result="success",le="+Inf"} 1
				scheduler_permit_wait_duration_seconds_sum{result="success"} 0.1
				scheduler_permit_wait_duration_seconds_count{result="success"} 1
			`,
		},
		{
			name:       "scheduler_plugin_evaluation_total",
			metricName: "scheduler_plugin_evaluation_total",
			update: func() {
				PluginEvaluationTotal.Reset()
				PluginEvaluationTotal.WithLabelValues("testPlugin", Filter, "default").Inc()
			},
			want: `
				# HELP scheduler_plugin_evaluation_total [BETA] Number of attempts to schedule pods by each plugin and the extension point (available only in PreFilter, Filter, PreScore, and Score).
				# TYPE scheduler_plugin_evaluation_total counter
				scheduler_plugin_evaluation_total{extension_point="Filter",plugin="testPlugin",profile="default"} 1
			`,
		},
		{
			name:       "scheduler_unschedulable_pods",
			metricName: "scheduler_unschedulable_pods",
			update: func() {
				unschedulableReasons.Reset()
				UnschedulableReason("testPlugin", "default").Set(2.0)
			},
			want: `
				# HELP scheduler_unschedulable_pods [BETA] The number of unschedulable pods broken down by plugin name. A pod will increment the gauge for all plugins that caused it to not schedule and so this metric have meaning only when broken down by plugin.
				# TYPE scheduler_unschedulable_pods gauge
				scheduler_unschedulable_pods{plugin="testPlugin",profile="default"} 2
			`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.update()

			exp := normalizeExpected(tt.want)
			if err := testutil.GatherAndCompare(GetGather(), strings.NewReader(exp), tt.metricName); err != nil {
				t.Fatal(err)
			}
		})
	}
}
