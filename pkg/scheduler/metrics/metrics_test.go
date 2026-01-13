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

			// NOTE: We use a structured assertion for this histogram to avoid fragile text-format comparisons for bucket exposition.
			if tt.metricName == "scheduler_permit_wait_duration_seconds" {
				histogramVec, err := testutil.GetHistogramVecFromGatherer(GetGather(), "scheduler_permit_wait_duration_seconds", map[string]string{"result": "success"})
				if err != nil {
					t.Fatalf("Failed to get histogram: %v", err)
				}
				if len(histogramVec) == 0 {
					t.Fatal("HistogramVec is empty")
				}
				if len(histogramVec) > 1 {
					t.Fatalf("Expected 1 histogram, got %d", len(histogramVec))
				}
				hist := histogramVec[0]

				// Assert sample count is 1
				if hist.GetSampleCount() != 1 {
					t.Errorf("Expected sample count 1, got %d", hist.GetSampleCount())
				}

				// Assert sample sum is 0.1
				if hist.GetSampleSum() != 0.1 {
					t.Errorf("Expected sample sum 0.1, got %f", hist.GetSampleSum())
				}

				// Assert cumulative bucket counts match expected pattern
				// For a value of 0.1, it should fall into the 0.128 bucket (first bucket >= 0.1)
				// So buckets < 0.128 should have count 0, buckets >= 0.128 should have count 1
				for _, bucket := range hist.Bucket {
					ub := bucket.GetUpperBound()
					cumulativeCount := bucket.GetCumulativeCount()
					if ub < 0.128 {
						if cumulativeCount != 0 {
							t.Errorf("Bucket with upper bound %f should have cumulative count 0, got %d", ub, cumulativeCount)
						}
					} else {
						if cumulativeCount != 1 {
							t.Errorf("Bucket with upper bound %f should have cumulative count 1, got %d", ub, cumulativeCount)
						}
					}
				}
			} else {
				// Use text comparison for other metrics
				exp := normalizeExpected(tt.want)
				if err := testutil.GatherAndCompare(GetGather(), strings.NewReader(exp), tt.metricName); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}
