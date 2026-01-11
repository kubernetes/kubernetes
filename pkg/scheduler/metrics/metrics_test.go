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

	dto "github.com/prometheus/client_model/go"
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
			name:       "scheduler_plugin_execution_duration_seconds",
			metricName: "scheduler_plugin_execution_duration_seconds",
			update: func() {
				PluginExecutionDuration.Reset()
				PluginExecutionDuration.WithLabelValues("testPlugin", Filter, "Success").Observe(0.00001)
			},
			want: `
				# HELP scheduler_plugin_execution_duration_seconds [BETA] Duration for running a plugin at a specific extension point.
				# TYPE scheduler_plugin_execution_duration_seconds histogram
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="1e-05"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="1.5000000000000002e-05"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="2.2500000000000005e-05"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="3.375000000000001e-05"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="5.062500000000001e-05"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="7.593750000000002e-05"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="0.00011390625000000003"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="0.00017085937500000006"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="0.0002562890625000001"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="0.00038443359375000017"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="0.0005766503906250003"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="0.0008649755859375004"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="0.0012974633789062506"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="0.0019461950683593758"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="0.0029192926025390638"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="0.004378938903808595"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="0.006568408355712893"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="0.009852612533569338"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="0.014778918800354007"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="0.02216837820053101"} 1
				scheduler_plugin_execution_duration_seconds_bucket{extension_point="Filter",plugin="testPlugin",status="Success",le="+Inf"} 1
				scheduler_plugin_execution_duration_seconds_sum{extension_point="Filter",plugin="testPlugin",status="Success"} 1e-05
				scheduler_plugin_execution_duration_seconds_count{extension_point="Filter",plugin="testPlugin",status="Success"} 1
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
				mfs, err := GetGather().Gather()
				if err != nil {
					t.Fatalf("Failed to gather metrics: %v", err)
				}

				var mf *dto.MetricFamily
				for _, family := range mfs {
					if family.GetName() == "scheduler_permit_wait_duration_seconds" {
						mf = family
						break
					}
				}
				if mf == nil {
					t.Fatal("MetricFamily scheduler_permit_wait_duration_seconds not found")
				}

				// Find the metric with label result="success"
				var metric *dto.Metric
				for _, m := range mf.GetMetric() {
					labels := m.GetLabel()
					hasResultSuccess := false
					for _, label := range labels {
						if label.GetName() == "result" && label.GetValue() == "success" {
							hasResultSuccess = true
							break
						}
					}
					if hasResultSuccess {
						metric = m
						break
					}
				}
				if metric == nil {
					t.Fatal("Metric with result=\"success\" not found")
				}

				hist := metric.GetHistogram()
				if hist == nil {
					t.Fatal("Metric is not a histogram")
				}

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
				buckets := hist.GetBucket()
				for _, bucket := range buckets {
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

func TestSchedulingAlgorithmLatency(t *testing.T) {
	// Register metrics
	Register()

	// SchedulingAlgorithmLatency is a Histogram (not HistogramVec), so it doesn't have Reset()
	// We observe a value to ensure the metric is registered and can be emitted
	SchedulingAlgorithmLatency.Observe(0.1)

	// Verify the metric is registered and has the correct stability level by checking help text
	mfs, err := GetGather().Gather()
	if err != nil {
		t.Fatalf("Failed to gather metrics: %v", err)
	}

	var found bool
	for _, mf := range mfs {
		if mf.GetName() == "scheduler_scheduling_algorithm_duration_seconds" {
			found = true
			help := mf.GetHelp()
			if !strings.Contains(help, "[BETA]") {
				t.Errorf("Expected help text to contain [BETA], got: %s", help)
			}
			if !strings.Contains(help, "Scheduling algorithm latency in seconds") {
				t.Errorf("Expected help text to contain description, got: %s", help)
			}
			break
		}
	}

	if !found {
		t.Error("scheduler_scheduling_algorithm_duration_seconds metric not found")
	}
}
