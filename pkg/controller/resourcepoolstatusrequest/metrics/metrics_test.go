/*
Copyright The Kubernetes Authors.

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

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

func TestRequestsProcessed(t *testing.T) {
	registry := metrics.NewKubeRegistry()
	defer registry.Reset()
	registry.MustRegister(RequestsProcessed)

	RequestsProcessed.WithLabelValues("driver-a.example.com").Inc()
	RequestsProcessed.WithLabelValues("driver-a.example.com").Inc()
	RequestsProcessed.WithLabelValues("driver-b.example.com").Inc()

	want := `# HELP resourcepoolstatusrequest_controller_requests_processed_total [ALPHA] Total number of ResourcePoolStatusRequests processed
# TYPE resourcepoolstatusrequest_controller_requests_processed_total counter
resourcepoolstatusrequest_controller_requests_processed_total{driver_name="driver-a.example.com"} 2
resourcepoolstatusrequest_controller_requests_processed_total{driver_name="driver-b.example.com"} 1
`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(want), "resourcepoolstatusrequest_controller_requests_processed_total"); err != nil {
		t.Errorf("unexpected metric output: %v", err)
	}
}

func TestRequestProcessingErrors(t *testing.T) {
	registry := metrics.NewKubeRegistry()
	defer registry.Reset()
	registry.MustRegister(RequestProcessingErrors)

	RequestProcessingErrors.WithLabelValues("driver-a.example.com").Inc()

	want := `# HELP resourcepoolstatusrequest_controller_request_processing_errors_total [ALPHA] Total number of errors encountered while processing ResourcePoolStatusRequests
# TYPE resourcepoolstatusrequest_controller_request_processing_errors_total counter
resourcepoolstatusrequest_controller_request_processing_errors_total{driver_name="driver-a.example.com"} 1
`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(want), "resourcepoolstatusrequest_controller_request_processing_errors_total"); err != nil {
		t.Errorf("unexpected metric output: %v", err)
	}
}

func TestRequestProcessingDuration(t *testing.T) {
	registry := metrics.NewKubeRegistry()
	defer registry.Reset()
	registry.MustRegister(RequestProcessingDuration)

	RequestProcessingDuration.WithLabelValues("driver-a.example.com").Observe(0.5)
	RequestProcessingDuration.WithLabelValues("driver-b.example.com").Observe(1.0)

	// Use gatherWithoutDurations to strip timing-dependent histogram fields
	// (SampleSum and Bucket), verifying only _count lines.
	want := `# HELP resourcepoolstatusrequest_controller_request_processing_duration_seconds [ALPHA] Time taken to process a ResourcePoolStatusRequest
# TYPE resourcepoolstatusrequest_controller_request_processing_duration_seconds histogram
resourcepoolstatusrequest_controller_request_processing_duration_seconds_count{driver_name="driver-a.example.com"} 1
resourcepoolstatusrequest_controller_request_processing_duration_seconds_count{driver_name="driver-b.example.com"} 1
`
	if err := testutil.GatherAndCompare(gatherWithoutDurations(registry), strings.NewReader(want), "resourcepoolstatusrequest_controller_request_processing_duration_seconds"); err != nil {
		t.Errorf("unexpected metric output: %v", err)
	}
}

// gatherWithoutDurations wraps a registry and strips timing-dependent fields
// (SampleSum and Bucket) from histograms so that tests only verify _count.
func gatherWithoutDurations(registry metrics.KubeRegistry) testutil.GathererFunc {
	return func() ([]*testutil.MetricFamily, error) {
		got, err := registry.Gather()
		for _, mf := range got {
			for _, m := range mf.Metric {
				if m.Histogram == nil {
					continue
				}
				m.Histogram.SampleSum = nil
				m.Histogram.Bucket = nil
			}
		}
		return got, err
	}
}

func TestRegister(t *testing.T) {
	// Verify Register does not panic when called multiple times.
	Register()
	Register()
}
