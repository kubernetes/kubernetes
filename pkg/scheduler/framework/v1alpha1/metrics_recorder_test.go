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

package v1alpha1

import (
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

func TestObservePluginDurationAsync(t *testing.T) {
	tests := []struct {
		name               string
		extensionPoint     string
		pluginName         string
		status             *Status
		value              float64
		wantExtensionPoint string
		wantStatus         Code
		wantCount          uint64
		wantSum            float64
	}{
		{
			name:           "PreFilter - Success",
			extensionPoint: "preFilter",
			pluginName:     "preFileterPlugin",
			status: &Status{
				code: Success,
			},
			value:              float64(0.1),
			wantExtensionPoint: "preFilter",
			wantStatus:         Success,
			wantCount:          uint64(1),
			wantSum:            float64(0.1),
		},
		{
			name:           "PreFilter - Error",
			extensionPoint: "preFilter",
			pluginName:     "preFileterPlugin",
			status: &Status{
				code: Error,
			},
			value:              float64(1.1),
			wantExtensionPoint: "preFilter",
			wantStatus:         Error,
			wantCount:          uint64(1),
			wantSum:            float64(1.1),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics.Register()
			metrics.PluginExecutionDuration.Reset()

			recorder := newMetricsRecorder(100, time.Nanosecond)
			recorder.observePluginDurationAsync(tt.extensionPoint, tt.pluginName, tt.status, tt.value)
			close(recorder.stopCh)
			<-recorder.isStoppedCh
			recorder.flushMetrics()

			collectAndComparePluginDurationMetrics(t, tt.extensionPoint, tt.pluginName, tt.wantStatus, tt.wantCount, tt.wantSum)
		})
	}
}

func collectAndComparePluginDurationMetrics(t *testing.T, wantExtensionPoint, wantPlugin string, wantStatus Code, wantCount uint64, wantSum float64) {
	t.Helper()
	m := collectHistogramMetric(metrics.PluginExecutionDuration)
	if len(m.Label) != 3 {
		t.Fatalf("Unexpected number of label pairs, got: %v, want: 3", len(m.Label))
	}

	if *m.Label[0].Value != wantExtensionPoint {
		t.Errorf("Unexpected extension point label, got: %q, want %q", *m.Label[0].Value, wantExtensionPoint)
	}

	if *m.Label[1].Value != wantPlugin {
		t.Errorf("Unexpected plugin label, got: %q, want %q", *m.Label[1].Value, wantPlugin)
	}

	if *m.Label[2].Value != wantStatus.String() {
		t.Errorf("Unexpected status code label, got: %q, want %q", *m.Label[2].Value, wantStatus)
	}

	if *m.Histogram.SampleCount != wantCount {
		t.Errorf("Unexpected sample count, got: %q, want %q", *m.Histogram.SampleCount, wantCount)
	}

	if *m.Histogram.SampleSum != wantSum {
		t.Errorf("Unexpected sample sum, got: %f, want %f", *m.Histogram.SampleSum, wantSum)
	}
}
