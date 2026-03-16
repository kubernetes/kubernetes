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
	"context"
	"strings"
	"testing"
	"time"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestFlowControlBetaMetrics(t *testing.T) {
	Register()

	tests := []struct {
		desc    string
		metrics []string
		update  func()
		want    string
	}{
		{
			desc:    "rejected_requests_total increments on AddReject",
			metrics: []string{"apiserver_flowcontrol_rejected_requests_total"},
			update: func() {
				AddReject(context.Background(), "test-pl", "test-fs", "queue-full")
			},
			want: `
			# HELP apiserver_flowcontrol_rejected_requests_total [BETA] Number of requests rejected by API Priority and Fairness subsystem
			# TYPE apiserver_flowcontrol_rejected_requests_total counter
			apiserver_flowcontrol_rejected_requests_total{flow_schema="test-fs",priority_level="test-pl",reason="queue-full"} 1
			`,
		},
		{
			desc:    "dispatched_requests_total increments on AddDispatch",
			metrics: []string{"apiserver_flowcontrol_dispatched_requests_total"},
			update: func() {
				AddDispatch(context.Background(), "test-pl", "test-fs")
			},
			want: `
			# HELP apiserver_flowcontrol_dispatched_requests_total [BETA] Number of requests executed by API Priority and Fairness subsystem
			# TYPE apiserver_flowcontrol_dispatched_requests_total counter
			apiserver_flowcontrol_dispatched_requests_total{flow_schema="test-fs",priority_level="test-pl"} 1
			`,
		},
		{
			desc:    "current_executing_requests updates on AddRequestsExecuting",
			metrics: []string{"apiserver_flowcontrol_current_executing_requests"},
			update: func() {
				AddRequestsExecuting(context.Background(), "test-pl", "test-fs", 2)
			},
			want: `
			# HELP apiserver_flowcontrol_current_executing_requests [BETA] Number of requests in initial (for a WATCH) or any (for a non-WATCH) execution stage in the API Priority and Fairness subsystem
			# TYPE apiserver_flowcontrol_current_executing_requests gauge
			apiserver_flowcontrol_current_executing_requests{flow_schema="test-fs",priority_level="test-pl"} 2
			`,
		},
		{
			desc:    "current_executing_seats updates on AddSeatConcurrencyInUse",
			metrics: []string{"apiserver_flowcontrol_current_executing_seats"},
			update: func() {
				AddSeatConcurrencyInUse("test-pl", "test-fs", 3)
			},
			want: `
			# HELP apiserver_flowcontrol_current_executing_seats [BETA] Concurrency (number of seats) occupied by the currently executing (initial stage for a WATCH, any stage otherwise) requests in the API Priority and Fairness subsystem
			# TYPE apiserver_flowcontrol_current_executing_seats gauge
			apiserver_flowcontrol_current_executing_seats{flow_schema="test-fs",priority_level="test-pl"} 3
			`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			Reset()
			tc.update()
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.want), tc.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestFlowControlHistogramMetrics(t *testing.T) {
	Register()
	Reset()

	ObserveWaitingDuration(context.Background(), "test-pl", "test-fs", "true", 50*time.Millisecond)
	ObserveExecutionDuration(context.Background(), "test-pl", "test-fs", 100*time.Millisecond)
	ObserveWorkEstimatedSeats("test-pl", "test-fs", 4)

	gathered, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("Failed to gather metrics: %v", err)
	}

	wantCounts := map[string]uint64{
		"apiserver_flowcontrol_request_wait_duration_seconds": 1,
		"apiserver_flowcontrol_request_execution_seconds":     1,
		"apiserver_flowcontrol_work_estimated_seats":          1,
	}

	gotCounts := make(map[string]uint64)
	for _, mf := range gathered {
		if _, ok := wantCounts[mf.GetName()]; ok {
			for _, m := range mf.GetMetric() {
				gotCounts[mf.GetName()] += m.GetHistogram().GetSampleCount()
			}
		}
	}

	for name, want := range wantCounts {
		if got := gotCounts[name]; got != want {
			t.Errorf("Metric %s: expected %d samples, got %d", name, want, got)
		}
	}
}
