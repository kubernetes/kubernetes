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

package staleness

import (
	"math"
	"strings"
	"testing"
	"time"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

// approxDuration checks if two durations are within a small delta.
func approxDuration(a, b, delta time.Duration) bool {
	return math.Abs(float64(a-b)) < float64(delta)
}

func TestPodProber_QueryStaleness(t *testing.T) {
	fakeNow := time.Now()

	testCases := []struct {
		name          string
		queue         []RVTime
		podRV         string
		wantStaleness time.Duration
		wantErr       bool
	}{
		{
			name: "when pod RV is between two RVs in queue",
			queue: []RVTime{
				{rv: "100", now: fakeNow.Add(-10 * time.Second)}, // T-10s
				{rv: "200", now: fakeNow.Add(-5 * time.Second)},  // T-5s
			},
			podRV:         "150",
			wantStaleness: 5 * time.Second,
			wantErr:       false,
		},
		{
			name: "when pod RV is older than all RVs in queue",
			queue: []RVTime{
				{rv: "100", now: fakeNow.Add(-10 * time.Second)}, // T-10s
				{rv: "200", now: fakeNow.Add(-5 * time.Second)},  // T-5s
			},
			podRV:         "50",
			wantStaleness: 10 * time.Second,
			wantErr:       false,
		},
		{
			name: "when pod RV is newer than all RVs in queue",
			queue: []RVTime{
				{rv: "100", now: fakeNow.Add(-10 * time.Second)}, // T-10s
				{rv: "200", now: fakeNow.Add(-5 * time.Second)},  // T-5s
			},
			podRV:         "250",
			wantStaleness: 0,
			wantErr:       false,
		},
		{
			name: "when pod RV is invalid",
			queue: []RVTime{
				{rv: "100", now: fakeNow.Add(-10 * time.Second)},
			},
			podRV:   "not-a-number",
			wantErr: true,
		},
		{
			name:    "when queue is empty",
			queue:   []RVTime{},
			podRV:   "100",
			wantErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			prober := &PodProber{
				rvQueue: tc.queue,
			}

			staleness, err := prober.QueryStaleness(tc.podRV)

			if tc.wantErr {
				if err == nil {
					t.Fatalf("Expected an error, but got nil")
				}
				return
			}

			if !tc.wantErr && err != nil {
				t.Fatalf("Did not expect an error, but got: %v", err)
			}

			if !approxDuration(staleness, tc.wantStaleness, 50*time.Millisecond) {
				t.Errorf("Expected staleness ~%v, but got %v", tc.wantStaleness, staleness)
			}
		})
	}
}

// TestStalenessMetrics validates the Prometheus output of the watch delay
// gauge and histogram. It simulates the metric updates that would
// happen inside the prober's metricsLoop.
func TestStalenessMetrics(t *testing.T) {
	// Reset the registry to ensure a clean state, since metrics are registered globally.
	legacyregistry.Reset()

	testedMetrics := []string{
		"controller_manager_watch_delay_seconds",
		"controller_manager_current_watch_delay_seconds",
	}

	testCases := []struct {
		name    string
		observe float64
		want    string
	}{
		{
			name:    "observe 3.5 seconds",
			observe: 3.5,
			// Buckets are calculated from the package variables:
			// minSample = 2.5s, factor = 1.6153..., count = 10
			// Buckets: 2.5, 4.0383..., 6.5234..., etc.
			// 3.5 fits in the 4.0383... bucket.
			want: `
# HELP controller_manager_current_watch_delay_seconds [ALPHA] Last observed watch delay seconds, for now calculated only for pods
# TYPE controller_manager_current_watch_delay_seconds gauge
controller_manager_current_watch_delay_seconds{group="",resource="pods"} 3.5
# HELP controller_manager_watch_delay_seconds [ALPHA] Watch delay seconds, for now calculated only for pods
# TYPE controller_manager_watch_delay_seconds histogram
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="2.5"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="4.255593600390397"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="7.244030756673482"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="12.33106037165235"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="20.99039264145255"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="35.73063223785886"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="60.822019955734"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="103.53351955457545"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="176.23863329693813"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="299.99999999999994"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="+Inf"} 1
controller_manager_watch_delay_seconds_sum{group="",resource="pods"} 3.5
controller_manager_watch_delay_seconds_count{group="",resource="pods"} 1
`,
		},
		{
			name:    "observe 0.5 seconds",
			observe: 0.5,
			// 0.5 fits in the 2.5 bucket.
			want: `
# HELP controller_manager_current_watch_delay_seconds [ALPHA] Last observed watch delay seconds, for now calculated only for pods
# TYPE controller_manager_current_watch_delay_seconds gauge
controller_manager_current_watch_delay_seconds{group="",resource="pods"} 0.5
# HELP controller_manager_watch_delay_seconds [ALPHA] Watch delay seconds, for now calculated only for pods
# TYPE controller_manager_watch_delay_seconds histogram
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="2.5"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="4.255593600390397"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="7.244030756673482"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="12.33106037165235"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="20.99039264145255"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="35.73063223785886"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="60.822019955734"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="103.53351955457545"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="176.23863329693813"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="299.99999999999994"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="+Inf"} 1
controller_manager_watch_delay_seconds_sum{group="",resource="pods"} 0.5
controller_manager_watch_delay_seconds_count{group="",resource="pods"} 1
`,
		},
		{
			name:    "observe 500 seconds",
			observe: 500.0,
			// 500.0 fits in the +Inf bucket.
			want: `
# HELP controller_manager_current_watch_delay_seconds [ALPHA] Last observed watch delay seconds, for now calculated only for pods
# TYPE controller_manager_current_watch_delay_seconds gauge
controller_manager_current_watch_delay_seconds{group="",resource="pods"} 500
# HELP controller_manager_watch_delay_seconds [ALPHA] Watch delay seconds, for now calculated only for pods
# TYPE controller_manager_watch_delay_seconds histogram
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="2.5"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="4.255593600390397"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="7.244030756673482"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="12.33106037165235"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="20.99039264145255"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="35.73063223785886"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="60.822019955734"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="103.53351955457545"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="176.23863329693813"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="299.99999999999994"} 0
controller_manager_watch_delay_seconds_sum{group="",resource="pods"} 500
controller_manager_watch_delay_seconds_count{group="",resource="pods"} 1
`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			watchDelayGauge.Reset()
			watchDelayHistogram.Reset()

			watchDelayGauge.WithLabelValues("", "pods").Set(tc.observe)
			watchDelayHistogram.WithLabelValues("", "pods").Observe(tc.observe)

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.want), testedMetrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
