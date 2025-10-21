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

package staleness

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	testingclock "k8s.io/utils/clock/testing"

	"go.uber.org/goleak"
)

func TestProbeInfo_QueryStaleness(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Unix(100, 0))
	fakeNow := fakeClock.Now()

	testCases := []struct {
		name             string
		latestRV         string
		latestProbedTime time.Time
		lastInformerTime time.Time
		lastStaleness    time.Duration
		podRV            string
		lastInformerRV   string
		wantStaleness    time.Duration
		wantChanged      bool
		wantErr          bool
	}{
		{
			name:             "when pod RV is older than latest RV",
			latestRV:         "100",
			latestProbedTime: fakeNow.Add(-10 * time.Second), // T-10s
			lastInformerTime: fakeNow,
			podRV:            "50",
			wantStaleness:    10 * time.Second,
			wantChanged:      true,
			wantErr:          false,
		},
		{
			name:             "when pod RV is older than latest RV and staleness recorded",
			latestRV:         "100",
			latestProbedTime: fakeNow.Add(-10 * time.Second), // T-10s
			lastStaleness:    15 * time.Second,
			podRV:            "50",
			wantStaleness:    15 * time.Second,
			wantChanged:      true,
			wantErr:          false,
		},
		{
			name:             "when pod RV is same as latest RV and no staleness recorded",
			latestRV:         "100",
			latestProbedTime: fakeNow.Add(-10 * time.Second), // T-10s
			lastInformerTime: fakeNow,
			podRV:            "100",
			wantStaleness:    10 * time.Second,
			wantChanged:      true,
			wantErr:          false,
		},
		{
			name:             "when pod RV is newer than latest RV",
			latestRV:         "100",
			latestProbedTime: fakeNow.Add(-10 * time.Second), // T-10s
			lastInformerTime: fakeNow,
			podRV:            "150",
			wantStaleness:    10 * time.Second,
			wantChanged:      true,
			wantErr:          false,
		},
		{
			name:             "when pod RV is invalid",
			latestRV:         "100",
			latestProbedTime: fakeNow.Add(-10 * time.Second),
			podRV:            "not-a-number",
			wantErr:          true,
		},
		{
			name:             "when pod RV is same as latest RV and staleness recorded",
			latestRV:         "100",
			latestProbedTime: fakeNow.Add(-10 * time.Second), // T-10s
			lastInformerTime: fakeNow,
			lastStaleness:    5 * time.Second,
			podRV:            "100",
			wantStaleness:    10 * time.Second,
			wantChanged:      true,
			wantErr:          false,
		},
		{
			name:     "when no probe has happened",
			latestRV: "",
			podRV:    "100",
			wantErr:  true,
		},
		{
			name:             "when pod RV hasn't changed since last informer update",
			latestRV:         "100",
			latestProbedTime: fakeNow.Add(-10 * time.Second), // T-10s
			podRV:            "50",
			lastInformerRV:   "50",
			wantChanged:      false,
			wantErr:          false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			probeMutex := &sync.Mutex{}
			probeInfo := &ProbeInfo{
				mux:              probeMutex,
				cond:             sync.NewCond(probeMutex),
				clock:            fakeClock,
				latestRV:         tc.latestRV,
				latestProbedTime: tc.latestProbedTime,
				currentStaleness: tc.lastStaleness,
				lastInformerRV:   tc.lastInformerRV,
				lastInformerTime: tc.lastInformerTime,
			}

			changed, err := probeInfo.UpdateStaleness(tc.podRV)

			if tc.wantErr {
				if err == nil {
					t.Fatalf("Expected an error, but got nil")
				}
				return
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("Did not expect an error, but got: %v", err)
			}
			if changed != tc.wantChanged {
				t.Errorf("Expected changed %v, but got %v", tc.wantChanged, changed)
			}
			if !changed {
				return
			}
			staleness := probeInfo.currentStaleness

			if staleness != tc.wantStaleness {
				t.Errorf("Expected staleness %v, but got %v", tc.wantStaleness, staleness)
			}
		})
	}
}

func TestProbeInfo_QueryStaleness_Signal(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Unix(100, 0))
	fakeNow := fakeClock.Now()

	testCases := []struct {
		name           string
		latestRV       string
		podRV          string
		lastInformerRV string
		expectSignal   bool
	}{
		{
			name:         "when pod RV is older than latest RV (stale)",
			latestRV:     "100",
			podRV:        "50",
			expectSignal: false,
		},
		{
			name:         "when pod RV is same as latest RV (caught up)",
			latestRV:     "100",
			podRV:        "100",
			expectSignal: true,
		},
		{
			name:         "when pod RV is newer than latest RV (caught up)",
			latestRV:     "100",
			podRV:        "150",
			expectSignal: true,
		},
		{
			name:           "when pod RV hasn't changed since last informer update",
			latestRV:       "100",
			podRV:          "150",
			lastInformerRV: "150",
			expectSignal:   false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			probeMutex := &sync.Mutex{}
			probeInfo := &ProbeInfo{
				mux:              probeMutex,
				cond:             sync.NewCond(probeMutex),
				clock:            fakeClock,
				latestRV:         tc.latestRV,
				latestProbedTime: fakeNow,
				lastInformerRV:   tc.lastInformerRV,
			}

			signalCh := make(chan struct{})
			startedWaitingCh := make(chan struct{})
			go func() {
				probeMutex.Lock()
				close(startedWaitingCh)
				probeInfo.cond.Wait()
				probeMutex.Unlock()
				close(signalCh)
			}()

			<-startedWaitingCh
			time.Sleep(50 * time.Millisecond) // Give the goroutine time to enter cond.Wait()

			_, err := probeInfo.UpdateStaleness(tc.podRV)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tc.expectSignal {
				select {
				case <-signalCh:
				case <-time.After(5 * time.Second):
					t.Fatalf("expected signal to cvar, but none was received after timeout")
				}
			} else {
				select {
				case <-signalCh:
					t.Fatalf("expected no signal to cvar, but a signal was received")
				case <-time.After(100 * time.Millisecond):
				}
			}

			// Clean up the goroutine by making sure it unblocks
			probeInfo.cond.Broadcast()

			<-signalCh
		})
	}
}

func TestControllerShutdown(t *testing.T) {
	defer goleak.VerifyNone(t)
	ctx, cancel := context.WithCancel(context.Background())

	fakeClient := fake.NewSimpleClientset()
	fakeStore := cache.NewStore(cache.MetaNamespaceKeyFunc)

	podProber := NewPodProber(ctx, fakeClient, fakeStore)

	runStopped := make(chan struct{})
	go func() {
		podProber.Run(ctx)
		close(runStopped)
	}()

	cancel()

	select {
	case <-runStopped:
	case <-time.After(5 * time.Second):
		t.Fatalf("Run did not conclude after context cancellation")
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
			// minSample = 1s, factor = 2, count = 10
			// Buckets: 1, 2, 4, 8, etc.
			// 3.5 fits in the 4 bucket.
			want: `
# HELP controller_manager_current_watch_delay_seconds [ALPHA] Last observed watch delay seconds, for now calculated only for pods
# TYPE controller_manager_current_watch_delay_seconds gauge
controller_manager_current_watch_delay_seconds{group="",resource="pods"} 3.5
# HELP controller_manager_watch_delay_seconds [ALPHA] Watch delay seconds, for now calculated only for pods
# TYPE controller_manager_watch_delay_seconds histogram
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="1"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="2"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="4"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="8"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="16"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="32"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="64"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="128"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="256"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="512"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="+Inf"} 1
controller_manager_watch_delay_seconds_sum{group="",resource="pods"} 3.5
controller_manager_watch_delay_seconds_count{group="",resource="pods"} 1
`,
		},
		{
			name:    "observe 0.5 seconds",
			observe: 0.5,
			// 0.5 fits in the 1 bucket.
			want: `
# HELP controller_manager_current_watch_delay_seconds [ALPHA] Last observed watch delay seconds, for now calculated only for pods
# TYPE controller_manager_current_watch_delay_seconds gauge
controller_manager_current_watch_delay_seconds{group="",resource="pods"} 0.5
# HELP controller_manager_watch_delay_seconds [ALPHA] Watch delay seconds, for now calculated only for pods
# TYPE controller_manager_watch_delay_seconds histogram
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="1"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="2"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="4"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="8"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="16"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="32"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="64"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="128"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="256"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="512"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="+Inf"} 1
controller_manager_watch_delay_seconds_sum{group="",resource="pods"} 0.5
controller_manager_watch_delay_seconds_count{group="",resource="pods"} 1
`,
		},
		{
			name:    "observe 500 seconds",
			observe: 500.0,
			// 500.0 fits in the 512 bucket.
			want: `
# HELP controller_manager_current_watch_delay_seconds [ALPHA] Last observed watch delay seconds, for now calculated only for pods
# TYPE controller_manager_current_watch_delay_seconds gauge
controller_manager_current_watch_delay_seconds{group="",resource="pods"} 500
# HELP controller_manager_watch_delay_seconds [ALPHA] Watch delay seconds, for now calculated only for pods
# TYPE controller_manager_watch_delay_seconds histogram
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="1"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="2"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="4"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="8"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="16"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="32"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="64"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="128"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="256"} 0
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="512"} 1
controller_manager_watch_delay_seconds_bucket{group="",resource="pods",le="+Inf"} 1
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
