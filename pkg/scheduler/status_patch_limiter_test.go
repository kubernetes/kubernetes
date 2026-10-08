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

package scheduler

import (
	"testing"
	"testing/synctest"

	componentmetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// assertLimiterAtCapacity fails if the limiter still has a free slot.
func assertLimiterAtCapacity(t *testing.T, l *statusPatchLimiter) {
	t.Helper()
	if held := len(l.tokens); held != cap(l.tokens) {
		t.Fatalf("Limiter holds %d of %d slots, want it at capacity", held, cap(l.tokens))
	}
}

func initThrottledPatchesMetricForTest() {
	// init() in scheduler_test.go has already run metrics.Register() with SchedulerAsyncAPICalls
	// off, which skips this metric, and Register() only runs once.
	componentmetrics.NewKubeRegistry().MustRegister(metrics.FailureHandlerThrottledPatches)
	metrics.FailureHandlerThrottledPatches.Set(0)
}

func assertThrottledPatchesMetric(t *testing.T, want float64) {
	t.Helper()
	got, err := testutil.GetGaugeMetricValue(metrics.FailureHandlerThrottledPatches)
	if err != nil {
		t.Fatalf("Failed to read FailureHandlerThrottledPatches metric: %v", err)
	}
	if got != want {
		t.Errorf("FailureHandlerThrottledPatches = %v, want %v", got, want)
	}
}

// stopMode selects when TestStatusPatchLimiter_Acquire closes stopCh.
type stopMode int

const (
	noStop stopMode = iota
	stopBeforeAcquire
	stopWhileWaiting
)

func TestStatusPatchLimiter_Acquire(t *testing.T) {
	tests := []struct {
		name string
		// capacity is the limiter size; 0 gives a nil limiter, which never limits.
		capacity int
		// held is the number of slots taken before the waiters start.
		held int
		// waiters is the number of concurrent acquire calls under test.
		waiters int
		stop    stopMode
		// wantWaiting is the throttled metric once every waiter has acquired a slot or parked.
		wantWaiting  int
		wantAcquired bool
		// wantHeld is the number of slots held at the end; a waiter that gave up must not hold one.
		wantHeld int
	}{
		{
			name:         "nil limiter never blocks",
			held:         3,
			waiters:      1,
			wantAcquired: true,
		},
		{
			name:         "free slot is taken without waiting",
			capacity:     1,
			waiters:      1,
			wantAcquired: true,
			wantHeld:     1,
		},
		{
			name:         "full limiter makes callers wait until release",
			capacity:     2,
			held:         2,
			waiters:      2,
			wantWaiting:  2,
			wantAcquired: true,
			wantHeld:     2,
		},
		{
			name:        "stop while waiting returns false",
			capacity:    1,
			held:        1,
			waiters:     1,
			stop:        stopWhileWaiting,
			wantWaiting: 1,
			wantHeld:    1,
		},
		{
			name:     "already stopped returns false even with a free slot",
			capacity: 1,
			waiters:  1,
			stop:     stopBeforeAcquire,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			initThrottledPatchesMetricForTest()
			synctest.Test(t, func(t *testing.T) {
				l := newStatusPatchLimiter(tt.capacity)
				for range tt.held {
					if !l.acquire(nil) {
						t.Fatal("Failed to take a slot before starting the waiters")
					}
				}

				stopCh := make(chan struct{})
				if tt.stop == stopBeforeAcquire {
					close(stopCh)
				}
				results := make(chan bool, tt.waiters)
				for range tt.waiters {
					go func() { results <- l.acquire(stopCh) }()
				}
				synctest.Wait()
				assertThrottledPatchesMetric(t, float64(tt.wantWaiting))

				if tt.stop == stopWhileWaiting {
					close(stopCh)
				} else {
					// Each release hands its slot to exactly one parked waiter.
					for waiting := tt.wantWaiting; waiting > 0; waiting-- {
						l.release()
						synctest.Wait()
						assertThrottledPatchesMetric(t, float64(waiting-1))
					}
				}

				// A waiter that never returns fails the test as a synctest deadlock.
				for range tt.waiters {
					if got := <-results; got != tt.wantAcquired {
						t.Errorf("acquire() = %v, want %v", got, tt.wantAcquired)
					}
				}
				assertThrottledPatchesMetric(t, 0)
				if l != nil && len(l.tokens) != tt.wantHeld {
					t.Errorf("Limiter holds %d slots, want %d", len(l.tokens), tt.wantHeld)
				}
			})
		})
	}
}
