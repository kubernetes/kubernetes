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
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	componentmetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// acquireLimiterAsync starts an acquire in the background and returns the channel carrying its result.
// Exactly one value is ever sent, so the caller can assert what it returned without leaving a
// waiter parked on the limiter.
func acquireLimiterAsync(l *statusPatchLimiter, stopCh <-chan struct{}) <-chan bool {
	result := make(chan bool, 1)
	go func() { result <- l.acquire(stopCh) }()
	return result
}

// assertLimiterAtCapacity fails if the limiter still has a free slot.
func assertLimiterAtCapacity(t *testing.T, l *statusPatchLimiter) {
	t.Helper()
	select {
	case l.tokens <- token:
		t.Fatal("a slot was still available while the limiter should be at capacity")
	default:
	}
}

func awaitAcquireResult(t *testing.T, result <-chan bool) bool {
	t.Helper()
	select {
	case got := <-result:
		return got
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("Timed out waiting for acquire to return")
		return false
	}
}

func initThrottledPatchesMetricForTest() {
	metrics.Register()
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

func TestStatusPatchLimiter_NilNeverBlocks(t *testing.T) {
	var l *statusPatchLimiter
	for range 3 {
		if !l.acquire(nil) {
			t.Fatal("acquire on a nil limiter should succeed")
		}
		l.release()
	}
}

func TestStatusPatchLimiter_NonPositiveReturnsNil(t *testing.T) {
	if newStatusPatchLimiter(0) != nil {
		t.Error("newStatusPatchLimiter(0) should return nil")
	}
	if newStatusPatchLimiter(-1) != nil {
		t.Error("newStatusPatchLimiter(-1) should return nil")
	}
}

func TestStatusPatchLimiter_CapsConcurrentHolders(t *testing.T) {
	initThrottledPatchesMetricForTest()
	synctest.Test(t, func(t *testing.T) {
		l := newStatusPatchLimiter(2)
		for range 2 {
			if !awaitAcquireResult(t, acquireLimiterAsync(l, nil)) {
				t.Fatal("the first two acquires should succeed immediately")
			}
		}
		assertThrottledPatchesMetric(t, 0)

		waiter := acquireLimiterAsync(l, nil)
		synctest.Wait()
		assertThrottledPatchesMetric(t, 1)

		// A second blocked caller is counted as well.
		waiter2 := acquireLimiterAsync(l, nil)
		synctest.Wait()
		assertThrottledPatchesMetric(t, 2)

		// Each release hands the slot to exactly one waiter and decrements the metric by one.
		l.release()
		synctest.Wait()
		assertThrottledPatchesMetric(t, 1)

		l.release()
		if !awaitAcquireResult(t, waiter) || !awaitAcquireResult(t, waiter2) {
			t.Error("both waiters should have acquired a released slot")
		}
		assertThrottledPatchesMetric(t, 0)
	})
}

func TestStatusPatchLimiter_ReturnsFalseWhenStopped(t *testing.T) {
	initThrottledPatchesMetricForTest()
	synctest.Test(t, func(t *testing.T) {
		l := newStatusPatchLimiter(1)
		if !awaitAcquireResult(t, acquireLimiterAsync(l, nil)) {
			t.Fatal("the first acquire should succeed immediately")
		}
		assertThrottledPatchesMetric(t, 0)

		stopCh := make(chan struct{})
		waiter := acquireLimiterAsync(l, stopCh)
		synctest.Wait()
		assertThrottledPatchesMetric(t, 1)

		close(stopCh)
		if awaitAcquireResult(t, waiter) {
			t.Error("acquire should report failure when stopped before a slot is free")
		}
		assertThrottledPatchesMetric(t, 0)

		// The abandoned acquire must not have consumed the slot, so releasing the one
		// real holder leaves the limiter empty.
		l.release()
		if !awaitAcquireResult(t, acquireLimiterAsync(l, nil)) {
			t.Error("the only slot should be free again after release")
		}
		assertThrottledPatchesMetric(t, 0)
	})
}

func TestStatusPatchLimiter_ReturnsFalseWhenAlreadyStopped(t *testing.T) {
	initThrottledPatchesMetricForTest()
	l := newStatusPatchLimiter(1)
	stopCh := make(chan struct{})
	close(stopCh)
	if awaitAcquireResult(t, acquireLimiterAsync(l, stopCh)) {
		t.Error("acquire should return false when stopCh is already closed, even if a slot is free")
	}
	assertThrottledPatchesMetric(t, 0)
}
