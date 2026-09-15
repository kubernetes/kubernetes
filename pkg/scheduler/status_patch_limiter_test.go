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
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
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
//
// This is deliberately synchronous. Asserting instead that a background acquire has not
// returned within some grace period would pass vacuously whenever that goroutine simply
// has not been scheduled yet, and so would not catch a limiter that never blocks at all.
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

func TestStatusPatchLimiter_NilNeverBlocks(t *testing.T) {
	var l *statusPatchLimiter
	for i := 0; i < 3; i++ {
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
	l := newStatusPatchLimiter(2)
	if !awaitAcquireResult(t, acquireLimiterAsync(l, nil)) || !awaitAcquireResult(t, acquireLimiterAsync(l, nil)) {
		t.Fatal("the first two acquires should succeed immediately")
	}
	assertLimiterAtCapacity(t, l)

	// Nothing frees a slot between the probe above and this acquire, so it has to block
	// until the release below hands it one.
	waiter := acquireLimiterAsync(l, nil)
	l.release()
	if !awaitAcquireResult(t, waiter) {
		t.Error("the waiter should have acquired the released slot")
	}
}

func TestStatusPatchLimiter_ReturnsFalseWhenStopped(t *testing.T) {
	l := newStatusPatchLimiter(1)
	if !awaitAcquireResult(t, acquireLimiterAsync(l, nil)) {
		t.Fatal("the first acquire should succeed immediately")
	}
	assertLimiterAtCapacity(t, l)

	// With no slot free, the acquire can only finish by observing stopCh.
	stopCh := make(chan struct{})
	waiter := acquireLimiterAsync(l, stopCh)

	close(stopCh)
	if awaitAcquireResult(t, waiter) {
		t.Error("acquire should report failure when stopped before a slot is free")
	}

	// The abandoned acquire must not have consumed the slot, so releasing the one
	// real holder leaves the limiter empty.
	l.release()
	if !awaitAcquireResult(t, acquireLimiterAsync(l, nil)) {
		t.Error("the only slot should be free again after release")
	}
}

func TestStatusPatchLimiter_FreeSlotTakenWhenAlreadyStopped(t *testing.T) {
	// The fast path must win, so a shutting-down scheduler still drains work
	// that can proceed without waiting.
	l := newStatusPatchLimiter(1)
	stopCh := make(chan struct{})
	close(stopCh)
	if !awaitAcquireResult(t, acquireLimiterAsync(l, stopCh)) {
		t.Error("acquire should succeed when a slot is immediately available")
	}
}
