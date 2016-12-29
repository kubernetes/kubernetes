/*
Copyright 2016 The Kubernetes Authors.

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

package ratelimit

import (
	"testing"
	"time"
)

func TestSimpleExhaustion(t *testing.T) {
	capacity := int64(3)
	b := NewBucketWithRate(1, capacity)

	// Empty the bucket
	for i := int64(0); i < capacity; i++ {
		testAvailable(t, b, capacity-i)
		testTakeNoDelay(t, b, 1)
	}
	testAvailable(t, b, 0)

	// A take on an empty bucket should incur a delay
	testTakeDelay(t, b, 1, 1*time.Second, 100*time.Millisecond)
	testAvailable(t, b, -1)
}

func TestRefill(t *testing.T) {
	capacity := int64(3)
	b := NewBucketWithRate(1, capacity)
	clock := b.lastRefill

	// Empty the bucket
	testAvailable(t, b, capacity)
	for i := int64(0); i < capacity; i++ {
		testTakeNoDelay(t, b, 1)
	}
	testAvailable(t, b, 0)

	// In one second, one unit should be refilled
	clock += time.Second.Nanoseconds()
	b.refillAtTimestamp(clock)
	testAvailable(t, b, 1)
	testTakeNoDelay(t, b, 1)
	testAvailable(t, b, 0)

	// Partial refill periods don't result in lost time
	for i := 0; i < 4; i++ {
		clock += time.Millisecond.Nanoseconds() * 200
		b.refillAtTimestamp(clock)
		testAvailable(t, b, 0)
	}
	clock += time.Millisecond.Nanoseconds() * 200
	b.refillAtTimestamp(clock)
	testAvailable(t, b, 1)
	testTakeNoDelay(t, b, 1)
	testAvailable(t, b, 0)
}

// TestSlowRefillRate checks we don't have problems with tiny refill rates
func TestSlowRefillRate(t *testing.T) {
	for _, capacity := range []int64{int64(1), int64(1E18)} {
		b := NewBucketWithRate(1E-9, capacity)
		clock := b.lastRefill

		// Empty the bucket
		testTakeNoDelay(t, b, b.available)

		// In one second, should refill nothing
		clock += time.Second.Nanoseconds()
		b.refillAtTimestamp(clock)
		testAvailable(t, b, 0)

		// We need to have 1E18 nanos to see any refill
		clock += 1E18
		b.refillAtTimestamp(clock)
		testAvailable(t, b, 1)
		testTakeNoDelay(t, b, 1)
		testAvailable(t, b, 0)
	}
}

// TestFastRefillRate checks for refill rates that are around 1 / ns (our granularity)
func TestFastRefillRate(t *testing.T) {
	for _, capacity := range []int64{int64(1), int64(1E18)} {
		b := NewBucketWithRate(1E9, capacity)

		// Empty the bucket
		testTakeNoDelay(t, b, b.available)

		// In one nanosecond, should refill exactly one unit
		clock := b.lastRefill + 1
		b.refillAtTimestamp(clock)
		testAvailable(t, b, 1)
		testTakeNoDelay(t, b, 1)
		testAvailable(t, b, 0)
	}
}

// TestRefillRatePrecision checks for rounding errors
func TestRefillRatePrecision(t *testing.T) {
	capacity := int64(1E18)
	b := NewBucketWithRate(1+1E9, capacity)

	// Empty the bucket
	testTakeNoDelay(t, b, b.available)

	// In one nanosecond, should refill exactly one unit
	clock := b.lastRefill + 1
	b.refillAtTimestamp(clock)
	testAvailable(t, b, 1)
	testTakeNoDelay(t, b, 1)
	testAvailable(t, b, 0)

	// In one second, should refill the 1 extra also
	clock += 1E9
	b.refillAtTimestamp(clock)
	testAvailable(t, b, 1000000001)
	testTakeNoDelay(t, b, 1000000001)
	testAvailable(t, b, 0)
}

// TestSlowRefillRate checks we don't have problems with ridiculously high refill rates
func TestHugeRefillRate(t *testing.T) {
	for _, capacity := range []int64{int64(1), int64(1E18)} {
		b := NewBucketWithRate(1E27, capacity)

		// Empty the bucket
		testTakeNoDelay(t, b, b.available)

		// In one nanosecond, should refill to capacity
		clock := b.lastRefill + 1
		b.refillAtTimestamp(clock)
		testAvailable(t, b, capacity)
		testTakeNoDelay(t, b, capacity)
		testAvailable(t, b, 0)

		// In one second, should refill to capacity, but with huge overflow that must be discarded
		clock += time.Second.Nanoseconds()
		b.refillAtTimestamp(clock)
		testAvailable(t, b, capacity)
		testTakeNoDelay(t, b, capacity)
		testAvailable(t, b, 0)
	}
}

func testAvailable(t *testing.T, b *Bucket, expected int64) {
	available := b.available
	if available != expected {
		t.Errorf("unexpected available; expected=%d, actual=%d", expected, available)
	}
}

func testTakeDelay(t *testing.T, b *Bucket, take int64, expected time.Duration, tolerance time.Duration) {
	actual := b.Take(take)
	error := expected.Nanoseconds() - actual.Nanoseconds()
	if error < 0 {
		error = -error
	}
	if error > tolerance.Nanoseconds() {
		t.Errorf("unexpected delay on take(%d); expected=%d, actual=%d", take, expected, actual)
	}
}

func testTakeNoDelay(t *testing.T, b *Bucket, take int64) {
	testTakeDelay(t, b, take, 0, 0)
}
