//go:build !providerless
// +build !providerless

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

package aws

import (
	"testing"
	"time"
)

// There follows a group of tests for the backoff logic.  There's nothing
// particularly special about the values chosen: if we tweak the values in the
// backoff logic then we might well have to update the tests.  However the key
// behavioural elements should remain (e.g. no errors => no backoff), and these
// are each tested by one of the tests below.

// Test that we don't apply any delays when there are no errors
func TestBackoffNoErrors(t *testing.T) {
	b := &Backoff{}
	b.init(decayIntervalSeconds, decayFraction, maxDelay)

	now := time.Now()
	for i := 0; i < 100; i++ {
		d := b.ComputeDelayForRequest(now)
		if d.Nanoseconds() != 0 {
			t.Fatalf("unexpected delay during no-error case")
		}
		now = now.Add(time.Second)
	}
}

// Test that we always apply a delay when there are errors, and also that we
// don't "flap" - that our own delay doesn't cause us to oscillate between
// delay and no-delay.
func TestBackoffAllErrors(t *testing.T) {
	b := &Backoff{}
	b.init(decayIntervalSeconds, decayFraction, maxDelay)

	now := time.Now()
	// Warm up
	for i := 0; i < 10; i++ {
		_ = b.ComputeDelayForRequest(now)
		b.ReportError()
		now = now.Add(time.Second)
	}

	for i := 0; i < 100; i++ {
		d := b.ComputeDelayForRequest(now)
		b.ReportError()
		if d.Seconds() < 5 {
			t.Fatalf("unexpected short-delay during all-error case: %v", d)
		}
		t.Logf("delay @%d %v", i, d)
		now = now.Add(d)
	}
}

// Test that we do come close to our max delay, when we see all errors at 1
// second intervals (this simulates multiple concurrent requests, because we
// don't wait for delay in between requests)
func TestBackoffHitsMax(t *testing.T) {
	b := &Backoff{}
	b.init(decayIntervalSeconds, decayFraction, maxDelay)

	now := time.Now()
	for i := 0; i < 100; i++ {
		_ = b.ComputeDelayForRequest(now)
		b.ReportError()
		now = now.Add(time.Second)
	}

	for i := 0; i < 10; i++ {
		d := b.ComputeDelayForRequest(now)
		b.ReportError()
		if float32(d.Nanoseconds()) < (float32(maxDelay.Nanoseconds()) * 0.95) {
			t.Fatalf("expected delay to be >= 95 percent of max delay, was %v", d)
		}
		t.Logf("delay @%d %v", i, d)
		now = now.Add(time.Second)
	}
}

// Test that after a phase of errors, we eventually stop applying a delay once there are
// no more errors.
func TestBackoffRecovers(t *testing.T) {
	b := &Backoff{}
	b.init(decayIntervalSeconds, decayFraction, maxDelay)

	now := time.Now()

	// Phase of all-errors
	for i := 0; i < 100; i++ {
		_ = b.ComputeDelayForRequest(now)
		b.ReportError()
		now = now.Add(time.Second)
	}

	for i := 0; i < 10; i++ {
		d := b.ComputeDelayForRequest(now)
		b.ReportError()
		if d.Seconds() < 5 {
			t.Fatalf("unexpected short-delay during all-error phase: %v", d)
		}
		t.Logf("error phase delay @%d %v", i, d)
		now = now.Add(time.Second)
	}

	// Phase of no errors
	for i := 0; i < 100; i++ {
		_ = b.ComputeDelayForRequest(now)
		now = now.Add(3 * time.Second)
	}

	for i := 0; i < 10; i++ {
		d := b.ComputeDelayForRequest(now)
		if d.Seconds() != 0 {
			t.Fatalf("unexpected delay during error recovery phase: %v", d)
		}
		t.Logf("no-error phase delay @%d %v", i, d)
		now = now.Add(time.Second)
	}
}
