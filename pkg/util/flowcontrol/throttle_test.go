/*
Copyright 2014 The Kubernetes Authors.

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

package flowcontrol

import (
	"math"
	"sync"
	"testing"
	"time"
)

func TestBasicThrottle(t *testing.T) {
	r := NewTokenBucketRateLimiter(1, 3)
	for i := 0; i < 3; i++ {
		if !r.TryAccept() {
			t.Error("unexpected false accept")
		}
	}
	if r.TryAccept() {
		t.Error("unexpected true accept")
	}
}

func TestIncrementThrottle(t *testing.T) {
	r := NewTokenBucketRateLimiter(1, 1)
	if !r.TryAccept() {
		t.Error("unexpected false accept")
	}
	if r.TryAccept() {
		t.Error("unexpected true accept")
	}

	// Allow to refill
	time.Sleep(2 * time.Second)

	if !r.TryAccept() {
		t.Error("unexpected false accept")
	}
}

func TestThrottle(t *testing.T) {
	r := NewTokenBucketRateLimiter(10, 5)

	// Should consume 5 tokens immediately, then
	// the remaining 11 should take at least 1 second (0.1s each)
	expectedFinish := time.Now().Add(time.Second * 1)
	for i := 0; i < 16; i++ {
		r.Accept()
	}
	if time.Now().Before(expectedFinish) {
		t.Error("rate limit was not respected, finished too early")
	}
}

func TestRateLimiterSaturation(t *testing.T) {
	const e = 0.000001
	tests := []struct {
		capacity int
		take     int

		expectedSaturation float64
	}{
		{1, 1, 1},
		{10, 3, 0.3},
	}
	for i, tt := range tests {
		rl := NewTokenBucketRateLimiter(1, tt.capacity)
		for i := 0; i < tt.take; i++ {
			rl.Accept()
		}
		if math.Abs(rl.Saturation()-tt.expectedSaturation) > e {
			t.Fatalf("#%d: Saturation rate difference isn't within tolerable range\n want=%f, get=%f",
				i, tt.expectedSaturation, rl.Saturation())
		}
	}
}

func TestAlwaysFake(t *testing.T) {
	rl := NewFakeAlwaysRateLimiter()
	if !rl.TryAccept() {
		t.Error("TryAccept in AlwaysFake should return true.")
	}
	// If this will block the test will timeout
	rl.Accept()
}

func TestNeverFake(t *testing.T) {
	rl := NewFakeNeverRateLimiter()
	if rl.TryAccept() {
		t.Error("TryAccept in NeverFake should return false.")
	}

	finished := false
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		rl.Accept()
		finished = true
		wg.Done()
	}()

	// Wait some time to make sure it never finished.
	time.Sleep(time.Second)
	if finished {
		t.Error("Accept should block forever in NeverFake.")
	}

	rl.Stop()
	wg.Wait()
	if !finished {
		t.Error("Stop should make Accept unblock in NeverFake.")
	}
}
