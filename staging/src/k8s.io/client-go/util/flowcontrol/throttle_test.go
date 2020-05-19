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
	"context"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestMultithreadedThrottling(t *testing.T) {
	// Bucket with 100QPS and no burst
	r := NewTokenBucketRateLimiter(100, 1)

	// channel to collect 100 tokens
	taken := make(chan bool, 100)

	// Set up goroutines to hammer the throttler
	startCh := make(chan bool)
	endCh := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			// wait for the starting signal
			<-startCh
			for {
				// get a token
				r.Accept()
				select {
				// try to add it to the taken channel
				case taken <- true:
					continue
				// if taken is full, notify and return
				default:
					endCh <- true
					return
				}
			}
		}()
	}

	// record wall time
	startTime := time.Now()
	// take the initial capacity so all tokens are the result of refill
	r.Accept()
	// start the thundering herd
	close(startCh)
	// wait for the first signal that we collected 100 tokens
	<-endCh
	// record wall time
	endTime := time.Now()

	// tolerate a 1% clock change because these things happen
	if duration := endTime.Sub(startTime); duration < (time.Second * 99 / 100) {
		// We shouldn't be able to get 100 tokens out of the bucket in less than 1 second of wall clock time, no matter what
		t.Errorf("Expected it to take at least 1 second to get 100 tokens, took %v", duration)
	} else {
		t.Logf("Took %v to get 100 tokens", duration)
	}
}

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

func TestWait(t *testing.T) {
	r := NewTokenBucketRateLimiter(0.0001, 1)

	ctx, cancelFn := context.WithTimeout(context.Background(), time.Second)
	defer cancelFn()
	if err := r.Wait(ctx); err != nil {
		t.Errorf("unexpected wait failed, err: %v", err)
	}

	ctx2, cancelFn2 := context.WithTimeout(context.Background(), time.Second)
	defer cancelFn2()
	if err := r.Wait(ctx2); err == nil {
		t.Errorf("unexpected wait success")
	} else {
		t.Log(fmt.Sprintf("wait err: %v", err))
	}
}
