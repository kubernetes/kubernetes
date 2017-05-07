/*
Copyright 2017 The Kubernetes Authors.

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

package policies

import (
	"fmt"

	"k8s.io/client-go/util/clock"

	"testing"
	"time"
)

func TestInitialRegistration(t *testing.T) {
	tries := 0
	start := time.Now()
	clock := clock.NewFakeClock(start)
	err := RegisterInitiallyWithInfiniteRetry(clock, func() bool {
		tries = tries + 1
		// 10,000 is an arbitrarily large number which indicates that retries continue forever.
		if tries < 10000 {
			return false
		}
		return true
	}, "test initial reg")
	elapsed := clock.Since(start).Seconds()
	if err != nil {
		t.Fatalf("Something went wrong, the test should have returned with no error.")
	}
	if clock.Since(start).Seconds() < 70000-500 {
		t.Fatalf("Backoff should have gone up to 70,000 seconds, but was too low: %v", elapsed)
	}
	if clock.Since(start).Seconds() > 80000 {
		t.Fatalf("Backoff should be less then 80,000 seconds, but was too high: %v.", elapsed)
	}
}

func TestNodeStatusBurst(t *testing.T) {
	start := time.Now()
	clock := clock.NewFakeClock(start)
	tries := 0
	err := UpdateNodeStatusWithBurstRetry(clock, func() error {
		if tries == 4 {
			return nil
		}
		tries = tries + 1
		return fmt.Errorf("retry me!")
	}, "test node status burst")
	if err != nil {
		t.Fatal("Several retries should have occured, with the last one succeeded, but it failed.")
	}

	// This bursty behaviour isn't necessarily ideal, but we assert it for the sake of being explicit.
	// See TODO's in the heartbeat module for future updates.
	if clock.Since(start).Seconds() > .03 {
		t.Fatal("Several retries should have completed within roughly 25 milliseconds")
	}
}
