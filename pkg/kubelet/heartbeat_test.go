package kubelet

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
	err := registerInitiallyWithInfiniteRetry(clock, func() bool {
		tries = tries + 1
		// 10,000 is an arbitrarily large number which indicates that retries continue forever.
		if tries < 10000 {
			return false
		}
		return true
	})
	elapsed := clock.Since(start).Seconds()
	if err != nil {
		t.Fatal("Something went wrong, the test should have returned with no error.")
	}
	if clock.Since(start).Seconds() < 70000-500 {
		t.Fatal("Backoff should have gone up to 70,000 seconds, but was too low: %v", elapsed)
	}
	if clock.Since(start).Seconds() > 80000 {
		t.Fatal("Backoff should be less then 80,000 seconds, but was too high: %v.", elapsed)
	}
}

func TestNodeStatusBurst(t *testing.T) {
	start := time.Now()
	clock := clock.NewFakeClock(start)
	tries := 0
	err := updateNodeStatusWithRetry(clock, func() error {
		if tries == 4 {
			return nil
		}
		tries = tries + 1
		return fmt.Errorf("retry me!")
	})
	if err != nil {
		t.Fatal("Several retries should have occured, with the last one succeeded, but it failed.")
	}

	// This bursty behaviour isn't necessarily ideal, but we assert it for the sake of being explicit.
	// See TODO's in the heartbeat module for future updates.
	if clock.Since(start).Seconds() > .03 {
		t.Fatal("Several retries should have completed within roughly 25 milliseconds")
	}
}
