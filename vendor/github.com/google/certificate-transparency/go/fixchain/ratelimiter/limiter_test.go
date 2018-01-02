package ratelimiter

import (
	"sync"
	"testing"
	"time"
)

var testlimits = []int{1, 10, 50, 100, 1000}

func TestRateLimiterSingleThreaded(t *testing.T) {
	for i, limit := range testlimits {
		l := NewLimiter(limit)
		count := 0
		tick := time.NewTicker(time.Second)
		go func() {
			for _ = range tick.C {
				// Allow a count up to one more than the limit as scheduling of
				// goroutine vs the main thread could cause this check to not be
				// run quite in time for limit.
				if count > limit+1 {
					t.Errorf("#%d: Too many operations per second. Expected %d, got %d", i, limit, count)
				}
				count = 0
			}
		}()

		for i := 0; i < 3*limit; i++ {
			l.Wait()
			count++
		}
		tick.Stop()
	}
}

func TestRateLimiterGoroutines(t *testing.T) {
	for i, limit := range testlimits {
		l := NewLimiter(limit)
		count := 0
		tick := time.NewTicker(time.Second)
		go func() {
			for _ = range tick.C {
				// Allow a count up to one more than the limit as scheduling of
				// goroutine vs the main thread could cause this check to not be
				// run quite in time for limit.
				if count > limit+1 {
					t.Errorf("#%d: Too many operations per second. Expected %d, got %d", i, limit, count)
				}
				count = 0
			}
		}()

		var wg sync.WaitGroup
		for i := 0; i < 3*limit; i++ {
			wg.Add(1)
			go func() {
				l.Wait()
				count++
				wg.Done()
			}()
		}
		wg.Wait()
		tick.Stop()
	}
}
