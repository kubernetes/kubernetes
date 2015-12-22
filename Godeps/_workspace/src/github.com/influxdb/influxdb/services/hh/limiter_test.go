package hh

import (
	"testing"
	"time"
)

func TestLimiter(t *testing.T) {
	l := NewRateLimiter(0)
	l.Update(500)
	if l.Delay().Nanoseconds() != 0 {
		t.Errorf("limiter with no limit mismatch: got %v, exp 0", l.Delay())
	}
}

func TestLimiterWithinLimit(t *testing.T) {
	if testing.Short() {
		t.Skip("Shipping TestLimiterWithinLimit")
	}

	l := NewRateLimiter(1000)
	for i := 0; i < 100; i++ {
		// 50 ever 100ms = 500/s which should be within the rate
		l.Update(50)
		l.Delay()
		time.Sleep(100 * time.Millisecond)
	}

	// Should not have any delay
	delay := l.Delay().Seconds()
	if exp := int(0); int(delay) != exp {
		t.Errorf("limiter rate mismatch: got %v, exp %v", int(delay), exp)
	}

}

func TestLimiterExceeded(t *testing.T) {
	l := NewRateLimiter(1000)
	for i := 0; i < 10; i++ {
		l.Update(200)
		l.Delay()
	}
	delay := l.Delay().Seconds()
	if int(delay) == 0 {
		t.Errorf("limiter rate mismatch. expected non-zero delay")
	}
}
