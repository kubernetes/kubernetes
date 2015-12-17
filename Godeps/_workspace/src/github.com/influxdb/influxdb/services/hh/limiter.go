package hh

import "time"

type limiter struct {
	count int64
	limit int64
	start time.Time
	delay float64
}

// NewRateLimiter returns a new limiter configured to restrict a process to the limit per second.
// limit is the maximum amount that can be used per second.  The limit should be > 0.  A limit
// <= 0, will not limit the processes.
func NewRateLimiter(limit int64) *limiter {
	return &limiter{
		start: time.Now(),
		limit: limit,
		delay: 0.5,
	}
}

// Update updates the amount used
func (t *limiter) Update(count int) {
	t.count += int64(count)
}

// Delay returns the amount of time, up to 1 second, that caller should wait
// to maintain the configured rate
func (t *limiter) Delay() time.Duration {
	if t.limit > 0 {

		delta := time.Now().Sub(t.start).Seconds()
		rate := int64(float64(t.count) / delta)

		// Determine how far off from the max rate we are
		delayAdj := float64((t.limit - rate)) / float64(t.limit)

		// Don't adjust by more than 1 second at a time
		delayAdj = t.clamp(delayAdj, -1, 1)

		t.delay -= delayAdj
		if t.delay < 0 {
			t.delay = 0
		}

		return time.Duration(t.delay) * time.Second
	}
	return time.Duration(0)
}

func (t *limiter) clamp(value, min, max float64) float64 {
	if value < min {
		return min
	}

	if value > max {
		return max
	}
	return value
}
