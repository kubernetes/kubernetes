package congestion

import "time"

// A Clock returns the current time
type Clock interface {
	Now() time.Time
}

// DefaultClock implements the Clock interface using the Go stdlib clock.
type DefaultClock struct{}

var _ Clock = DefaultClock{}

// Now gets the current time
func (DefaultClock) Now() time.Time {
	return time.Now()
}
