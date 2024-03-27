// Package backoff implements backoff algorithms for retrying operations.
//
// Use Retry function for retrying operations that may fail.
// If Retry does not meet your needs,
// copy/paste the function into your project and modify as you wish.
//
// There is also Ticker type similar to time.Ticker.
// You can use it if you need to work with channels.
//
// See Examples section below for usage examples.
package backoff

import "time"

// BackOff is a backoff policy for retrying an operation.
type BackOff interface {
	// NextBackOff returns the duration to wait before retrying the operation,
	// or backoff. Stop to indicate that no more retries should be made.
	//
	// Example usage:
	//
	// 	duration := backoff.NextBackOff();
	// 	if (duration == backoff.Stop) {
	// 		// Do not retry operation.
	// 	} else {
	// 		// Sleep for duration and retry operation.
	// 	}
	//
	NextBackOff() time.Duration

	// Reset to initial state.
	Reset()
}

// Stop indicates that no more retries should be made for use in NextBackOff().
const Stop time.Duration = -1

// ZeroBackOff is a fixed backoff policy whose backoff time is always zero,
// meaning that the operation is retried immediately without waiting, indefinitely.
type ZeroBackOff struct{}

func (b *ZeroBackOff) Reset() {}

func (b *ZeroBackOff) NextBackOff() time.Duration { return 0 }

// StopBackOff is a fixed backoff policy that always returns backoff.Stop for
// NextBackOff(), meaning that the operation should never be retried.
type StopBackOff struct{}

func (b *StopBackOff) Reset() {}

func (b *StopBackOff) NextBackOff() time.Duration { return Stop }

// ConstantBackOff is a backoff policy that always returns the same backoff delay.
// This is in contrast to an exponential backoff policy,
// which returns a delay that grows longer as you call NextBackOff() over and over again.
type ConstantBackOff struct {
	Interval time.Duration
}

func (b *ConstantBackOff) Reset()                     {}
func (b *ConstantBackOff) NextBackOff() time.Duration { return b.Interval }

func NewConstantBackOff(d time.Duration) *ConstantBackOff {
	return &ConstantBackOff{Interval: d}
}
