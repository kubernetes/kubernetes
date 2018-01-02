// Package ratelimiter provides an exceedingly simple rate limiter.
package ratelimiter

import (
	"github.com/juju/ratelimit"
)

// Limiter is a simple rate limiter.
type Limiter struct {
	bucket *ratelimit.Bucket
}

// Wait blocks for the amount of time required by the Limiter so as to not
// exceed its rate.
func (l *Limiter) Wait() {
	l.bucket.Wait(1)
}

// NewLimiter creates a new Limiter with a rate of limit per second.
func NewLimiter(limit int) *Limiter {
	return &Limiter{bucket: ratelimit.NewBucketWithRate(float64(limit), 1)}
}
