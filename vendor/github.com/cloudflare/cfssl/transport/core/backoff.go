package core

// http://www.awsarchitectureblog.com/2015/03/backoff.html

import (
	"math"
	mrand "math/rand"
	"sync"
	"time"
)

// DefaultInterval is used when a Backoff is initialised with a
// zero-value Interval.
var DefaultInterval = 5 * time.Minute

// DefaultMaxDuration is maximum amount of time that the backoff will
// delay for.
var DefaultMaxDuration = 6 * time.Hour

// A Backoff contains the information needed to intelligently backoff
// and retry operations using an exponential backoff algorithm. It may
// be initialised with all zero values and it will behave sanely.
type Backoff struct {
	// MaxDuration is the largest possible duration that can be
	// returned from a call to Duration.
	MaxDuration time.Duration

	// Interval controls the time step for backing off.
	Interval time.Duration

	// Jitter controls whether to use the "Full Jitter"
	// improvement to attempt to smooth out spikes in a high
	// contention scenario.
	Jitter bool

	tries int
	lock  *sync.Mutex // lock guards tries
}

func (b *Backoff) setup() {
	if b.Interval == 0 {
		b.Interval = DefaultInterval
	}

	if b.MaxDuration == 0 {
		b.MaxDuration = DefaultMaxDuration
	}

	if b.lock == nil {
		b.lock = new(sync.Mutex)
	}
}

// Duration returns a time.Duration appropriate for the backoff,
// incrementing the attempt counter.
func (b *Backoff) Duration() time.Duration {
	b.setup()
	b.lock.Lock()
	defer b.lock.Unlock()

	pow := 1 << uint(b.tries)

	// MaxInt16 is an arbitrary choice on an upper bound; the
	// implication is that every 16 tries, the counter resets.
	if pow > math.MaxInt16 {
		b.tries = 0
		pow = 1
	}

	t := time.Duration(pow)
	b.tries++
	t = b.Interval * t
	if t > b.MaxDuration {
		t = b.MaxDuration
	}

	if b.Jitter {
		t = time.Duration(mrand.Int63n(int64(t)))
	}

	return t
}

// Reset clears the backoff.
func (b *Backoff) Reset() {
	b.setup()
	b.lock.Lock()
	b.tries = 0
	b.lock.Unlock()
}
