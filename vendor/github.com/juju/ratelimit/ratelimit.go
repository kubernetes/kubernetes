// Copyright 2014 Canonical Ltd.
// Licensed under the LGPLv3 with static-linking exception.
// See LICENCE file for details.

// Package ratelimit provides an efficient token bucket implementation
// that can be used to limit the rate of arbitrary things.
// See http://en.wikipedia.org/wiki/Token_bucket.
package ratelimit

import (
	"math"
	"strconv"
	"sync"
	"time"
)

// Bucket represents a token bucket that fills at a predetermined rate.
// Methods on Bucket may be called concurrently.
type Bucket struct {
	startTime    time.Time
	capacity     int64
	quantum      int64
	fillInterval time.Duration
	clock        Clock

	// The mutex guards the fields following it.
	mu sync.Mutex

	// avail holds the number of available tokens
	// in the bucket, as of availTick ticks from startTime.
	// It will be negative when there are consumers
	// waiting for tokens.
	avail     int64
	availTick int64
}

// Clock is used to inject testable fakes.
type Clock interface {
	Now() time.Time
	Sleep(d time.Duration)
}

// realClock implements Clock in terms of standard time functions.
type realClock struct{}

// Now is identical to time.Now.
func (realClock) Now() time.Time {
	return time.Now()
}

// Sleep is identical to time.Sleep.
func (realClock) Sleep(d time.Duration) {
	time.Sleep(d)
}

// NewBucket returns a new token bucket that fills at the
// rate of one token every fillInterval, up to the given
// maximum capacity. Both arguments must be
// positive. The bucket is initially full.
func NewBucket(fillInterval time.Duration, capacity int64) *Bucket {
	return NewBucketWithClock(fillInterval, capacity, realClock{})
}

// NewBucketWithClock is identical to NewBucket but injects a testable clock
// interface.
func NewBucketWithClock(fillInterval time.Duration, capacity int64, clock Clock) *Bucket {
	return NewBucketWithQuantumAndClock(fillInterval, capacity, 1, clock)
}

// rateMargin specifes the allowed variance of actual
// rate from specified rate. 1% seems reasonable.
const rateMargin = 0.01

// NewBucketWithRate returns a token bucket that fills the bucket
// at the rate of rate tokens per second up to the given
// maximum capacity. Because of limited clock resolution,
// at high rates, the actual rate may be up to 1% different from the
// specified rate.
func NewBucketWithRate(rate float64, capacity int64) *Bucket {
	return NewBucketWithRateAndClock(rate, capacity, realClock{})
}

// NewBucketWithRateAndClock is identical to NewBucketWithRate but injects a
// testable clock interface.
func NewBucketWithRateAndClock(rate float64, capacity int64, clock Clock) *Bucket {
	for quantum := int64(1); quantum < 1<<50; quantum = nextQuantum(quantum) {
		fillInterval := time.Duration(1e9 * float64(quantum) / rate)
		if fillInterval <= 0 {
			continue
		}
		tb := NewBucketWithQuantumAndClock(fillInterval, capacity, quantum, clock)
		if diff := math.Abs(tb.Rate() - rate); diff/rate <= rateMargin {
			return tb
		}
	}
	panic("cannot find suitable quantum for " + strconv.FormatFloat(rate, 'g', -1, 64))
}

// nextQuantum returns the next quantum to try after q.
// We grow the quantum exponentially, but slowly, so we
// get a good fit in the lower numbers.
func nextQuantum(q int64) int64 {
	q1 := q * 11 / 10
	if q1 == q {
		q1++
	}
	return q1
}

// NewBucketWithQuantum is similar to NewBucket, but allows
// the specification of the quantum size - quantum tokens
// are added every fillInterval.
func NewBucketWithQuantum(fillInterval time.Duration, capacity, quantum int64) *Bucket {
	return NewBucketWithQuantumAndClock(fillInterval, capacity, quantum, realClock{})
}

// NewBucketWithQuantumAndClock is identical to NewBucketWithQuantum but injects
// a testable clock interface.
func NewBucketWithQuantumAndClock(fillInterval time.Duration, capacity, quantum int64, clock Clock) *Bucket {
	if fillInterval <= 0 {
		panic("token bucket fill interval is not > 0")
	}
	if capacity <= 0 {
		panic("token bucket capacity is not > 0")
	}
	if quantum <= 0 {
		panic("token bucket quantum is not > 0")
	}
	return &Bucket{
		clock:        clock,
		startTime:    clock.Now(),
		capacity:     capacity,
		quantum:      quantum,
		avail:        capacity,
		fillInterval: fillInterval,
	}
}

// Wait takes count tokens from the bucket, waiting until they are
// available.
func (tb *Bucket) Wait(count int64) {
	if d := tb.Take(count); d > 0 {
		tb.clock.Sleep(d)
	}
}

// WaitMaxDuration is like Wait except that it will
// only take tokens from the bucket if it needs to wait
// for no greater than maxWait. It reports whether
// any tokens have been removed from the bucket
// If no tokens have been removed, it returns immediately.
func (tb *Bucket) WaitMaxDuration(count int64, maxWait time.Duration) bool {
	d, ok := tb.TakeMaxDuration(count, maxWait)
	if d > 0 {
		tb.clock.Sleep(d)
	}
	return ok
}

const infinityDuration time.Duration = 0x7fffffffffffffff

// Take takes count tokens from the bucket without blocking. It returns
// the time that the caller should wait until the tokens are actually
// available.
//
// Note that if the request is irrevocable - there is no way to return
// tokens to the bucket once this method commits us to taking them.
func (tb *Bucket) Take(count int64) time.Duration {
	d, _ := tb.take(tb.clock.Now(), count, infinityDuration)
	return d
}

// TakeMaxDuration is like Take, except that
// it will only take tokens from the bucket if the wait
// time for the tokens is no greater than maxWait.
//
// If it would take longer than maxWait for the tokens
// to become available, it does nothing and reports false,
// otherwise it returns the time that the caller should
// wait until the tokens are actually available, and reports
// true.
func (tb *Bucket) TakeMaxDuration(count int64, maxWait time.Duration) (time.Duration, bool) {
	return tb.take(tb.clock.Now(), count, maxWait)
}

// TakeAvailable takes up to count immediately available tokens from the
// bucket. It returns the number of tokens removed, or zero if there are
// no available tokens. It does not block.
func (tb *Bucket) TakeAvailable(count int64) int64 {
	return tb.takeAvailable(tb.clock.Now(), count)
}

// takeAvailable is the internal version of TakeAvailable - it takes the
// current time as an argument to enable easy testing.
func (tb *Bucket) takeAvailable(now time.Time, count int64) int64 {
	if count <= 0 {
		return 0
	}
	tb.mu.Lock()
	defer tb.mu.Unlock()

	tb.adjust(now)
	if tb.avail <= 0 {
		return 0
	}
	if count > tb.avail {
		count = tb.avail
	}
	tb.avail -= count
	return count
}

// Available returns the number of available tokens. It will be negative
// when there are consumers waiting for tokens. Note that if this
// returns greater than zero, it does not guarantee that calls that take
// tokens from the buffer will succeed, as the number of available
// tokens could have changed in the meantime. This method is intended
// primarily for metrics reporting and debugging.
func (tb *Bucket) Available() int64 {
	return tb.available(tb.clock.Now())
}

// available is the internal version of available - it takes the current time as
// an argument to enable easy testing.
func (tb *Bucket) available(now time.Time) int64 {
	tb.mu.Lock()
	defer tb.mu.Unlock()
	tb.adjust(now)
	return tb.avail
}

// Capacity returns the capacity that the bucket was created with.
func (tb *Bucket) Capacity() int64 {
	return tb.capacity
}

// Rate returns the fill rate of the bucket, in tokens per second.
func (tb *Bucket) Rate() float64 {
	return 1e9 * float64(tb.quantum) / float64(tb.fillInterval)
}

// take is the internal version of Take - it takes the current time as
// an argument to enable easy testing.
func (tb *Bucket) take(now time.Time, count int64, maxWait time.Duration) (time.Duration, bool) {
	if count <= 0 {
		return 0, true
	}
	tb.mu.Lock()
	defer tb.mu.Unlock()

	currentTick := tb.adjust(now)
	avail := tb.avail - count
	if avail >= 0 {
		tb.avail = avail
		return 0, true
	}
	// Round up the missing tokens to the nearest multiple
	// of quantum - the tokens won't be available until
	// that tick.
	endTick := currentTick + (-avail+tb.quantum-1)/tb.quantum
	endTime := tb.startTime.Add(time.Duration(endTick) * tb.fillInterval)
	waitTime := endTime.Sub(now)
	if waitTime > maxWait {
		return 0, false
	}
	tb.avail = avail
	return waitTime, true
}

// adjust adjusts the current bucket capacity based on the current time.
// It returns the current tick.
func (tb *Bucket) adjust(now time.Time) (currentTick int64) {
	currentTick = int64(now.Sub(tb.startTime) / tb.fillInterval)

	if tb.avail >= tb.capacity {
		return
	}
	tb.avail += (currentTick - tb.availTick) * tb.quantum
	if tb.avail > tb.capacity {
		tb.avail = tb.capacity
	}
	tb.availTick = currentTick
	return
}
