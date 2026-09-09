// Package clockwork contains a simple fake clock for Go.
package clockwork

import (
	"context"
	"errors"
	"slices"
	"sync"
	"time"
)

// Clock provides an interface that packages can use instead of directly using
// the [time] module, so that chronology-related behavior can be tested.
type Clock interface {
	After(d time.Duration) <-chan time.Time
	Sleep(d time.Duration)
	Now() time.Time
	Since(t time.Time) time.Duration
	Until(t time.Time) time.Duration
	NewTicker(d time.Duration) Ticker
	NewTimer(d time.Duration) Timer
	AfterFunc(d time.Duration, f func()) Timer
}

// NewRealClock returns a Clock which simply delegates calls to the actual time
// package; it should be used by packages in production.
func NewRealClock() Clock {
	return &realClock{}
}

type realClock struct{}

func (rc *realClock) After(d time.Duration) <-chan time.Time {
	return time.After(d)
}

func (rc *realClock) Sleep(d time.Duration) {
	time.Sleep(d)
}

func (rc *realClock) Now() time.Time {
	return time.Now()
}

func (rc *realClock) Since(t time.Time) time.Duration {
	return rc.Now().Sub(t)
}

func (rc *realClock) Until(t time.Time) time.Duration {
	return t.Sub(rc.Now())
}

func (rc *realClock) NewTicker(d time.Duration) Ticker {
	return realTicker{time.NewTicker(d)}
}

func (rc *realClock) NewTimer(d time.Duration) Timer {
	return realTimer{time.NewTimer(d)}
}

func (rc *realClock) AfterFunc(d time.Duration, f func()) Timer {
	return realTimer{time.AfterFunc(d, f)}
}

// FakeClock provides an interface for a clock which can be manually advanced
// through time.
//
// FakeClock maintains a list of "waiters," which consists of all callers
// waiting on the underlying clock (i.e. Tickers and Timers including callers of
// Sleep or After). Users can call BlockUntil to block until the clock has an
// expected number of waiters.
type FakeClock struct {
	// l protects all attributes of the clock, including all attributes of all
	// waiters and blockers.
	l        sync.RWMutex
	waiters  []expirer
	blockers []*blocker
	time     time.Time
}

// NewFakeClock returns a FakeClock implementation which can be
// manually advanced through time for testing. The initial time of the
// FakeClock will be the current system time.
//
// Tests that require a deterministic time must use NewFakeClockAt.
func NewFakeClock() *FakeClock {
	return NewFakeClockAt(time.Now())
}

// NewFakeClockAt returns a FakeClock initialised at the given time.Time.
func NewFakeClockAt(t time.Time) *FakeClock {
	return &FakeClock{
		time: t,
	}
}

// blocker is a caller of BlockUntil.
type blocker struct {
	count int

	// ch is closed when the underlying clock has the specified number of blockers.
	ch chan struct{}
}

// expirer is a timer or ticker that expires at some point in the future.
type expirer interface {
	// expire the expirer at the given time, returning the desired duration until
	// the next expiration, if any.
	expire(now time.Time) (next *time.Duration)

	// Get and set the expiration time.
	expiration() time.Time
	setExpiration(time.Time)
}

// After mimics [time.After]; it waits for the given duration to elapse on the
// fakeClock, then sends the current time on the returned channel.
func (fc *FakeClock) After(d time.Duration) <-chan time.Time {
	return fc.NewTimer(d).Chan()
}

// Sleep blocks until the given duration has passed on the fakeClock.
func (fc *FakeClock) Sleep(d time.Duration) {
	<-fc.After(d)
}

// Now returns the current time of the fakeClock
func (fc *FakeClock) Now() time.Time {
	fc.l.RLock()
	defer fc.l.RUnlock()
	return fc.time
}

// Since returns the duration that has passed since the given time on the
// fakeClock.
func (fc *FakeClock) Since(t time.Time) time.Duration {
	return fc.Now().Sub(t)
}

// Until returns the duration that has to pass from the given time on the fakeClock
// to reach the given time.
func (fc *FakeClock) Until(t time.Time) time.Duration {
	return t.Sub(fc.Now())
}

// NewTicker returns a Ticker that will expire only after calls to
// FakeClock.Advance() have moved the clock past the given duration.
//
// The duration d must be greater than zero; if not, NewTicker will panic.
func (fc *FakeClock) NewTicker(d time.Duration) Ticker {
	// Maintain parity with
	// https://cs.opensource.google/go/go/+/refs/tags/go1.20.3:src/time/tick.go;l=23-25
	if d <= 0 {
		panic(errors.New("non-positive interval for NewTicker"))
	}
	ft := newFakeTicker(fc, d)
	fc.l.Lock()
	defer fc.l.Unlock()
	fc.setExpirer(ft, d)
	return ft
}

// NewTimer returns a Timer that will fire only after calls to
// fakeClock.Advance() have moved the clock past the given duration.
func (fc *FakeClock) NewTimer(d time.Duration) Timer {
	t, _ := fc.newTimer(d, nil)
	return t
}

// AfterFunc mimics [time.AfterFunc]; it returns a Timer that will invoke the
// given function only after calls to fakeClock.Advance() have moved the clock
// past the given duration.
func (fc *FakeClock) AfterFunc(d time.Duration, f func()) Timer {
	t, _ := fc.newTimer(d, f)
	return t
}

// newTimer returns a new timer using an optional afterFunc and the time that
// timer expires.
func (fc *FakeClock) newTimer(d time.Duration, afterfunc func()) (*fakeTimer, time.Time) {
	ft := newFakeTimer(fc, afterfunc)
	fc.l.Lock()
	defer fc.l.Unlock()
	fc.setExpirer(ft, d)
	return ft, ft.expiration()
}

// newTimerAtTime is like newTimer, but uses a time instead of a duration.
//
// It is used to ensure FakeClock's lock is held constant through calling
// fc.After(t.Sub(fc.Now())). It should not be exposed externally.
func (fc *FakeClock) newTimerAtTime(t time.Time, afterfunc func()) *fakeTimer {
	ft := newFakeTimer(fc, afterfunc)
	fc.l.Lock()
	defer fc.l.Unlock()
	fc.setExpirer(ft, t.Sub(fc.time))
	return ft
}

// Advance advances fakeClock to a new point in time, ensuring waiters and
// blockers are notified appropriately before returning.
func (fc *FakeClock) Advance(d time.Duration) {
	fc.l.Lock()
	defer fc.l.Unlock()
	end := fc.time.Add(d)
	// Expire the earliest waiter until the earliest waiter's expiration is after
	// end.
	//
	// We don't iterate because the callback of the waiter might register a new
	// waiter, so the list of waiters might change as we execute this.
	for len(fc.waiters) > 0 && !end.Before(fc.waiters[0].expiration()) {
		w := fc.waiters[0]
		fc.waiters = fc.waiters[1:]

		// Use the waiter's expiration as the current time for this expiration.
		now := w.expiration()
		fc.time = now
		if d := w.expire(now); d != nil {
			// Set the new expiration if needed.
			fc.setExpirer(w, *d)
		}
	}
	fc.time = end
}

// BlockUntil blocks until the FakeClock has the given number of waiters.
//
// Prefer BlockUntilContext in new code, which offers context cancellation to
// prevent deadlock.
//
// Deprecated: New code should prefer BlockUntilContext.
func (fc *FakeClock) BlockUntil(n int) {
	fc.BlockUntilContext(context.TODO(), n)
}

// BlockUntilContext blocks until the fakeClock has the given number of waiters
// or the context is cancelled.
func (fc *FakeClock) BlockUntilContext(ctx context.Context, n int) error {
	b := fc.newBlocker(n)
	if b == nil {
		return nil
	}

	select {
	case <-b.ch:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (fc *FakeClock) newBlocker(n int) *blocker {
	fc.l.Lock()
	defer fc.l.Unlock()
	// Fast path: we already have >= n waiters.
	if len(fc.waiters) >= n {
		return nil
	}
	// Set up a new blocker to wait for more waiters.
	b := &blocker{
		count: n,
		ch:    make(chan struct{}),
	}
	fc.blockers = append(fc.blockers, b)
	return b
}

// stop stops an expirer, returning true if the expirer was stopped.
func (fc *FakeClock) stop(e expirer) bool {
	fc.l.Lock()
	defer fc.l.Unlock()
	return fc.stopExpirer(e)
}

// stopExpirer stops an expirer, returning true if the expirer was stopped.
//
// The caller must hold fc.l.
func (fc *FakeClock) stopExpirer(e expirer) bool {
	idx := slices.Index(fc.waiters, e)
	if idx == -1 {
		return false
	}
	// Remove element, maintaining order, setting inaccessible elements to nil so
	// they can be garbage collected.
	copy(fc.waiters[idx:], fc.waiters[idx+1:])
	fc.waiters[len(fc.waiters)-1] = nil
	fc.waiters = fc.waiters[:len(fc.waiters)-1]
	return true
}

// setExpirer sets an expirer to expire at a future point in time.
//
// The caller must hold fc.l.
func (fc *FakeClock) setExpirer(e expirer, d time.Duration) {
	if d.Nanoseconds() <= 0 {
		// Special case for timers with duration <= 0: trigger immediately, never
		// reset.
		//
		// Tickers never get here, they panic if d is < 0.
		e.expire(fc.time)
		return
	}
	// Add the expirer to the set of waiters and notify any blockers.
	e.setExpiration(fc.time.Add(d))
	fc.waiters = append(fc.waiters, e)
	slices.SortFunc(fc.waiters, func(a, b expirer) int {
		return a.expiration().Compare(b.expiration())
	})

	// Notify blockers of our new waiter.
	count := len(fc.waiters)
	fc.blockers = slices.DeleteFunc(fc.blockers, func(b *blocker) bool {
		if b.count <= count {
			close(b.ch)
			return true
		}
		return false
	})
}
