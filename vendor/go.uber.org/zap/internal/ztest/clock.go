// Copyright (c) 2023 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package ztest

import (
	"sort"
	"sync"
	"time"
)

// MockClock is a fake source of time.
// It implements standard time operations,
// but allows the user to control the passage of time.
//
// Use the [Add] method to progress time.
type MockClock struct {
	mu  sync.RWMutex
	now time.Time

	// The MockClock works by maintaining a list of waiters.
	// Each waiter knows the time at which it should be resolved.
	// When the clock advances, all waiters that are in range are resolved
	// in chronological order.
	waiters []waiter
}

// NewMockClock builds a new mock clock
// using the current actual time as the initial time.
func NewMockClock() *MockClock {
	return &MockClock{
		now: time.Now(),
	}
}

// Now reports the current time.
func (c *MockClock) Now() time.Time {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.now
}

// NewTicker returns a time.Ticker that ticks at the specified frequency.
//
// As with [time.NewTicker],
// the ticker will drop ticks if the receiver is slow,
// and the channel is never closed.
//
// Calling Stop on the returned ticker is a no-op.
// The ticker only runs when the clock is advanced.
func (c *MockClock) NewTicker(d time.Duration) *time.Ticker {
	ch := make(chan time.Time, 1)

	var tick func(time.Time)
	tick = func(now time.Time) {
		next := now.Add(d)
		c.runAt(next, func() {
			defer tick(next)

			select {
			case ch <- next:
				// ok
			default:
				// The receiver is slow.
				// Drop the tick and continue.
			}
		})
	}
	tick(c.Now())

	return &time.Ticker{C: ch}
}

// runAt schedules the given function to be run at the given time.
// The function runs without a lock held, so it may schedule more work.
func (c *MockClock) runAt(t time.Time, fn func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.waiters = append(c.waiters, waiter{until: t, fn: fn})
}

type waiter struct {
	until time.Time
	fn    func()
}

// Add progresses time by the given duration.
// Other operations waiting for the time to advance
// will be resolved if they are within range.
//
// Side effects of operations waiting for the time to advance
// will take effect on a best-effort basis.
// Avoid racing with operations that have side effects.
//
// Panics if the duration is negative.
func (c *MockClock) Add(d time.Duration) {
	if d < 0 {
		panic("cannot add negative duration")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	sort.Slice(c.waiters, func(i, j int) bool {
		return c.waiters[i].until.Before(c.waiters[j].until)
	})

	newTime := c.now.Add(d)
	// newTime won't be recorded until the end of this method.
	// This ensures that any waiters that are resolved
	// are resolved at the time they were expecting.

	for len(c.waiters) > 0 {
		w := c.waiters[0]
		if w.until.After(newTime) {
			break
		}
		c.waiters[0] = waiter{} // avoid memory leak
		c.waiters = c.waiters[1:]

		// The waiter is within range.
		// Travel to the time of the waiter and resolve it.
		c.now = w.until

		// The waiter may schedule more work
		// so we must release the lock.
		c.mu.Unlock()
		w.fn()
		// Sleeping here is necessary to let the side effects of waiters
		// take effect before we continue.
		time.Sleep(1 * time.Millisecond)
		c.mu.Lock()
	}

	c.now = newTime
}
