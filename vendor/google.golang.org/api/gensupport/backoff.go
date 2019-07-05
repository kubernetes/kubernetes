// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"math/rand"
	"time"
)

// BackoffStrategy defines the set of functions that a backoff-er must
// implement.
type BackoffStrategy interface {
	// Pause returns the duration of the next pause and true if the operation should be
	// retried, or false if no further retries should be attempted.
	Pause() (time.Duration, bool)

	// Reset restores the strategy to its initial state.
	Reset()
}

// ExponentialBackoff performs exponential backoff as per https://en.wikipedia.org/wiki/Exponential_backoff.
// The initial pause time is given by Base.
// Once the total pause time exceeds Max, Pause will indicate no further retries.
type ExponentialBackoff struct {
	Base  time.Duration
	Max   time.Duration
	total time.Duration
	n     uint
}

// Pause returns the amount of time the caller should wait.
func (eb *ExponentialBackoff) Pause() (time.Duration, bool) {
	if eb.total > eb.Max {
		return 0, false
	}

	// The next pause is selected from randomly from [0, 2^n * Base).
	d := time.Duration(rand.Int63n((1 << eb.n) * int64(eb.Base)))
	eb.total += d
	eb.n++
	return d, true
}

// Reset resets the backoff strategy such that the next Pause call will begin
// counting from the start. It is not safe to call concurrently with Pause.
func (eb *ExponentialBackoff) Reset() {
	eb.n = 0
	eb.total = 0
}
