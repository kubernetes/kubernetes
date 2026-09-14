// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rate

import (
	"sync"
	"time"
)

// Sometimes will perform an action occasionally.  The First, Every, and
// Interval fields govern the behavior of Do, which performs the action.
// A zero Sometimes value will perform an action exactly once.
//
// # Example: logging with rate limiting
//
//	var sometimes = rate.Sometimes{First: 3, Interval: 10*time.Second}
//	func Spammy() {
//	        sometimes.Do(func() { log.Info("here I am!") })
//	}
type Sometimes struct {
	First    int           // if non-zero, the first N calls to Do will run f.
	Every    int           // if non-zero, every Nth call to Do will run f.
	Interval time.Duration // if non-zero and Interval has elapsed since f's last run, Do will run f.

	mu    sync.Mutex
	count int       // number of Do calls
	last  time.Time // last time f was run
}

// Do runs the function f as allowed by First, Every, and Interval.
//
// The model is a union (not intersection) of filters.  The first call to Do
// always runs f.  Subsequent calls to Do run f if allowed by First or Every or
// Interval.
//
// A non-zero First:N causes the first N Do(f) calls to run f.
//
// A non-zero Every:M causes every Mth Do(f) call, starting with the first, to
// run f.
//
// A non-zero Interval causes Do(f) to run f if Interval has elapsed since
// Do last ran f.
//
// Specifying multiple filters produces the union of these execution streams.
// For example, specifying both First:N and Every:M causes the first N Do(f)
// calls and every Mth Do(f) call, starting with the first, to run f.  See
// Examples for more.
//
// If Do is called multiple times simultaneously, the calls will block and run
// serially.  Therefore, Do is intended for lightweight operations.
//
// Because a call to Do may block until f returns, if f causes Do to be called,
// it will deadlock.
func (s *Sometimes) Do(f func()) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.count == 0 ||
		(s.First > 0 && s.count < s.First) ||
		(s.Every > 0 && s.count%s.Every == 0) ||
		(s.Interval > 0 && time.Since(s.last) >= s.Interval) {
		f()
		if s.Interval > 0 {
			s.last = time.Now()
		}
	}
	s.count++
}
