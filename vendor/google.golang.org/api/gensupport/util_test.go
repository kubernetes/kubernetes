// Copyright 2016 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"io"
	"time"
)

// errReader reads out of a buffer until it is empty, then returns the specified error.
type errReader struct {
	buf []byte
	err error
}

func (er *errReader) Read(p []byte) (int, error) {
	if len(er.buf) == 0 {
		if er.err == nil {
			return 0, io.EOF
		}
		return 0, er.err
	}
	n := copy(p, er.buf)
	er.buf = er.buf[n:]
	return n, nil
}

// UniformPauseStrategy implements BackoffStrategy with uniform pause.
type UniformPauseStrategy time.Duration

func (p UniformPauseStrategy) Pause() (time.Duration, bool) { return time.Duration(p), true }
func (p UniformPauseStrategy) Reset()                       {}

// NoPauseStrategy implements BackoffStrategy with infinite 0-length pauses.
const NoPauseStrategy = UniformPauseStrategy(0)

// LimitRetryStrategy wraps a BackoffStrategy but limits the number of retries.
type LimitRetryStrategy struct {
	Max      int
	Strategy BackoffStrategy
	n        int
}

func (l *LimitRetryStrategy) Pause() (time.Duration, bool) {
	l.n++
	if l.n > l.Max {
		return 0, false
	}
	return l.Strategy.Pause()
}

func (l *LimitRetryStrategy) Reset() {
	l.n = 0
	l.Strategy.Reset()
}
