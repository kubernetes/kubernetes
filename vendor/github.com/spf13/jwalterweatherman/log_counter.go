// Copyright © 2016 Steve Francia <spf@spf13.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package jwalterweatherman

import (
	"io"
	"sync/atomic"
)

// Counter is an io.Writer that increments a counter on Write.
type Counter struct {
	count uint64
}

func (c *Counter) incr() {
	atomic.AddUint64(&c.count, 1)
}

// Reset resets the counter.
func (c *Counter) Reset() {
	atomic.StoreUint64(&c.count, 0)
}

// Count returns the current count.
func (c *Counter) Count() uint64 {
	return atomic.LoadUint64(&c.count)
}

func (c *Counter) Write(p []byte) (n int, err error) {
	c.incr()
	return len(p), nil
}

// LogCounter creates a LogListener that counts log statements >= the given threshold.
func LogCounter(counter *Counter, t1 Threshold) LogListener {
	return func(t2 Threshold) io.Writer {
		if t2 < t1 {
			// Not interested in this threshold.
			return nil
		}
		return counter
	}
}
