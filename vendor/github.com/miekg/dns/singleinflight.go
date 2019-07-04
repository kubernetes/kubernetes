// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Adapted for dns package usage by Miek Gieben.

package dns

import "sync"
import "time"

// call is an in-flight or completed singleflight.Do call
type call struct {
	wg   sync.WaitGroup
	val  *Msg
	rtt  time.Duration
	err  error
	dups int
}

// singleflight represents a class of work and forms a namespace in
// which units of work can be executed with duplicate suppression.
type singleflight struct {
	sync.Mutex                  // protects m
	m          map[string]*call // lazily initialized

	dontDeleteForTesting bool // this is only to be used by TestConcurrentExchanges
}

// Do executes and returns the results of the given function, making
// sure that only one execution is in-flight for a given key at a
// time. If a duplicate comes in, the duplicate caller waits for the
// original to complete and receives the same results.
// The return value shared indicates whether v was given to multiple callers.
func (g *singleflight) Do(key string, fn func() (*Msg, time.Duration, error)) (v *Msg, rtt time.Duration, err error, shared bool) {
	g.Lock()
	if g.m == nil {
		g.m = make(map[string]*call)
	}
	if c, ok := g.m[key]; ok {
		c.dups++
		g.Unlock()
		c.wg.Wait()
		return c.val, c.rtt, c.err, true
	}
	c := new(call)
	c.wg.Add(1)
	g.m[key] = c
	g.Unlock()

	c.val, c.rtt, c.err = fn()
	c.wg.Done()

	if !g.dontDeleteForTesting {
		g.Lock()
		delete(g.m, key)
		g.Unlock()
	}

	return c.val, c.rtt, c.err, c.dups > 0
}
