// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package event provides an infinitely buffered double-ended queue of events.
package event // import "golang.org/x/exp/shiny/driver/internal/event"

import (
	"sync"
)

// Deque is an infinitely buffered double-ended queue of events. The zero value
// is usable, but a Deque value must not be copied.
type Deque struct {
	mu    sync.Mutex
	cond  sync.Cond     // cond.L is lazily initialized to &Deque.mu.
	back  []interface{} // FIFO.
	front []interface{} // LIFO.
}

// NextEvent implements the screen.EventQueue interface.
func (q *Deque) NextEvent() interface{} {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.cond.L == nil {
		q.cond.L = &q.mu
	}

	for {
		if n := len(q.front); n > 0 {
			e := q.front[n-1]
			q.front = q.front[:n-1]
			return e
		}

		if n := len(q.back); n > 0 {
			e := q.back[0]
			q.back = q.back[1:]
			return e
		}

		q.cond.Wait()
	}
}

// Send implements the screen.EventQueue interface.
func (q *Deque) Send(event interface{}) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.cond.L == nil {
		q.cond.L = &q.mu
	}

	q.back = append(q.back, event)
	q.cond.Signal()
}

// SendFirst implements the screen.EventQueue interface.
func (q *Deque) SendFirst(event interface{}) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.cond.L == nil {
		q.cond.L = &q.mu
	}

	q.front = append(q.front, event)
	q.cond.Signal()
}
