/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package synctrack contains utilities for helping controllers track whether
// they are "synced" or not, that is, whether they have processed all items
// from the informer's initial list.
package synctrack

import (
	"context"
	"sync"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/util/sets"
)

// Synced supports waiting for some activity to finish without polling.
type Synced interface {
	// Name returns a string describing the entity that is being waited for.
	//
	// Note that this name might be computed, so callers should only
	// get the name outside of a hot code path.
	Name() string

	// Done returns a channel that will be closed on completion
	// of the activity.
	Done() <-chan struct{}
}

// AsyncTracker helps propagate HasSynced in the face of multiple worker threads.
type AsyncTracker[T comparable] struct {
	UpstreamHasSynced func() bool

	lock    sync.Mutex
	waiting sets.Set[T]

	// TODO
	// synced       chan struct{}
	// syncedClosed bool
}

// Start should be called prior to processing each key which is part of the
// initial list.
func (t *AsyncTracker[T]) Start(key T) {
	t.lock.Lock()
	defer t.lock.Unlock()
	if t.waiting == nil {
		t.waiting = sets.New[T](key)
	} else {
		t.waiting.Insert(key)
	}
}

// Finished should be called when finished processing a key which was part of
// the initial list. Since keys are tracked individually, nothing bad happens
// if you call Finished without a corresponding call to Start. This makes it
// easier to use this in combination with e.g. queues which don't make it easy
// to plumb through the isInInitialList boolean.
func (t *AsyncTracker[T]) Finished(key T) {
	t.lock.Lock()
	defer t.lock.Unlock()
	if t.waiting != nil {
		t.waiting.Delete(key)
	}
}

// HasSynced returns true if the source is synced and every key present in the
// initial list has been processed. This relies on the source not considering
// itself synced until *after* it has delivered the notification for the last
// key, and that notification handler must have called Start.
func (t *AsyncTracker[T]) HasSynced() bool {
	// Call UpstreamHasSynced first: it might take a lock, which might take
	// a significant amount of time, and we can't hold our lock while
	// waiting on that or a user is likely to get a deadlock.
	if !t.UpstreamHasSynced() {
		return false
	}
	t.lock.Lock()
	defer t.lock.Unlock()
	return t.waiting.Len() == 0
}

// // NamedHasSynced is done if the source is synced and every key present in the
// // initial list has been processed. This relies on the source not considering
// // itself synced until *after* it has delivered the notification for the last
// // key, and that notification handler must have called Start.
// func (t *AsyncTracker[T]) NamedHasSynced() Synced {
// 	if t.
// 	return &asyncTrackerSynced[T]{
// 		tracker: t,
// 	}
// }

// type asyncTrackerSynced[T comparable] struct {
// 	tracker *AsyncTracker[T]
// }

// func (s *asyncTrackerSynced[T]) Name() {
// 	return "TODO"
// }

// func (s *asyncTrackerSynced[T]) Done() <-chan struct{} {
// 	return s.tracker.synced
// }

// SingleFileTracker helps propagate HasSynced when events are processed in
// order (i.e. via a queue). The user has to monitor the upstream "has synced"
// and notify the tracker when that changes from false to true.
type SingleFileTracker struct {
	// name describes the instance.
	name string

	// Important: count is used with atomic operations so it must be 64-bit
	// aligned, otherwise atomic operations will panic. Having it at the top of
	// the struct will guarantee that, even on 32-bit arches.
	// See https://pkg.go.dev/sync/atomic#pkg-note-BUG for more information.
	count int64

	// upstreamHasSynced is changed from false (initial value) to true
	// when UpstreamHasSynced is called.
	upstreamHasSynced atomic.Bool

	// synced gets canceled once both the tracker and upstream are synced.
	// A context is convenient for this because it gives us a channel
	// and handles thread-safety.
	synced context.Context
	cancel func()
}

func NewSingleFileTracker(name string) *SingleFileTracker {
	t := &SingleFileTracker{
		name: name,
	}
	t.synced, t.cancel = context.WithCancel(context.Background())
	return t
}

// UpstreamHasSynced needs to be called at least once as soon as
// the upstream "has synced" becomes true. It tells SingleFileTracker
// that the source is synced.
func (t *SingleFileTracker) UpstreamHasSynced() {
	// Upstream is done, but we might not be yet.
	t.upstreamHasSynced.Store(true)
	if atomic.LoadInt64(&t.count) == 0 {
		// Mark as synced.
		t.cancel()
	}
}

// Start should be called prior to processing each key which is part of the
// initial list.
func (t *SingleFileTracker) Start() {
	atomic.AddInt64(&t.count, 1)
}

// Finished should be called when finished processing a key which was part of
// the initial list. You must never call Finished() before (or without) its
// corresponding Start(), that is a logic error that could cause HasSynced to
// return a wrong value. To help you notice this should it happen, Finished()
// will panic if the internal counter goes negative.
func (t *SingleFileTracker) Finished() {
	result := atomic.AddInt64(&t.count, -1)
	if result < 0 {
		panic("synctrack: negative counter; this logic error means HasSynced may return incorrect value")
	}

	// Not synced yet before, but maybe now?
	if result == 0 && t.synced.Err() == nil {
		// We are done, but upstream might not be.
		if t.upstreamHasSynced.Load() {
			// Mark as synced.
			t.cancel()
		}
	}
}

// HasSynced returns true if the source is synced and every key present in the
// initial list has been processed. This relies on the source not considering
// itself synced until *after* it has delivered the notification for the last
// key, and that notification handler must have called Start.
func (t *SingleFileTracker) HasSynced() bool {
	return t.synced.Err() != nil
}

// Done returns a channel that is closed if the source is synced and every key present in the
// initial list has been processed. This relies on the source not considering
// itself synced until *after* it has delivered the notification for the last
// key, and that notification handler must have called Start.
func (t *SingleFileTracker) Done() <-chan struct{} {
	return t.synced.Done()
}

func (t *SingleFileTracker) Name() string {
	return t.name
}
