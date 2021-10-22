// Copyright 2016 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bundler supports bundling (batching) of items. Bundling amortizes an
// action with fixed costs over multiple items. For example, if an API provides
// an RPC that accepts a list of items as input, but clients would prefer
// adding items one at a time, then a Bundler can accept individual items from
// the client and bundle many of them into a single RPC.
//
// This package is experimental and subject to change without notice.
package bundler

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"time"

	"golang.org/x/sync/semaphore"
)

type mode int

const (
	DefaultDelayThreshold       = time.Second
	DefaultBundleCountThreshold = 10
	DefaultBundleByteThreshold  = 1e6 // 1M
	DefaultBufferedByteLimit    = 1e9 // 1G
)

const (
	none mode = iota
	add
	addWait
)

var (
	// ErrOverflow indicates that Bundler's stored bytes exceeds its BufferedByteLimit.
	ErrOverflow = errors.New("bundler reached buffered byte limit")

	// ErrOversizedItem indicates that an item's size exceeds the maximum bundle size.
	ErrOversizedItem = errors.New("item size exceeds bundle byte limit")

	// errMixedMethods indicates that mutually exclusive methods has been
	// called subsequently.
	errMixedMethods = errors.New("calls to Add and AddWait cannot be mixed")
)

// A Bundler collects items added to it into a bundle until the bundle
// exceeds a given size, then calls a user-provided function to handle the
// bundle.
//
// The exported fields are only safe to modify prior to the first call to Add
// or AddWait.
type Bundler struct {
	// Starting from the time that the first message is added to a bundle, once
	// this delay has passed, handle the bundle. The default is DefaultDelayThreshold.
	DelayThreshold time.Duration

	// Once a bundle has this many items, handle the bundle. Since only one
	// item at a time is added to a bundle, no bundle will exceed this
	// threshold, so it also serves as a limit. The default is
	// DefaultBundleCountThreshold.
	BundleCountThreshold int

	// Once the number of bytes in current bundle reaches this threshold, handle
	// the bundle. The default is DefaultBundleByteThreshold. This triggers handling,
	// but does not cap the total size of a bundle.
	BundleByteThreshold int

	// The maximum size of a bundle, in bytes. Zero means unlimited.
	BundleByteLimit int

	// The maximum number of bytes that the Bundler will keep in memory before
	// returning ErrOverflow. The default is DefaultBufferedByteLimit.
	BufferedByteLimit int

	// The maximum number of handler invocations that can be running at once.
	// The default is 1.
	HandlerLimit int

	handler       func(interface{}) // called to handle a bundle
	itemSliceZero reflect.Value     // nil (zero value) for slice of items

	mu           sync.Mutex          // guards access to fields below
	flushTimer   *time.Timer         // implements DelayThreshold
	handlerCount int                 // # of bundles currently being handled (i.e. handler is invoked on them)
	sem          *semaphore.Weighted // enforces BufferedByteLimit
	semOnce      sync.Once           // guards semaphore initialization
	// The current bundle we're adding items to. Not yet in the queue.
	// Appended to the queue once the flushTimer fires or the bundle
	// thresholds/limits are reached. If curBundle is nil and tail is
	// not, we first try to add items to tail. Once tail is full or handled,
	// we create a new curBundle for the incoming item.
	curBundle *bundle
	// The next bundle in the queue to be handled. Nil if the queue is
	// empty.
	head *bundle
	// The last bundle in the queue to be handled. Nil if the queue is
	// empty. If curBundle is nil and tail isn't, we attempt to add new
	// items to the tail until if becomes full or has been passed to the
	// handler.
	tail      *bundle
	curFlush  *sync.WaitGroup // counts outstanding bundles since last flush
	prevFlush chan bool       // signal used to wait for prior flush

	// The first call to Add or AddWait, mode will be add or addWait respectively.
	// If there wasn't call yet then mode is none.
	mode mode
	// TODO: consider alternative queue implementation for head/tail bundle. see:
	// https://code-review.googlesource.com/c/google-api-go-client/+/47991/4/support/bundler/bundler.go#74
}

// A bundle is a group of items that were added individually and will be passed
// to a handler as a slice.
type bundle struct {
	items reflect.Value   // slice of T
	size  int             // size in bytes of all items
	next  *bundle         // bundles are handled in order as a linked list queue
	flush *sync.WaitGroup // the counter that tracks flush completion
}

// add appends item to this bundle and increments the total size. It requires
// that b.mu is locked.
func (bu *bundle) add(item interface{}, size int) {
	bu.items = reflect.Append(bu.items, reflect.ValueOf(item))
	bu.size += size
}

// NewBundler creates a new Bundler.
//
// itemExample is a value of the type that will be bundled. For example, if you
// want to create bundles of *Entry, you could pass &Entry{} for itemExample.
//
// handler is a function that will be called on each bundle. If itemExample is
// of type T, the argument to handler is of type []T. handler is always called
// sequentially for each bundle, and never in parallel.
//
// Configure the Bundler by setting its thresholds and limits before calling
// any of its methods.
func NewBundler(itemExample interface{}, handler func(interface{})) *Bundler {
	b := &Bundler{
		DelayThreshold:       DefaultDelayThreshold,
		BundleCountThreshold: DefaultBundleCountThreshold,
		BundleByteThreshold:  DefaultBundleByteThreshold,
		BufferedByteLimit:    DefaultBufferedByteLimit,
		HandlerLimit:         1,

		handler:       handler,
		itemSliceZero: reflect.Zero(reflect.SliceOf(reflect.TypeOf(itemExample))),
		curFlush:      &sync.WaitGroup{},
	}
	return b
}

func (b *Bundler) initSemaphores() {
	// Create the semaphores lazily, because the user may set limits
	// after NewBundler.
	b.semOnce.Do(func() {
		b.sem = semaphore.NewWeighted(int64(b.BufferedByteLimit))
	})
}

// enqueueCurBundle moves curBundle to the end of the queue. The bundle may be
// handled immediately if we are below HandlerLimit. It requires that b.mu is
// locked.
func (b *Bundler) enqueueCurBundle() {
	// We don't require callers to check if there is a pending bundle. It
	// may have already been appended to the queue. If so, return early.
	if b.curBundle == nil {
		return
	}
	// If we are below the HandlerLimit, the queue must be empty. Handle
	// immediately with a new goroutine.
	if b.handlerCount < b.HandlerLimit {
		b.handlerCount++
		go b.handle(b.curBundle)
	} else if b.tail != nil {
		// There are bundles on the queue, so append to the end
		b.tail.next = b.curBundle
		b.tail = b.curBundle
	} else {
		// The queue is empty, so initialize the queue
		b.head = b.curBundle
		b.tail = b.curBundle
	}
	b.curBundle = nil
	if b.flushTimer != nil {
		b.flushTimer.Stop()
		b.flushTimer = nil
	}
}

// setMode sets the state of Bundler's mode. If mode was defined before
// and passed state is different from it then return an error.
func (b *Bundler) setMode(m mode) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.mode == m || b.mode == none {
		b.mode = m
		return nil
	}
	return errMixedMethods
}

// canFit returns true if bu can fit an additional item of size bytes based
// on the limits of Bundler b.
func (b *Bundler) canFit(bu *bundle, size int) bool {
	return (b.BundleByteLimit <= 0 || bu.size+size <= b.BundleByteLimit) &&
		(b.BundleCountThreshold <= 0 || bu.items.Len() < b.BundleCountThreshold)
}

// Add adds item to the current bundle. It marks the bundle for handling and
// starts a new one if any of the thresholds or limits are exceeded.
//
// If the item's size exceeds the maximum bundle size (Bundler.BundleByteLimit), then
// the item can never be handled. Add returns ErrOversizedItem in this case.
//
// If adding the item would exceed the maximum memory allowed
// (Bundler.BufferedByteLimit) or an AddWait call is blocked waiting for
// memory, Add returns ErrOverflow.
//
// Add never blocks.
func (b *Bundler) Add(item interface{}, size int) error {
	if err := b.setMode(add); err != nil {
		return err
	}
	// If this item exceeds the maximum size of a bundle,
	// we can never send it.
	if b.BundleByteLimit > 0 && size > b.BundleByteLimit {
		return ErrOversizedItem
	}

	// If adding this item would exceed our allotted memory
	// footprint, we can't accept it.
	// (TryAcquire also returns false if anything is waiting on the semaphore,
	// so calls to Add and AddWait shouldn't be mixed.)
	b.initSemaphores()
	if !b.sem.TryAcquire(int64(size)) {
		return ErrOverflow
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	return b.add(item, size)
}

// add adds item to the tail of the bundle queue or curBundle depending on space
// and nil-ness (see inline comments). It marks curBundle for handling (by
// appending it to the queue) if any of the thresholds or limits are exceeded.
// curBundle is lazily initialized. It requires that b.mu is locked.
func (b *Bundler) add(item interface{}, size int) error {
	// If we don't have a curBundle, see if we can add to the queue tail.
	if b.tail != nil && b.curBundle == nil && b.canFit(b.tail, size) {
		b.tail.add(item, size)
		return nil
	}

	// If we can't fit in the existing curBundle, move it onto the queue.
	if b.curBundle != nil && !b.canFit(b.curBundle, size) {
		b.enqueueCurBundle()
	}

	// Create a curBundle if we don't have one.
	if b.curBundle == nil {
		b.curFlush.Add(1)
		b.curBundle = &bundle{
			items: b.itemSliceZero,
			flush: b.curFlush,
		}
	}

	// Add the item.
	b.curBundle.add(item, size)

	// If curBundle is ready for handling, move it to the queue.
	if b.curBundle.size >= b.BundleByteThreshold ||
		b.curBundle.items.Len() == b.BundleCountThreshold {
		b.enqueueCurBundle()
	}

	// If we created a new bundle and it wasn't immediately handled, set a timer
	if b.curBundle != nil && b.flushTimer == nil {
		b.flushTimer = time.AfterFunc(b.DelayThreshold, b.tryHandleBundles)
	}

	return nil
}

// tryHandleBundles is the timer callback that handles or queues any current
// bundle after DelayThreshold time, even if the bundle isn't completely full.
func (b *Bundler) tryHandleBundles() {
	b.mu.Lock()
	b.enqueueCurBundle()
	b.mu.Unlock()
}

// next returns the next bundle that is ready for handling and removes it from
// the internal queue. It requires that b.mu is locked.
func (b *Bundler) next() *bundle {
	if b.head == nil {
		return nil
	}
	out := b.head
	b.head = b.head.next
	if b.head == nil {
		b.tail = nil
	}
	out.next = nil
	return out
}

// handle calls the user-specified handler on the given bundle. handle is
// intended to be run as a goroutine. After the handler returns, we update the
// byte total. handle continues processing additional bundles that are ready.
// If no more bundles are ready, the handler count is decremented and the
// goroutine ends.
func (b *Bundler) handle(bu *bundle) {
	for bu != nil {
		b.handler(bu.items.Interface())
		bu = b.postHandle(bu)
	}
	b.mu.Lock()
	b.handlerCount--
	b.mu.Unlock()
}

func (b *Bundler) postHandle(bu *bundle) *bundle {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.sem.Release(int64(bu.size))
	bu.flush.Done()
	return b.next()
}

// AddWait adds item to the current bundle. It marks the bundle for handling and
// starts a new one if any of the thresholds or limits are exceeded.
//
// If the item's size exceeds the maximum bundle size (Bundler.BundleByteLimit), then
// the item can never be handled. AddWait returns ErrOversizedItem in this case.
//
// If adding the item would exceed the maximum memory allowed (Bundler.BufferedByteLimit),
// AddWait blocks until space is available or ctx is done.
//
// Calls to Add and AddWait should not be mixed on the same Bundler.
func (b *Bundler) AddWait(ctx context.Context, item interface{}, size int) error {
	if err := b.setMode(addWait); err != nil {
		return err
	}
	// If this item exceeds the maximum size of a bundle,
	// we can never send it.
	if b.BundleByteLimit > 0 && size > b.BundleByteLimit {
		return ErrOversizedItem
	}
	// If adding this item would exceed our allotted memory footprint, block
	// until space is available. The semaphore is FIFO, so there will be no
	// starvation.
	b.initSemaphores()
	if err := b.sem.Acquire(ctx, int64(size)); err != nil {
		return err
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	return b.add(item, size)
}

// Flush invokes the handler for all remaining items in the Bundler and waits
// for it to return.
func (b *Bundler) Flush() {
	b.mu.Lock()

	// If a curBundle is pending, move it to the queue.
	b.enqueueCurBundle()

	// Store a pointer to the WaitGroup that counts outstanding bundles
	// in the current flush and create a new one to track the next flush.
	wg := b.curFlush
	b.curFlush = &sync.WaitGroup{}

	// Flush must wait for all prior, outstanding flushes to complete.
	// We use a channel to communicate completion between each flush in
	// the sequence.
	prev := b.prevFlush
	next := make(chan bool)
	b.prevFlush = next

	b.mu.Unlock()

	// Wait until the previous flush is finished.
	if prev != nil {
		<-prev
	}

	// Wait until this flush is finished.
	wg.Wait()

	// Allow the next flush to finish.
	close(next)
}
