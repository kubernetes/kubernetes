// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package bundler supports bundling (batching) of items. Bundling amortizes an
// action with fixed costs over multiple items. For example, if an API provides
// an RPC that accepts a list of items as input, but clients would prefer
// adding items one at a time, then a Bundler can accept individual items from
// the client and bundle many of them into a single RPC.
//
// This package is experimental and subject to change without notice.
package bundler

import (
	"errors"
	"math"
	"reflect"
	"sync"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/sync/semaphore"
)

const (
	DefaultDelayThreshold       = time.Second
	DefaultBundleCountThreshold = 10
	DefaultBundleByteThreshold  = 1e6 // 1M
	DefaultBufferedByteLimit    = 1e9 // 1G
)

var (
	// ErrOverflow indicates that Bundler's stored bytes exceeds its BufferedByteLimit.
	ErrOverflow = errors.New("bundler reached buffered byte limit")

	// ErrOversizedItem indicates that an item's size exceeds the maximum bundle size.
	ErrOversizedItem = errors.New("item size exceeds bundle byte limit")
)

// A Bundler collects items added to it into a bundle until the bundle
// exceeds a given size, then calls a user-provided function to handle the bundle.
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
	flushTimer    *time.Timer       // implements DelayThreshold

	mu        sync.Mutex
	sem       *semaphore.Weighted // enforces BufferedByteLimit
	semOnce   sync.Once
	curBundle bundle // incoming items added to this bundle

	// Each bundle is assigned a unique ticket that determines the order in which the
	// handler is called. The ticket is assigned with mu locked, but waiting for tickets
	// to be handled is done via mu2 and cond, below.
	nextTicket uint64 // next ticket to be assigned

	mu2         sync.Mutex
	cond        *sync.Cond
	nextHandled uint64 // next ticket to be handled

	// In this implementation, active uses space proportional to HandlerLimit, and
	// waitUntilAllHandled takes time proportional to HandlerLimit each time an acquire
	// or release occurs, so large values of HandlerLimit max may cause performance
	// issues.
	active map[uint64]bool // tickets of bundles actively being handled
}

type bundle struct {
	items reflect.Value // slice of item type
	size  int           // size in bytes of all items
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
		active:        map[uint64]bool{},
	}
	b.curBundle.items = b.itemSliceZero
	b.cond = sync.NewCond(&b.mu2)
	return b
}

func (b *Bundler) initSemaphores() {
	// Create the semaphores lazily, because the user may set limits
	// after NewBundler.
	b.semOnce.Do(func() {
		b.sem = semaphore.NewWeighted(int64(b.BufferedByteLimit))
	})
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
	b.add(item, size)
	return nil
}

// add adds item to the current bundle. It marks the bundle for handling and
// starts a new one if any of the thresholds or limits are exceeded.
func (b *Bundler) add(item interface{}, size int) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// If adding this item to the current bundle would cause it to exceed the
	// maximum bundle size, close the current bundle and start a new one.
	if b.BundleByteLimit > 0 && b.curBundle.size+size > b.BundleByteLimit {
		b.startFlushLocked()
	}
	// Add the item.
	b.curBundle.items = reflect.Append(b.curBundle.items, reflect.ValueOf(item))
	b.curBundle.size += size

	// Start a timer to flush the item if one isn't already running.
	// startFlushLocked clears the timer and closes the bundle at the same time,
	// so we only allocate a new timer for the first item in each bundle.
	// (We could try to call Reset on the timer instead, but that would add a lot
	// of complexity to the code just to save one small allocation.)
	if b.flushTimer == nil {
		b.flushTimer = time.AfterFunc(b.DelayThreshold, b.Flush)
	}

	// If the current bundle equals the count threshold, close it.
	if b.curBundle.items.Len() == b.BundleCountThreshold {
		b.startFlushLocked()
	}
	// If the current bundle equals or exceeds the byte threshold, close it.
	if b.curBundle.size >= b.BundleByteThreshold {
		b.startFlushLocked()
	}
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
	// Here, we've reserved space for item. Other goroutines can call AddWait
	// and even acquire space, but no one can take away our reservation
	// (assuming sem.Release is used correctly). So there is no race condition
	// resulting from locking the mutex after sem.Acquire returns.
	b.add(item, size)
	return nil
}

// Flush invokes the handler for all remaining items in the Bundler and waits
// for it to return.
func (b *Bundler) Flush() {
	b.mu.Lock()
	b.startFlushLocked()
	// Here, all bundles with tickets < b.nextTicket are
	// either finished or active. Those are the ones
	// we want to wait for.
	t := b.nextTicket
	b.mu.Unlock()
	b.initSemaphores()
	b.waitUntilAllHandled(t)
}

func (b *Bundler) startFlushLocked() {
	if b.flushTimer != nil {
		b.flushTimer.Stop()
		b.flushTimer = nil
	}
	if b.curBundle.items.Len() == 0 {
		return
	}
	// Here, both semaphores must have been initialized.
	bun := b.curBundle
	b.curBundle = bundle{items: b.itemSliceZero}
	ticket := b.nextTicket
	b.nextTicket++
	go func() {
		defer func() {
			b.sem.Release(int64(bun.size))
			b.release(ticket)
		}()
		b.acquire(ticket)
		b.handler(bun.items.Interface())
	}()
}

// acquire blocks until ticket is the next to be served, then returns. In order for N
// acquire calls to return, the tickets must be in the range [0, N). A ticket must
// not be presented to acquire more than once.
func (b *Bundler) acquire(ticket uint64) {
	b.mu2.Lock()
	defer b.mu2.Unlock()
	if ticket < b.nextHandled {
		panic("bundler: acquire: arg too small")
	}
	for !(ticket == b.nextHandled && len(b.active) < b.HandlerLimit) {
		b.cond.Wait()
	}
	// Here,
	// ticket == b.nextHandled: the caller is the next one to be handled;
	// and len(b.active) < b.HandlerLimit: there is space available.
	b.active[ticket] = true
	b.nextHandled++
	// Broadcast, not Signal: although at most one acquire waiter can make progress,
	// there might be waiters in waitUntilAllHandled.
	b.cond.Broadcast()
}

// If a ticket is used for a call to acquire, it must later be passed to release. A
// ticket must not be presented to release more than once.
func (b *Bundler) release(ticket uint64) {
	b.mu2.Lock()
	defer b.mu2.Unlock()
	if !b.active[ticket] {
		panic("bundler: release: not an active ticket")
	}
	delete(b.active, ticket)
	b.cond.Broadcast()
}

// waitUntilAllHandled blocks until all tickets < n have called release, meaning
// all bundles with tickets < n have been handled.
func (b *Bundler) waitUntilAllHandled(n uint64) {
	// Proof of correctness of this function.
	// "N is acquired" means acquire(N) has returned.
	// "N is released" means release(N) has returned.
	// 1. If N is acquired, N-1 is acquired.
	//    Follows from the loop test in acquire, and the fact
	//    that nextHandled is incremented by 1.
	// 2. If nextHandled >= N, then N-1 is acquired.
	//    Because we only increment nextHandled to N after N-1 is acquired.
	// 3. If nextHandled >= N, then all n < N is acquired.
	//    Follows from #1 and #2.
	// 4. If N is acquired and N is not in active, then N is released.
	//    Because we put N in active before acquire returns, and only
	//    remove it when it is released.
	// Let min(active) be the smallest member of active, or infinity if active is empty.
	// 5. If nextHandled >= N and N <= min(active), then all n < N is released.
	//    From nextHandled >= N and #3, all n < N is acquired.
	//    N <= min(active) implies n < min(active) for all n < N. So all n < N is not in active.
	//    So from #4, all n < N is released.
	// The loop test below is the antecedent of #5.
	b.mu2.Lock()
	defer b.mu2.Unlock()
	for !(b.nextHandled >= n && n <= min(b.active)) {
		b.cond.Wait()
	}
}

// min returns the minimum value of the set s, or the largest uint64 if
// s is empty.
func min(s map[uint64]bool) uint64 {
	var m uint64 = math.MaxUint64
	for n := range s {
		if n < m {
			m = n
		}
	}
	return m
}
