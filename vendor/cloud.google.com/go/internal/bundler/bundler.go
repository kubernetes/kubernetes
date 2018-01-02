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
package bundler

import (
	"errors"
	"reflect"
	"sync"
	"time"
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

	handler       func(interface{}) // called to handle a bundle
	itemSliceZero reflect.Value     // nil (zero value) for slice of items
	donec         chan struct{}     // closed when the Bundler is closed
	handlec       chan int          // sent to when a bundle is ready for handling
	timer         *time.Timer       // implements DelayThreshold

	mu            sync.Mutex
	bufferedSize  int           // total bytes buffered
	closedBundles []bundle      // bundles waiting to be handled
	curBundle     bundle        // incoming items added to this bundle
	calledc       chan struct{} // closed and re-created after handler is called
}

type bundle struct {
	items reflect.Value // slice of item type
	size  int           // size in bytes of all items
}

// NewBundler creates a new Bundler. When you are finished with a Bundler, call
// its Close method.
//
// itemExample is a value of the type that will be bundled. For example, if you
// want to create bundles of *Entry, you could pass &Entry{} for itemExample.
//
// handler is a function that will be called on each bundle. If itemExample is
// of type T, the the argument to handler is of type []T.
func NewBundler(itemExample interface{}, handler func(interface{})) *Bundler {
	b := &Bundler{
		DelayThreshold:       DefaultDelayThreshold,
		BundleCountThreshold: DefaultBundleCountThreshold,
		BundleByteThreshold:  DefaultBundleByteThreshold,
		BufferedByteLimit:    DefaultBufferedByteLimit,

		handler:       handler,
		itemSliceZero: reflect.Zero(reflect.SliceOf(reflect.TypeOf(itemExample))),
		donec:         make(chan struct{}),
		handlec:       make(chan int, 1),
		calledc:       make(chan struct{}),
		timer:         time.NewTimer(1000 * time.Hour), // harmless initial timeout
	}
	b.curBundle.items = b.itemSliceZero
	go b.background()
	return b
}

// Add adds item to the current bundle. It marks the bundle for handling and
// starts a new one if any of the thresholds or limits are exceeded.
//
// If the item's size exceeds the maximum bundle size (Bundler.BundleByteLimit), then
// the item can never be handled. Add returns ErrOversizedItem in this case.
//
// If adding the item would exceed the maximum memory allowed (Bundler.BufferedByteLimit),
// Add returns ErrOverflow.
//
// Add never blocks.
func (b *Bundler) Add(item interface{}, size int) error {
	// If this item exceeds the maximum size of a bundle,
	// we can never send it.
	if b.BundleByteLimit > 0 && size > b.BundleByteLimit {
		return ErrOversizedItem
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	// If adding this item would exceed our allotted memory
	// footprint, we can't accept it.
	if b.bufferedSize+size > b.BufferedByteLimit {
		return ErrOverflow
	}
	// If adding this item to the current bundle would cause it to exceed the
	// maximum bundle size, close the current bundle and start a new one.
	if b.BundleByteLimit > 0 && b.curBundle.size+size > b.BundleByteLimit {
		b.closeAndHandleBundle()
	}
	// Add the item.
	b.curBundle.items = reflect.Append(b.curBundle.items, reflect.ValueOf(item))
	b.curBundle.size += size
	b.bufferedSize += size
	// If this is the first item in the bundle, restart the timer.
	if b.curBundle.items.Len() == 1 {
		b.timer.Reset(b.DelayThreshold)
	}
	// If the current bundle equals the count threshold, close it.
	if b.curBundle.items.Len() == b.BundleCountThreshold {
		b.closeAndHandleBundle()
	}
	// If the current bundle equals or exceeds the byte threshold, close it.
	if b.curBundle.size >= b.BundleByteThreshold {
		b.closeAndHandleBundle()
	}
	return nil
}

// Flush waits until all items in the Bundler have been handled.
func (b *Bundler) Flush() {
	b.mu.Lock()
	b.closeBundle()
	// Unconditionally trigger the handling goroutine, to ensure calledc is closed
	// even if there are no outstanding bundles.
	select {
	case b.handlec <- 1:
	default:
	}
	calledc := b.calledc // remember locally, because it may change
	b.mu.Unlock()
	<-calledc
}

// Close calls Flush, then shuts down the Bundler. Close should always be
// called on a Bundler when it is no longer needed. You must wait for all calls
// to Add to complete before calling Close. Calling Add concurrently with Close
// may result in the added items being ignored.
func (b *Bundler) Close() {
	b.Flush()
	b.mu.Lock()
	b.timer.Stop()
	b.mu.Unlock()
	close(b.donec)
}

func (b *Bundler) closeAndHandleBundle() {
	if b.closeBundle() {
		// We have created a closed bundle.
		// Send to handlec without blocking.
		select {
		case b.handlec <- 1:
		default:
		}
	}
}

// closeBundle finishes the current bundle, adds it to the list of closed
// bundles and informs the background goroutine that there are bundles ready
// for processing.
//
// This should always be called with b.mu held.
func (b *Bundler) closeBundle() bool {
	if b.curBundle.items.Len() == 0 {
		return false
	}
	b.closedBundles = append(b.closedBundles, b.curBundle)
	b.curBundle.items = b.itemSliceZero
	b.curBundle.size = 0
	return true
}

// background runs in a separate goroutine, waiting for events and handling
// bundles.
func (b *Bundler) background() {
	done := false
	for {
		timedOut := false
		// Wait for something to happen.
		select {
		case <-b.handlec:
		case <-b.donec:
			done = true
		case <-b.timer.C:
			timedOut = true
		}
		// Handle closed bundles.
		b.mu.Lock()
		if timedOut {
			b.closeBundle()
		}
		buns := b.closedBundles
		b.closedBundles = nil
		// Closing calledc means we've sent all bundles. We need
		// a new channel for the next set of bundles, which may start
		// accumulating as soon as we release the lock.
		calledc := b.calledc
		b.calledc = make(chan struct{})
		b.mu.Unlock()
		for i, bun := range buns {
			b.handler(bun.items.Interface())
			// Drop the bundle's items, reducing our memory footprint.
			buns[i].items = reflect.Value{} // buns[i] because bun is a copy
			// Note immediately that we have more space, so Adds that occur
			// during this loop will have a chance of succeeding.
			b.mu.Lock()
			b.bufferedSize -= bun.size
			b.mu.Unlock()
		}
		// Signal that we've sent all outstanding bundles.
		close(calledc)
		if done {
			break
		}
	}
}
