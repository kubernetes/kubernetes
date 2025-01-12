/*
Copyright 2025 The Kubernetes Authors.

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

package cache

import (
	"fmt"
	"k8s.io/apimachinery/pkg/util/sets"
	"sync"
)

// RealFIFO is a Queue in which every notification from the Reflector is passed
// in order to the Queue via Pop.
// This means that it
// 1. delivers notifications for items that have been deleted
// 2. delivers multiple notifications per item instead of simply the most recent value
type RealFIFO struct {
	lock sync.RWMutex
	cond sync.Cond

	items []Delta

	// knownObjects list keys that are "known" --- affecting Delete(),
	// Replace(), and Resync()
	knownObjects KeyListerGetter

	// TODO restore if the experiment goes decently
	// Called with every object if non-nil.
	// Looks like a real API mistake that I bet someone wanted to use to reduce cache size but placed in the wrong spot
	//transformer TransformFunc

	// populated is true if the first batch of items inserted by Replace() has been populated
	// or Delete/Add/Update was called first.
	populated bool
	// initialPopulationCount is the number of items inserted by the first call of Replace()
	initialPopulationCount int

	// keyFunc is used to make the key used for queued item insertion and retrieval, and
	// should be deterministic.
	keyFunc KeyFunc

	// Indication the queue is closed.
	// Used to indicate a queue is closed so a control loop can exit when a queue is empty.
	// Currently, not used to gate any of CRUD operations.
	closed bool
}

var (
	_ = Queue(&RealFIFO{}) // RealFIFO is a Queue
)

// Close the queue.
func (f *RealFIFO) Close() {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.closed = true
	f.cond.Broadcast()
}

// KeyOf exposes f's keyFunc, but also detects the key of a Deltas object or
// DeletedFinalStateUnknown objects.
func (f *RealFIFO) KeyOf(obj interface{}) (string, error) {
	if d, ok := obj.(Deltas); ok {
		if len(d) == 0 {
			return "", KeyError{obj, ErrZeroLengthDeltasObject}
		}
		obj = d.Newest().Object
	}
	if d, ok := obj.(DeletedFinalStateUnknown); ok {
		return d.Key, nil
	}
	return f.keyFunc(obj)
}

// HasSynced returns true if an Add/Update/Delete/AddIfNotPresent are called first,
// or the first batch of items inserted by Replace() has been popped.
func (f *RealFIFO) HasSynced() bool {
	f.lock.Lock()
	defer f.lock.Unlock()
	return f.hasSynced_locked()
}

func (f *RealFIFO) hasSynced_locked() bool {
	return f.populated && f.initialPopulationCount == 0
}

// Add inserts an item, and puts it in the queue. The item is only enqueued
// if it doesn't already exist in the set.
func (f *RealFIFO) Add(obj interface{}) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	f.populated = true
	f.items = append(f.items, Delta{
		Type:   Added,
		Object: obj,
	})

	f.cond.Broadcast()
	return nil
}

// AddIfNotPresent inserts an item, and puts it in the queue. If the item is already
// present in the set, it is neither enqueued nor added to the set.
//
// This is useful in a single producer/consumer scenario so that the consumer can
// safely retry items without contending with the producer and potentially enqueueing
// stale items.
func (f *RealFIFO) AddIfNotPresent(obj interface{}) error {
	// this is not logically supported by this use-case because adding after the fact can make the
	// cache go back in time by adding stale data to the end of the items
	return nil
}

// Update is the same as Add in this implementation.
func (f *RealFIFO) Update(obj interface{}) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	f.populated = true
	f.items = append(f.items, Delta{
		Type:   Updated,
		Object: obj,
	})

	f.cond.Broadcast()
	return nil
}

// Delete removes an item. It doesn't add it to the queue, because
// this implementation assumes the consumer only cares about the objects,
// not the order in which they were created/added.
func (f *RealFIFO) Delete(obj interface{}) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	f.populated = true
	f.items = append(f.items, Delta{
		Type:   Deleted,
		Object: obj,
	})

	f.cond.Broadcast()
	return nil
}

// IsClosed checks if the queue is closed
func (f *RealFIFO) IsClosed() bool {
	f.lock.Lock()
	defer f.lock.Unlock()
	return f.closed
}

// Pop waits until an item is ready and processes it. If multiple items are
// ready, they are returned in the order in which they were added/updated.
// The item is removed from the queue (and the store) before it is processed,
// so if you don't successfully process it, it should be added back with
// AddIfNotPresent(). process function is called under lock, so it is safe
// update data structures in it that need to be in sync with the queue.
func (f *RealFIFO) Pop(process PopProcessFunc) (interface{}, error) {
	f.lock.Lock()
	defer f.lock.Unlock()
	for {
		for len(f.items) == 0 {
			// When the queue is empty, invocation of Pop() is blocked until new item is enqueued.
			// When Close() is called, the f.closed is set and the condition is broadcasted.
			// Which causes this loop to continue and return from the Pop().
			if f.closed {
				return nil, ErrFIFOClosed
			}

			f.cond.Wait()
		}
		isInInitialList := !f.hasSynced_locked()
		item := f.items[0]
		f.items = f.items[1:]
		if f.initialPopulationCount > 0 {
			f.initialPopulationCount--
		}
		err := process(Deltas{item}, isInInitialList)
		if e, ok := err.(ErrRequeue); ok {
			panic(fmt.Sprintf("figure out end up here in CI, %v", e))
		}
		return item, err
	}
}

// Replace will delete the contents of 'f', using instead the given map.
// 'f' takes ownership of the map, you should not reference the map again
// after calling this function. f's queue is reset, too; upon return, it
// will contain the items in the map, in no particular order.
func (f *RealFIFO) Replace(list []interface{}, resourceVersion string) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	if !f.populated {
		f.populated = true
		f.initialPopulationCount = len(list)
	}

	oldItems := f.items

	// Add all the new items
	f.items = make([]Delta, len(list))
	newKeys := sets.Set[string]{}
	for _, obj := range list {
		f.items = append(f.items, Delta{
			Type:   Replaced,
			Object: obj,
		})
		if key, err := f.keyFunc(obj); err == nil {
			newKeys.Insert(key)
		}
	}

	// Do deletion detection against objects in the queue
	oldKeys := sets.Set[string]{}
	for _, oldItem := range oldItems {
		oldKey, err := f.keyFunc(oldItem)
		if err != nil {
			continue
		}
		oldKeys.Insert(oldKey)

		if newKeys.Has(oldKey) {
			continue
		}

		// Delete items in the old list that are not in the new list.
		// This could happen if watch deletion event was missed while
		// disconnected from apiserver.
		deletedObj, exists, err := f.knownObjects.GetByKey(oldKey)
		if err != nil {
			deletedObj = nil
			//f.logger.Error(err, "Unexpected error during lookup, placing DeleteFinalStateUnknown marker without object", "key", k)
		} else if !exists {
			deletedObj = nil
			//f.logger.Info("Key does not exist in known objects store, placing DeleteFinalStateUnknown marker without object", "key", k)
		}
		f.items = append(f.items, Delta{
			Type: Deleted,
			Object: DeletedFinalStateUnknown{
				Key: oldKey,
				Obj: deletedObj,
			},
		})
	}

	// Detect deletions for objects not present in the queue, but present in KnownObjects
	knownKeys := f.knownObjects.ListKeys()
	for _, knownKey := range knownKeys {
		if newKeys.Has(knownKey) { // still present
			continue
		}
		if oldKeys.Has(knownKey) { // already added delete for these
			continue
		}

		deletedObj, exists, err := f.knownObjects.GetByKey(knownKey)
		if err != nil {
			deletedObj = nil
			//f.logger.Error(err, "Unexpected error during lookup, placing DeleteFinalStateUnknown marker without object", "key", k)
		} else if !exists {
			deletedObj = nil
			//f.logger.Info("Key does not exist in known objects store, placing DeleteFinalStateUnknown marker without object", "key", k)
		}
		f.items = append(f.items, Delta{
			Type: Deleted,
			Object: DeletedFinalStateUnknown{
				Key: knownKey,
				Obj: deletedObj,
			},
		})
	}

	if !f.populated {
		f.populated = true
		f.initialPopulationCount = len(f.items)
	}

	if len(f.items) > 0 {
		f.cond.Broadcast()
	}

	return nil
}

// Resync will ensure that every object in the Store has its key in the queue.
// This should be a no-op, because that property is maintained by all operations.
func (f *RealFIFO) Resync() error {
	// TODO this cannot logically be done by the FIFO, it can only be done by the indexer
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.knownObjects == nil {
		return nil
	}

	keysInQueue := sets.Set[string]{}
	for _, item := range f.items {
		key, err := f.keyFunc(item)
		if err != nil {
			continue
		}
		keysInQueue.Insert(key)
	}

	knownKeys := f.knownObjects.ListKeys()
	for _, knownKey := range knownKeys {
		// If we are doing Resync() and there is already an event queued for that object,
		// we ignore the Resync for it. This is to avoid the race, in which the resync
		// comes with the previous value of object (since queueing an event for the object
		// doesn't trigger changing the underlying store <knownObjects>.
		if keysInQueue.Has(knownKey) {
			continue
		}

		knownObj, exists, err := f.knownObjects.GetByKey(knownKey)
		if err != nil {
			//f.logger.Error(err, "Unexpected error during lookup, unable to queue object for sync", "key", key)
			continue
		} else if !exists {
			//f.logger.Info("Key does not exist in known objects store, unable to queue object for sync", "key", key)
			continue
		}

		f.items = append(f.items, Delta{
			Type:   Sync,
			Object: knownObj,
		})
	}

	if len(f.items) > 0 {
		f.cond.Broadcast()
	}
	return nil
}

// NewRealFIFO returns a Store which can be used to queue up items to
// process.
func NewRealFIFO(keyFunc KeyFunc, knownObjects KeyListerGetter) *RealFIFO {
	f := &RealFIFO{
		items:        make([]Delta, 10),
		keyFunc:      keyFunc,
		knownObjects: knownObjects,
	}
	f.cond.L = &f.lock
	return f
}
