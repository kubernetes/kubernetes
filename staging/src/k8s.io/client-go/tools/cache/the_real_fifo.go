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
	"sync"
	"time"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utiltrace "k8s.io/utils/trace"
)

// RealFIFOOptions is the configuration parameters for RealFIFO.
type RealFIFOOptions struct {
	// KeyFunction is used to figure out what key an object should have. (It's
	// exposed in the returned RealFIFO's keyOf() method, with additional
	// handling around deleted objects and queue state).
	// Optional, the default is MetaNamespaceKeyFunc.
	KeyFunction KeyFunc

	// KnownObjects is expected to return a list of keys that the consumer of
	// this queue "knows about". It is used to decide which items are missing
	// when Replace() is called; 'Deleted' deltas are produced for the missing items.
	// KnownObjects is required.
	KnownObjects KeyListerGetter

	// If set, will be called for objects before enqueueing them. Please
	// see the comment on TransformFunc for details.
	Transformer TransformFunc
}

const (
	defaultBatchSize = 1000
)

var _ QueueWithBatch = &RealFIFO{}

// RealFIFO is a Queue in which every notification from the Reflector is passed
// in order to the Queue via Pop.
// This means that it
// 1. delivers notifications for items that have been deleted
// 2. delivers multiple notifications per item instead of simply the most recent value
type RealFIFO struct {
	lock sync.RWMutex
	cond sync.Cond

	items []Delta

	// populated is true if the first batch of items inserted by Replace() has been populated
	// or Delete/Add/Update was called first.
	populated bool
	// initialPopulationCount is the number of items inserted by the first call of Replace()
	initialPopulationCount int

	// keyFunc is used to make the key used for queued item insertion and retrieval, and
	// should be deterministic.
	keyFunc KeyFunc

	// knownObjects list keys that are "known" --- affecting Delete(),
	// Replace(), and Resync()
	knownObjects KeyListerGetter

	// Indication the queue is closed.
	// Used to indicate a queue is closed so a control loop can exit when a queue is empty.
	// Currently, not used to gate any of CRUD operations.
	closed bool

	// Called with every object if non-nil.
	transformer TransformFunc

	// batchSize determines the maximum number of objects we can combine into a batch.
	batchSize int
}

var (
	_ = Queue(&RealFIFO{})             // RealFIFO is a Queue
	_ = TransformingStore(&RealFIFO{}) // RealFIFO implements TransformingStore to allow memory optimizations
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
func (f *RealFIFO) keyOf(obj interface{}) (string, error) {
	if d, ok := obj.(Deltas); ok {
		if len(d) == 0 {
			return "", KeyError{obj, ErrZeroLengthDeltasObject}
		}
		obj = d.Newest().Object
	}
	if d, ok := obj.(Delta); ok {
		obj = d.Object
	}
	if d, ok := obj.(DeletedFinalStateUnknown); ok {
		return d.Key, nil
	}
	return f.keyFunc(obj)
}

// HasSynced returns true if an Add/Update/Delete are called first,
// or the first batch of items inserted by Replace() has been popped.
func (f *RealFIFO) HasSynced() bool {
	f.lock.Lock()
	defer f.lock.Unlock()
	return f.hasSynced_locked()
}

// ignoring lint to reduce delta to the original for review.  It's ok adjust later.
//
//lint:file-ignore ST1003: should not use underscores in Go names
func (f *RealFIFO) hasSynced_locked() bool {
	return f.populated && f.initialPopulationCount == 0
}

// addToItems_locked appends to the delta list.
func (f *RealFIFO) addToItems_locked(deltaActionType DeltaType, skipTransform bool, obj interface{}) error {
	// we must be able to read the keys in order to determine whether the knownObjcts and the items
	// in this FIFO overlap
	_, err := f.keyOf(obj)
	if err != nil {
		return KeyError{obj, err}
	}

	// Every object comes through this code path once, so this is a good
	// place to call the transform func.
	//
	// If obj is a DeletedFinalStateUnknown tombstone or the action is a Sync,
	// then the object have already gone through the transformer.
	//
	// If the objects already present in the cache are passed to Replace(),
	// the transformer must be idempotent to avoid re-mutating them,
	// or coordinate with all readers from the cache to avoid data races.
	// Default informers do not pass existing objects to Replace.
	if f.transformer != nil {
		_, isTombstone := obj.(DeletedFinalStateUnknown)
		if !isTombstone && !skipTransform {
			var err error
			obj, err = f.transformer(obj)
			if err != nil {
				return err
			}
		}
	}

	f.items = append(f.items, Delta{
		Type:   deltaActionType,
		Object: obj,
	})
	f.cond.Broadcast()

	return nil
}

// Add inserts an item, and puts it in the queue. The item is only enqueued
// if it doesn't already exist in the set.
func (f *RealFIFO) Add(obj interface{}) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	f.populated = true
	retErr := f.addToItems_locked(Added, false, obj)

	return retErr
}

// Update is the same as Add in this implementation.
func (f *RealFIFO) Update(obj interface{}) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	f.populated = true
	retErr := f.addToItems_locked(Updated, false, obj)

	return retErr
}

// Delete removes an item. It doesn't add it to the queue, because
// this implementation assumes the consumer only cares about the objects,
// not the order in which they were created/added.
func (f *RealFIFO) Delete(obj interface{}) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	f.populated = true
	retErr := f.addToItems_locked(Deleted, false, obj)

	return retErr
}

// IsClosed checks if the queue is closed
func (f *RealFIFO) IsClosed() bool {
	f.lock.Lock()
	defer f.lock.Unlock()
	return f.closed
}

// Pop waits until an item is ready and processes it. If multiple items are
// ready, they are returned in the order in which they were added/updated.
// The item is removed from the queue (and the store) before it is processed.
// process function is called under lock, so it is safe
// update data structures in it that need to be in sync with the queue.
func (f *RealFIFO) Pop(process PopProcessFunc) (interface{}, error) {
	f.lock.Lock()
	defer f.lock.Unlock()

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
	// The underlying array still exists and references this object, so the object will not be garbage collected unless we zero the reference.
	f.items[0] = Delta{}
	f.items = f.items[1:]
	if f.initialPopulationCount > 0 {
		f.initialPopulationCount--
	}

	// Only log traces if the queue depth is greater than 10 and it takes more than
	// 100 milliseconds to process one item from the queue.
	// Queue depth never goes high because processing an item is locking the queue,
	// and new items can't be added until processing finish.
	// https://github.com/kubernetes/kubernetes/issues/103789
	if len(f.items) > 10 {
		id, _ := f.keyOf(item)
		trace := utiltrace.New("RealFIFO Pop Process",
			utiltrace.Field{Key: "ID", Value: id},
			utiltrace.Field{Key: "Depth", Value: len(f.items)},
			utiltrace.Field{Key: "Reason", Value: "slow event handlers blocking the queue"})
		defer trace.LogIfLong(100 * time.Millisecond)
	}

	// we wrap in Deltas here to be compatible with preview Pop functions and those interpreting the return value.
	err := process(Deltas{item}, isInInitialList)
	return Deltas{item}, err
}

func (f *RealFIFO) PopBatch(process ProcessBatchFunc) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	for len(f.items) == 0 {
		// When the queue is empty, invocation of Pop() is blocked until new item is enqueued.
		// When Close() is called, the f.closed is set and the condition is broadcasted.
		// Which causes this loop to continue and return from the Pop().
		if f.closed {
			return ErrFIFOClosed
		}

		f.cond.Wait()
	}

	isInInitialList := !f.hasSynced_locked()
	unique := sets.NewString()
	deltas := make([]Delta, 0, min(len(f.items), f.batchSize))
	// only bundle unique items into a batch
	for i := 0; i < f.batchSize && i < len(f.items); i++ {
		if f.initialPopulationCount > 0 && i >= f.initialPopulationCount {
			break
		}
		item := f.items[i]
		id, err := f.keyOf(item)
		if err != nil {
			// close the batch here if error happens
			// TODO: log the error when RealFIFOOptions supports passing klog instance like deprecated DeltaFIFO
			// still pop the broken item out of queue to be compatible with the non-batch behavior it should be safe
			// when 1st element is broken, however for Nth broken element, there's possible risk that broken item
			// still can be processed and broke the uniqueness of the batch unexpectedly.
			deltas = append(deltas, item)
			// The underlying array still exists and references this object, so the object will not be garbage collected unless we zero the reference.
			f.items[i] = Delta{}
			break
		}
		if unique.Has(id) {
			break
		}
		unique.Insert(id)
		deltas = append(deltas, item)
		// The underlying array still exists and references this object, so the object will not be garbage collected unless we zero the reference.
		f.items[i] = Delta{}
	}
	if f.initialPopulationCount > 0 {
		f.initialPopulationCount -= len(deltas)
	}
	f.items = f.items[len(deltas):]

	// Only log traces if the queue depth is greater than 10 and it takes more than
	// 100 milliseconds to process one item from the queue (with a max of 1 second for the whole batch)
	// Queue depth never goes high because processing an item is locking the queue,
	// and new items can't be added until processing finish.
	// https://github.com/kubernetes/kubernetes/issues/103789
	if len(f.items) > 10 {
		id, _ := f.keyOf(deltas[0])
		trace := utiltrace.New("RealFIFO PopBatch Process",
			utiltrace.Field{Key: "ID", Value: id},
			utiltrace.Field{Key: "Depth", Value: len(f.items)},
			utiltrace.Field{Key: "Reason", Value: "slow event handlers blocking the queue"},
			utiltrace.Field{Key: "BatchSize", Value: len(deltas)})
		defer trace.LogIfLong(min(100*time.Millisecond*time.Duration(len(deltas)), time.Second))
	}

	err := process(deltas, isInInitialList)
	return err
}

// Replace
// 1. finds those items in f.items that are not in newItems and creates synthetic deletes for them
// 2. finds items in knownObjects that are not in newItems and creates synthetic deletes for them
// 3. adds the newItems to the queue
func (f *RealFIFO) Replace(newItems []interface{}, resourceVersion string) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	// determine the keys of everything we're adding.  We cannot add the items until after the synthetic deletes have been
	// created for items that don't existing in newItems
	newKeys := sets.Set[string]{}
	for _, obj := range newItems {
		key, err := f.keyOf(obj)
		if err != nil {
			return KeyError{obj, err}
		}
		newKeys.Insert(key)
	}

	queuedItems := f.items
	queuedKeys := []string{}
	lastQueuedItemForKey := map[string]Delta{}
	for _, queuedItem := range queuedItems {
		queuedKey, err := f.keyOf(queuedItem.Object)
		if err != nil {
			return KeyError{queuedItem.Object, err}
		}

		if _, seen := lastQueuedItemForKey[queuedKey]; !seen {
			queuedKeys = append(queuedKeys, queuedKey)
		}
		lastQueuedItemForKey[queuedKey] = queuedItem
	}

	// all the deletes already in the queue are important. There are two cases
	// 1. queuedItems has delete for key/X and newItems has replace for key/X.  This means the queued UID was deleted and a new one was created.
	// 2. queuedItems has a delete for key/X and newItems does NOT have key/X.  This means the queued item was deleted.
	// Do deletion detection against objects in the queue.
	for _, queuedKey := range queuedKeys {
		if newKeys.Has(queuedKey) {
			continue
		}

		// Delete pre-existing items not in the new list.
		// This could happen if watch deletion event was missed while
		// disconnected from apiserver.
		lastQueuedItem := lastQueuedItemForKey[queuedKey]
		// if we've already got the item marked as deleted, no need to add another delete
		if lastQueuedItem.Type == Deleted {
			continue
		}

		// if we got here, then the last entry we have for the queued item is *not* a deletion and we need to add a delete
		deletedObj := lastQueuedItem.Object

		retErr := f.addToItems_locked(Deleted, true, DeletedFinalStateUnknown{
			Key: queuedKey,
			Obj: deletedObj,
		})
		if retErr != nil {
			return fmt.Errorf("couldn't enqueue object: %w", retErr)
		}
	}

	// Detect deletions for objects not present in the queue, but present in KnownObjects
	knownKeys := f.knownObjects.ListKeys()
	for _, knownKey := range knownKeys {
		if newKeys.Has(knownKey) { // still present
			continue
		}
		if _, inQueuedItems := lastQueuedItemForKey[knownKey]; inQueuedItems { // already added delete for these
			continue
		}

		deletedObj, exists, err := f.knownObjects.GetByKey(knownKey)
		if err != nil {
			deletedObj = nil
			utilruntime.HandleError(fmt.Errorf("error during lookup, placing DeleteFinalStateUnknown marker without object: key=%q, err=%w", knownKey, err))
		} else if !exists {
			deletedObj = nil
			utilruntime.HandleError(fmt.Errorf("key does not exist in known objects store, placing DeleteFinalStateUnknown marker without object: key=%q", knownKey))
		}
		retErr := f.addToItems_locked(Deleted, false, DeletedFinalStateUnknown{
			Key: knownKey,
			Obj: deletedObj,
		})
		if retErr != nil {
			return fmt.Errorf("couldn't enqueue object: %w", retErr)
		}
	}

	// now that we have the deletes we need for items, we can add the newItems to the items queue
	for _, obj := range newItems {
		retErr := f.addToItems_locked(Replaced, false, obj)
		if retErr != nil {
			return fmt.Errorf("couldn't enqueue object: %w", retErr)
		}
	}

	if !f.populated {
		f.populated = true
		f.initialPopulationCount = len(f.items)
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
		key, err := f.keyOf(item.Object)
		if err != nil {
			return KeyError{item, err}
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
			utilruntime.HandleError(fmt.Errorf("unable to queue object for sync: key=%q, err=%w", knownKey, err))
			continue
		} else if !exists {
			utilruntime.HandleError(fmt.Errorf("key does not exist in known objects store, unable to queue object for sync: key=%q", knownKey))
			continue
		}

		retErr := f.addToItems_locked(Sync, true, knownObj)
		if retErr != nil {
			return fmt.Errorf("couldn't queue object: %w", err)
		}
	}

	return nil
}

// Transformer implements the TransformingStore interface.
func (f *RealFIFO) Transformer() TransformFunc {
	return f.transformer
}

// NewRealFIFO returns a Store which can be used to queue up items to
// process.
func NewRealFIFO(keyFunc KeyFunc, knownObjects KeyListerGetter, transformer TransformFunc) *RealFIFO {
	return NewRealFIFOWithOptions(RealFIFOOptions{
		KeyFunction:  keyFunc,
		KnownObjects: knownObjects,
		Transformer:  transformer,
	})
}

// NewRealFIFOWithOptions returns a Queue which can be used to process changes to
// items. See also the comment on RealFIFO.
func NewRealFIFOWithOptions(opts RealFIFOOptions) *RealFIFO {
	if opts.KeyFunction == nil {
		opts.KeyFunction = MetaNamespaceKeyFunc
	}

	if opts.KnownObjects == nil {
		panic("coding error: knownObjects must be provided")
	}

	f := &RealFIFO{
		items:        make([]Delta, 0, 10),
		keyFunc:      opts.KeyFunction,
		knownObjects: opts.KnownObjects,
		transformer:  opts.Transformer,
		batchSize:    defaultBatchSize,
	}

	f.cond.L = &f.lock
	return f
}
