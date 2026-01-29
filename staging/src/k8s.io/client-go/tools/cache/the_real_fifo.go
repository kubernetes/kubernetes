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
	// KnownObjects is required if AtomicEvents is false since it is used to
	// query the state of the internal store for Replace and Resync handling.
	KnownObjects KeyListerGetter

	// If set, will be called for objects before enqueueing them. Please
	// see the comment on TransformFunc for details.
	Transformer TransformFunc

	// AtomicEvents is used to specify whether the RealFIFO will emit events
	// atomically or not. If it is set, a single event will be emitted
	// atomically for Replace and Resync operations.
	// If AtomicEvents is true, KnownObjects must be nil.
	AtomicEvents bool

	// UnlockWhileProcessing is used to specify whether the RealFIFO can unlock
	// the lock while processing events. If it is set, the lock can be unlocked
	// while processing events to allow other goroutines to add items to the queue.
	// If UnlockWhileProcessing is true, AtomicEvents must be true as well.
	UnlockWhileProcessing bool

	// Identifier is used to identify this FIFO for metrics and logging purposes.
	// Optional. If zero value, metrics will not be published and trace logs will not
	// include Name or Resource fields.
	Identifier InformerNameAndResource

	// MetricsProvider is used to create metrics for the FIFO.
	MetricsProvider FIFOMetricsProvider
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
	// Replace(), and Resync().
	// It is nil if emitAtomicEvents is true.
	knownObjects KeyListerGetter

	// Indication the queue is closed.
	// Used to indicate a queue is closed so a control loop can exit when a queue is empty.
	// Currently, not used to gate any of CRUD operations.
	closed bool

	// Called with every object if non-nil.
	transformer TransformFunc

	// batchSize determines the maximum number of objects we can combine into a batch.
	batchSize int

	// emitAtomicEvents defines whether events like Replace and Resync should be emitted
	// atomically rather than as a series of events. This means that any call to the FIFO
	// will emit a single event.
	// If it is set:
	// * a single ReplacedAll event will be emitted instead of multiple Replace events
	// * a single SyncAll event will be emitted instead of multiple Sync events
	emitAtomicEvents bool

	// unlockWhileProcessing defines whether we can unlock while processing events.
	// This may only be set if emitAtomicEvents is true. If unlockWhileProcessing is true,
	// Pop and PopBatch must be called from a single threaded consumer.
	unlockWhileProcessing bool

	// identifier is used to identify this FIFO for metrics and logging purposes.
	identifier InformerNameAndResource

	// metrics holds all metrics for this FIFO.
	metrics *fifoMetrics
}

// ReplacedAllInfo is the object associated with a Delta of type=ReplacedAll
type ReplacedAllInfo struct {
	// ResourceVersion is the resource version passed to the Replace() call that created this Delta
	ResourceVersion string
	// Objects are the list of objects passed to the Replace() call that created this Delta,
	// with any configured transformation already applied.
	Objects []interface{}
}

// SyncAllInfo is the object associated with a Delta of type=SyncAll
// It is used to trigger a resync of the entire queue.
type SyncAllInfo struct{}

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
	f.metrics.numberOfQueuedItem.Set(float64(len(f.items)))

	return nil
}

// addReplaceToItemsLocked appends to the delta list.
func (f *RealFIFO) addReplaceToItemsLocked(objs []interface{}, resourceVersion string) error {
	// Replaced items must be transformed before being added to the queue. These objects must
	// all be objects that have not been transformed yet.
	if f.transformer != nil {
		transformedObjs := make([]interface{}, len(objs))
		for i, obj := range objs {
			transformedObj, err := f.transformer(obj)
			if err != nil {
				return err
			}
			transformedObjs[i] = transformedObj
		}
		objs = transformedObjs
	}

	info := ReplacedAllInfo{
		ResourceVersion: resourceVersion,
		Objects:         objs,
	}
	f.items = append(f.items, Delta{
		Type:   ReplacedAll,
		Object: info,
	})
	f.cond.Broadcast()
	f.metrics.numberOfQueuedItem.Set(float64(len(f.items)))

	return nil
}

func (f *RealFIFO) addResyncToItemsLocked() error {
	f.items = append(f.items, Delta{
		Type:   SyncAll,
		Object: SyncAllInfo{},
	})
	f.cond.Broadcast()
	f.metrics.numberOfQueuedItem.Set(float64(len(f.items)))

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
// ready, they are returned in the order in which they were added/updated. The
// item is removed from the queue (and the store) before it is processed. The
// process function is only guaranteed to be called under lock if
// UnlockWhileProcessing is false. If the process function is updating data
// structures that need to be in sync with the queue, ensure
// UnlockWhileProcessing is false. It is expected that the caller of Pop will be
// a single threaded consumer since otherwise it is possible for multiple
// PopProcessFuncs to be running simultaneously.
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
	// Decrement initialPopulationCount if needed.
	// This is done in a defer so we only do this *after* processing is complete,
	// so concurrent calls to hasSynced will not incorrectly return true while processing is still happening.
	defer func() {
		if f.initialPopulationCount > 0 {
			f.initialPopulationCount--
		}
	}()

	// Only log traces if the queue depth is greater than 10 and it takes more than
	// 100 milliseconds to process one item from the queue.
	// Queue depth never goes high because processing an item is locking the queue,
	// and new items can't be added until processing finish.
	// https://github.com/kubernetes/kubernetes/issues/103789
	if len(f.items) > 10 {
		id, _ := f.keyOf(item)
		fields := []utiltrace.Field{
			{Key: "ID", Value: id},
			{Key: "Depth", Value: len(f.items)},
			{Key: "Reason", Value: "slow event handlers blocking the queue"},
		}
		if name := f.identifier.Name(); len(name) > 0 {
			fields = append(fields, utiltrace.Field{Key: "Name", Value: name})
		}
		if gvr := f.identifier.GroupVersionResource(); !gvr.Empty() {
			fields = append(fields, utiltrace.Field{Key: "Resource", Value: gvr})
		}
		trace := utiltrace.New("RealFIFO Pop Process", fields...)
		defer trace.LogIfLong(100 * time.Millisecond)
	}
	f.metrics.numberOfQueuedItem.Set(float64(len(f.items)))

	// Process the item, this may unlock the lock, and allow other goroutines to add items to the queue.
	err := f.whileProcessing_locked(func() error {
		// we wrap in Deltas here to be compatible with preview Pop functions and those interpreting the return value.
		return process(Deltas{item}, isInInitialList)
	})
	return Deltas{item}, err
}

// whileProcessing_locked calls the `process` function.
// The lock must be held before calling `whileProcessing_locked`, and is held when `whileProcessing_locked` returns.
// whileProcessing_locked releases the lock during the call to `process` if f.unlockWhileProcessing is true and the f.items queue is not too long.
func (f *RealFIFO) whileProcessing_locked(process func() error) error {
	// Unlock before calling `process` so new items can be enqueued during processing.
	// Only do this if the queue contains less than 2 full batches of items,
	// to prevent the queue from growing unboundedly.
	if f.unlockWhileProcessing && len(f.items) < f.batchSize*2 {
		f.lock.Unlock()
		defer f.lock.Lock()
	}
	return process()
}

// batchable stores the delta types that can be batched
var batchable = map[DeltaType]bool{
	Sync:     true,
	Replaced: true,
	Added:    true,
	Updated:  true,
	Deleted:  true,
}

// PopBatch pops as many items as possible to be processed as a batch using processBatch,
// or pop a single item using processSingle if multiple items cannot be batched.
//
// The processBatch and processSingle functions are only guaranteed to be called
// under lock if UnlockWhileProcessing is false. If the process functions are
// updating data structures that need to be in sync with the queue, ensure
// UnlockWhileProcessing is false. It is expected that the caller of PopBatch
// will be a single threaded consumer, since otherwise it is possible for
// multiple ProcessBatchFunc or PopProcessFunc's to be running simultaneously.
func (f *RealFIFO) PopBatch(processBatch ProcessBatchFunc, processSingle PopProcessFunc) error {
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
	moveDeltaToProcessList := func(i int) {
		deltas = append(deltas, f.items[i])
		// The underlying array still exists and references this object, so the object will not be garbage collected unless we zero the reference.
		f.items[i] = Delta{}
	}
	// only bundle unique items into a batch
	for i := 0; i < f.batchSize && i < len(f.items); i++ {
		if f.initialPopulationCount > 0 && i >= f.initialPopulationCount {
			break
		}
		item := f.items[i]
		if !batchable[item.Type] {
			if len(deltas) == 0 {
				// if an unbatchable delta is first in the list, process just that one by itself
				moveDeltaToProcessList(i)
			}
			// close the batch when an unbatchable delta is encountered
			break
		}
		id, err := f.keyOf(item)
		if err != nil {
			// close the batch here if error happens
			// TODO: log the error when RealFIFOOptions supports passing klog instance like deprecated DeltaFIFO
			// still pop the broken item out of queue to be compatible with the non-batch behavior it should be safe
			// when 1st element is broken, however for Nth broken element, there's possible risk that broken item
			// still can be processed and broke the uniqueness of the batch unexpectedly.
			moveDeltaToProcessList(i)
			break
		}
		if unique.Has(id) {
			// close the batch if a duplicate item is encountered
			break
		}
		unique.Insert(id)
		moveDeltaToProcessList(i)
	}

	f.items = f.items[len(deltas):]
	// Decrement initialPopulationCount if needed.
	// This is done in a defer so we only do this *after* processing is complete,
	// so concurrent calls to hasSynced will not incorrectly return true while processing is still happening.
	defer func() {
		if f.initialPopulationCount > 0 {
			f.initialPopulationCount -= len(deltas)
		}
	}()

	// Only log traces if the queue depth is greater than 10 and it takes more than
	// 100 milliseconds to process one item from the queue (with a max of 1 second for the whole batch)
	// Queue depth never goes high because processing an item is locking the queue,
	// and new items can't be added until processing finish.
	// https://github.com/kubernetes/kubernetes/issues/103789
	if len(f.items) > 10 {
		id, _ := f.keyOf(deltas[0])
		fields := []utiltrace.Field{
			{Key: "ID", Value: id},
			{Key: "Depth", Value: len(f.items)},
			{Key: "Reason", Value: "slow event handlers blocking the queue"},
			{Key: "BatchSize", Value: len(deltas)},
		}
		if name := f.identifier.Name(); len(name) > 0 {
			fields = append(fields, utiltrace.Field{Key: "Name", Value: name})
		}
		if gvr := f.identifier.GroupVersionResource(); !gvr.Empty() {
			fields = append(fields, utiltrace.Field{Key: "Resource", Value: gvr})
		}
		trace := utiltrace.New("RealFIFO PopBatch Process", fields...)
		defer trace.LogIfLong(min(100*time.Millisecond*time.Duration(len(deltas)), time.Second))
	}
	f.metrics.numberOfQueuedItem.Set(float64(len(f.items)))

	if len(deltas) == 1 {
		return f.whileProcessing_locked(func() error {
			return processSingle(Deltas{deltas[0]}, isInInitialList)
		})
	}
	return f.whileProcessing_locked(func() error {
		return processBatch(deltas, isInInitialList)
	})
}

// Replace
// 1. finds those items in f.items that are not in newItems and creates synthetic deletes for them
// 2. finds items in knownObjects that are not in newItems and creates synthetic deletes for them
// 3. adds the newItems to the queue
func (f *RealFIFO) Replace(newItems []interface{}, resourceVersion string) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	var err error
	if f.emitAtomicEvents {
		err = f.addReplaceToItemsLocked(newItems, resourceVersion)
	} else {
		err = reconcileReplacement(f.items, f.knownObjects, newItems, f.keyOf,
			func(obj DeletedFinalStateUnknown) error {
				return f.addToItems_locked(Deleted, true, obj)
			},
			func(obj interface{}) error {
				return f.addToItems_locked(Replaced, false, obj)
			})
	}
	if err != nil {
		return err
	}

	if !f.populated {
		f.populated = true
		f.initialPopulationCount = len(f.items)
	}

	return nil
}

// reconcileReplacement takes the items that are already in the queue and the set of new items
// and based upon the state of the items in the queue and known objects will call onDelete and onReplace
// depending upon whether the item is being deleted or replaced/added.
func reconcileReplacement(
	queuedItems []Delta,
	knownObjects KeyListerGetter,
	newItems []interface{},
	keyOf func(obj interface{}) (string, error),
	onDelete func(obj DeletedFinalStateUnknown) error,
	onReplace func(obj interface{}) error,
) error {
	// determine the keys of everything we're adding.  We cannot add the items until after the synthetic deletes have been
	// created for items that don't existing in newItems
	newKeys := sets.Set[string]{}
	for _, obj := range newItems {
		key, err := keyOf(obj)
		if err != nil {
			return KeyError{obj, err}
		}
		newKeys.Insert(key)
	}

	queuedKeys := []string{}
	lastQueuedItemForKey := map[string]Delta{}
	for _, queuedItem := range queuedItems {
		queuedKey, err := keyOf(queuedItem.Object)
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

		retErr := onDelete(DeletedFinalStateUnknown{
			Key: queuedKey,
			Obj: deletedObj,
		})
		if retErr != nil {
			return fmt.Errorf("couldn't enqueue object: %w", retErr)
		}
	}

	// Detect deletions for objects not present in the queue, but present in KnownObjects
	knownKeys := knownObjects.ListKeys()
	for _, knownKey := range knownKeys {
		if newKeys.Has(knownKey) { // still present
			continue
		}
		if _, inQueuedItems := lastQueuedItemForKey[knownKey]; inQueuedItems { // already added delete for these
			continue
		}

		deletedObj, exists, err := knownObjects.GetByKey(knownKey)
		if err != nil {
			deletedObj = nil
			utilruntime.HandleError(fmt.Errorf("error during lookup, placing DeleteFinalStateUnknown marker without object: key=%q, err=%w", knownKey, err))
		} else if !exists {
			deletedObj = nil
			utilruntime.HandleError(fmt.Errorf("key does not exist in known objects store, placing DeleteFinalStateUnknown marker without object: key=%q", knownKey))
		}
		retErr := onDelete(DeletedFinalStateUnknown{
			Key: knownKey,
			Obj: deletedObj,
		})
		if retErr != nil {
			return fmt.Errorf("couldn't enqueue object: %w", retErr)
		}
	}

	// now that we have the deletes we need for items, we can add the newItems to the items queue
	for _, obj := range newItems {
		if err := onReplace(obj); err != nil {
			return fmt.Errorf("couldn't enqueue object: %w", err)
		}
	}

	return nil
}

// Resync will ensure that every object in the Store has its key in the queue.
// This should be a no-op, because that property is maintained by all operations.
func (f *RealFIFO) Resync() error {
	// TODO this cannot logically be done by the FIFO, it can only be done by the indexer
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.emitAtomicEvents {
		return f.addResyncToItemsLocked()
	}

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
//
// Deprecated: Use NewRealFIFOWithOptions instead.
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

	if opts.AtomicEvents {
		// If we are emitting atomic events, we must not rely on the known objects store
		// as it is a requirement to be able to release the lock while processing events.
		if opts.KnownObjects != nil {
			panic("coding error: knownObjects must not be provided when AtomicEvents is true")
		}
	} else {
		if opts.UnlockWhileProcessing {
			panic("coding error: UnlockWhileProcessing must be false when AtomicEvents is false")
		}
		if opts.KnownObjects == nil {
			panic("coding error: knownObjects must be provided when AtomicEvents is false")
		}
	}

	f := &RealFIFO{
		items:                 make([]Delta, 0, 10),
		keyFunc:               opts.KeyFunction,
		knownObjects:          opts.KnownObjects,
		transformer:           opts.Transformer,
		batchSize:             defaultBatchSize,
		emitAtomicEvents:      opts.AtomicEvents,
		unlockWhileProcessing: opts.UnlockWhileProcessing,
		identifier:            opts.Identifier,
		metrics:               newFIFOMetrics(opts.Identifier, opts.MetricsProvider),
	}

	f.cond.L = &f.lock
	return f
}
