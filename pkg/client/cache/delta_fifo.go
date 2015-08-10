/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"errors"
	"fmt"
	"sync"

	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
)

// NewDeltaFIFO returns a Store which can be used process changes to items.
//
// keyFunc is used to figure out what key an object should have. (It's
// exposed in the returned DeltaFIFO's KeyOf() method, with bonus features.)
//
// 'compressor' may compress as many or as few items as it wants
// (including returning an empty slice), but it should do what it
// does quickly since it is called while the queue is locked.
// 'compressor' may be nil if you don't want any delta compression.
//
// 'keyLister' is expected to return a list of keys that the consumer of
// this queue "knows about". It is used to decide which items are missing
// when Replace() is called; 'Deleted' deltas are produced for these items.
// It may be nil if you don't need to detect all deletions.
// TODO: consider merging keyLister with this object, tracking a list of
//       "known" keys when Pop() is called. Have to think about how that
//       affects error retrying.
//
// Also see the comment on DeltaFIFO.
func NewDeltaFIFO(keyFunc KeyFunc, compressor DeltaCompressor, knownObjectKeys KeyLister) *DeltaFIFO {
	f := &DeltaFIFO{
		items:           map[string]Deltas{},
		queue:           []string{},
		keyFunc:         keyFunc,
		deltaCompressor: compressor,
		knownObjectKeys: knownObjectKeys,
	}
	f.cond.L = &f.lock
	return f
}

// DeltaFIFO is like FIFO, but allows you to process deletes.
//
// DeltaFIFO is a producer-consumer queue, where a Reflector is
// intended to be the producer, and the consumer is whatever calls
// the Pop() method.
//
// DeltaFIFO solves this use case:
//  * You want to process every object change (delta) at most once.
//  * When you process an object, you want to see everything
//    that's happened to it since you last processed it.
//  * You want to process the deletion of objects.
//  * You might want to periodically reprocess objects.
//
// DeltaFIFO's Pop(), Get(), and GetByKey() methods return
// interface{} to satisfy the Store/Queue interfaces, but it
// will always return an object of type Deltas.
//
// A note on threading: If you call Pop() in parallel from multiple
// threads, you could end up with multiple threads processing slightly
// different versions of the same object.
//
// A note on the KeyLister used by the DeltaFIFO: It's main purpose is
// to list keys that are "known", for the puspose of figuring out which
// items have been deleted when Replace() is called. If the given KeyLister
// also satisfies the KeyGetter interface, the deleted objet will be
// included in the DeleteFinalStateUnknown markers. These objects
// could be stale.
//
// You may provide a function to compress deltas (e.g., represent a
// series of Updates as a single Update).
type DeltaFIFO struct {
	// lock/cond protects access to 'items' and 'queue'.
	lock sync.RWMutex
	cond sync.Cond

	// We depend on the property that items in the set are in
	// the queue and vice versa, and that all Deltas in this
	// map have at least one Delta.
	items map[string]Deltas
	queue []string

	// keyFunc is used to make the key used for queued item
	// insertion and retrieval, and should be deterministic.
	keyFunc KeyFunc

	// deltaCompressor tells us how to combine two or more
	// deltas. It may be nil.
	deltaCompressor DeltaCompressor

	// knownObjectKeys list keys that are "known", for the
	// purpose of figuring out which items have been deleted
	// when Replace() is called.
	knownObjectKeys KeyLister
}

var (
	_ = Queue(&DeltaFIFO{}) // DeltaFIFO is a Queue
)

var (
	// ErrZeroLengthDeltasObject is returned in a KeyError if a Deltas
	// object with zero length is encountered (should be impossible,
	// even if such an object is accidentally produced by a DeltaCompressor--
	// but included for completeness).
	ErrZeroLengthDeltasObject = errors.New("0 length Deltas object; can't get key")
)

// KeyOf exposes f's keyFunc, but also detects the key of a Deltas object or
// DeletedFinalStateUnknown objects.
func (f *DeltaFIFO) KeyOf(obj interface{}) (string, error) {
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

// Add inserts an item, and puts it in the queue. The item is only enqueued
// if it doesn't already exist in the set.
func (f *DeltaFIFO) Add(obj interface{}) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	return f.queueActionLocked(Added, obj)
}

// Update is just like Add, but makes an Updated Delta.
func (f *DeltaFIFO) Update(obj interface{}) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	return f.queueActionLocked(Updated, obj)
}

// Delete is just like Add, but makes an Deleted Delta.
func (f *DeltaFIFO) Delete(obj interface{}) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	return f.queueActionLocked(Deleted, obj)
}

// AddIfNotPresent inserts an item, and puts it in the queue. If the item is already
// present in the set, it is neither enqueued nor added to the set.
//
// This is useful in a single producer/consumer scenario so that the consumer can
// safely retry items without contending with the producer and potentially enqueueing
// stale items.
//
// Important: obj must be a Deltas (the output of the Pop() function). Yes, this is
// different from the Add/Update/Delete functions.
func (f *DeltaFIFO) AddIfNotPresent(obj interface{}) error {
	deltas, ok := obj.(Deltas)
	if !ok {
		return fmt.Errorf("object must be of type deltas, but got: %#v", obj)
	}
	id, err := f.KeyOf(deltas.Newest().Object)
	if err != nil {
		return KeyError{obj, err}
	}
	f.lock.Lock()
	defer f.lock.Unlock()
	if _, exists := f.items[id]; exists {
		return nil
	}

	f.queue = append(f.queue, id)
	f.items[id] = deltas
	f.cond.Broadcast()
	return nil
}

// queueActionLocked appends to the delta list for the object, calling
// f.deltaCompressor if needed
func (f *DeltaFIFO) queueActionLocked(actionType DeltaType, obj interface{}) error {
	id, err := f.KeyOf(obj)
	if err != nil {
		return KeyError{obj, err}
	}
	newDeltas := append(f.items[id], Delta{actionType, obj})
	if f.deltaCompressor != nil {
		newDeltas = f.deltaCompressor.Compress(newDeltas)
	}

	_, exists := f.items[id]
	if len(newDeltas) > 0 {
		if !exists {
			f.queue = append(f.queue, id)
		}
		f.items[id] = newDeltas
		f.cond.Broadcast()
	} else if exists {
		// The compression step removed all deltas, so
		// we need to remove this from our map (extra items
		// in the queue are ignored if they are not in the
		// map).
		delete(f.items, id)
	}
	return nil
}

// List returns a list of all the items; it returns the object
// from the most recent Delta.
// You should treat the items returned inside the deltas as immutable.
func (f *DeltaFIFO) List() []interface{} {
	f.lock.RLock()
	defer f.lock.RUnlock()
	list := make([]interface{}, 0, len(f.items))
	for _, item := range f.items {
		// Copy item's slice so operations on this slice (delta
		// compression) won't interfere with the object we return.
		item = copyDeltas(item)
		list = append(list, item.Newest().Object)
	}
	return list
}

// ListKeys returns a list of all the keys of the objects currently
// in the FIFO.
func (f *DeltaFIFO) ListKeys() []string {
	f.lock.RLock()
	defer f.lock.RUnlock()
	list := make([]string, 0, len(f.items))
	for key := range f.items {
		list = append(list, key)
	}
	return list
}

// Get returns the complete list of deltas for the requested item,
// or sets exists=false.
// You should treat the items returned inside the deltas as immutable.
func (f *DeltaFIFO) Get(obj interface{}) (item interface{}, exists bool, err error) {
	key, err := f.KeyOf(obj)
	if err != nil {
		return nil, false, KeyError{obj, err}
	}
	return f.GetByKey(key)
}

// GetByKey returns the complete list of deltas for the requested item,
// setting exists=false if that list is empty.
// You should treat the items returned inside the deltas as immutable.
func (f *DeltaFIFO) GetByKey(key string) (item interface{}, exists bool, err error) {
	f.lock.RLock()
	defer f.lock.RUnlock()
	d, exists := f.items[key]
	if exists {
		// Copy item's slice so operations on this slice (delta
		// compression) won't interfere with the object we return.
		d = copyDeltas(d)
	}
	return d, exists, nil
}

// Pop blocks until an item is added to the queue, and then returns it.  If
// multiple items are ready, they are returned in the order in which they were
// added/updated. The item is removed from the queue (and the store) before it
// is returned, so if you don't successfully process it, you need to add it back
// with AddIfNotPresent().
//
// Pop returns a 'Deltas', which has a complete list of all the things
// that happened to the object (deltas) while it was sitting in the queue.
func (f *DeltaFIFO) Pop() interface{} {
	f.lock.Lock()
	defer f.lock.Unlock()
	for {
		for len(f.queue) == 0 {
			f.cond.Wait()
		}
		id := f.queue[0]
		f.queue = f.queue[1:]
		item, ok := f.items[id]
		if !ok {
			// Item may have been deleted subsequently.
			continue
		}
		delete(f.items, id)
		// Don't need to copyDeltas here, because we're transferring
		// ownership to the caller.
		return item
	}
}

// Replace will delete the contents of 'f', using instead the given map.
// 'f' takes ownership of the map, you should not reference the map again
// after calling this function. f's queue is reset, too; upon return, it
// will contain the items in the map, in no particular order.
func (f *DeltaFIFO) Replace(list []interface{}) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	for _, item := range list {
		if err := f.queueActionLocked(Sync, item); err != nil {
			return fmt.Errorf("couldn't enqueue object: %v", err)
		}
	}
	if f.knownObjectKeys == nil {
		return nil
	}

	keySet := make(util.StringSet, len(list))
	for _, item := range list {
		key, err := f.KeyOf(item)
		if err != nil {
			return KeyError{item, err}
		}
		keySet.Insert(key)
	}

	// Detect deletions not already in the queue.
	knownKeys := f.knownObjectKeys.ListKeys()
	for _, k := range knownKeys {
		if _, exists := keySet[k]; exists {
			continue
		}

		// This key isn't in the complete set we got, so it must have been deleted.
		if d, exists := f.items[k]; exists {
			// Don't issue a delete delta if we have one enqueued as the most
			// recent delta.
			if d.Newest().Type == Deleted {
				continue
			}
		}
		var deletedObj interface{}
		if keyGetter, ok := f.knownObjectKeys.(KeyGetter); ok {
			var exists bool
			var err error
			deletedObj, exists, err = keyGetter.GetByKey(k)
			if err != nil || !exists {
				deletedObj = nil
				if err != nil {
					glog.Errorf("Unexpected error %v during lookup of key %v, placing DeleteFinalStateUnknown marker without object", err, k)
				} else {
					glog.Infof("Key %v does not exist in known objects store, placing DeleteFinalStateUnknown marker without object", k)
				}
			}
		}
		if err := f.queueActionLocked(Deleted, DeletedFinalStateUnknown{k, deletedObj}); err != nil {
			return err
		}
	}
	return nil
}

// A KeyLister is anything that knows how to list its keys.
type KeyLister interface {
	ListKeys() []string
}

// A KeyGetter is anything that knows how to get the value stored under a given key.
type KeyGetter interface {
	GetByKey(key string) (interface{}, bool, error)
}

// DeltaCompressor is an algorithm that removes redundant changes.
type DeltaCompressor interface {
	Compress(Deltas) Deltas
}

// DeltaCompressorFunc should remove redundant changes; but changes that
// are redundant depend on one's desired semantics, so this is an
// injectable function.
//
// DeltaCompressorFunc adapts a raw function to be a DeltaCompressor.
type DeltaCompressorFunc func(Deltas) Deltas

// Compress just calls dc.
func (dc DeltaCompressorFunc) Compress(d Deltas) Deltas {
	return dc(d)
}

// DeltaType is the type of a change (addition, deletion, etc)
type DeltaType string

const (
	Added   DeltaType = "Added"
	Updated DeltaType = "Updated"
	Deleted DeltaType = "Deleted"
	// The other types are obvious. You'll get Sync deltas when:
	//  * A watch expires/errors out and a new list/watch cycle is started.
	//  * You've turned on periodic syncs.
	// (Anything that trigger's DeltaFIFO's Replace() method.)
	Sync DeltaType = "Sync"
)

// Delta is the type stored by a DeltaFIFO. It tells you what change
// happened, and the object's state after* that change.
//
// [*] Unless the change is a deletion, and then you'll get the final
//     state of the object before it was deleted.
type Delta struct {
	Type   DeltaType
	Object interface{}
}

// Deltas is a list of one or more 'Delta's to an individual object.
// The oldest delta is at index 0, the newest delta is the last one.
type Deltas []Delta

// Oldest is a convenience function that returns the oldest delta, or
// nil if there are no deltas.
func (d Deltas) Oldest() *Delta {
	if len(d) > 0 {
		return &d[0]
	}
	return nil
}

// Newest is a convenience function that returns the newest delta, or
// nil if there are no deltas.
func (d Deltas) Newest() *Delta {
	if n := len(d); n > 0 {
		return &d[n-1]
	}
	return nil
}

// copyDeltas returns a shallow copy of d; that is, it copies the slice but not
// the objects in the slice. This allows Get/List to return an object that we
// know won't be clobbered by a subsequent call to a delta compressor.
func copyDeltas(d Deltas) Deltas {
	d2 := make(Deltas, len(d))
	copy(d2, d)
	return d2
}

// DeletedFinalStateUnknown is placed into a DeltaFIFO in the case where
// an object was deleted but the watch deletion event was missed. In this
// case we don't know the final "resting" state of the object, so there's
// a chance the included `Obj` is stale.
type DeletedFinalStateUnknown struct {
	Key string
	Obj interface{}
}
