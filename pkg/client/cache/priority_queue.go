/*
Copyright 2014 The Kubernetes Authors.

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
	"sync"

	"k8s.io/kubernetes/pkg/util/sets"
    "k8s.io/kubernetes/pkg/api/meta"
)

// PopProcessFunc is passed to Pop() method of Queue interface.
// It is supposed to process the element popped from the queue.
//type PopProcessFunc func(interface{}) error

// ErrRequeue may be returned by a PopProcessFunc to safely requeue
// the current item. The value of Err will be returned from Pop.
//type ErrRequeue struct {
//	// Err is returned by the Pop function
//	Err error
//}

//func (e ErrRequeue) Error() string {
//	if e.Err == nil {
//		return "the popped item should be requeued without returning an error"
//	}
//	return e.Err.Error()
//}

// Queue is exactly like a Store, but has a Pop() method too.
//type Queue interface {
//	Store
//
//	// Pop blocks until it has something to process.
//	// It returns the object that was process and the result of processing.
//	// The PopProcessFunc may return an ErrRequeue{...} to indicate the item
//	// should be requeued before releasing the lock on the queue.
//	Pop(PopProcessFunc) (interface{}, error)
//
//	// AddIfNotPresent adds a value previously
//	// returned by Pop back into the queue as long
//	// as nothing else (presumably more recent)
//	// has since been added.
//	AddIfNotPresent(interface{}) error
//
//	// Return true if the first batch of items has been popped
//	HasSynced() bool
//}

// Helper function for popping from Queue.
// WARNING: Do NOT use this function in non-test code to avoid races
// unless you really really really really know what you are doing.
//func Pop(queue Queue) interface{} {
//	var result interface{}
//	queue.Pop(func(obj interface{}) error {
//		result = obj
//		return nil
//	})
//	return result
//}

// PriorityQueue recieves adds and updates from a Reflector, and puts them in a queue for
// processing based on priorty. If multiple adds/updates of a single item happen while an
// item is in the queue before it has been processed, it will only be processed once, and
// when it is processed, the most recent version will be processed. This can't be done
// with a channel.
//
// PriorityQueue solves this use case:
//  * You want to process every object (exactly) once.
//  * You want to process the most recent version of the object when you process it.
//  * You do not want to process deleted objects, they should be removed from the queue.
//  * You need to periodically reorder the queue based on a priority function.
//  Compare with FIFO or DeltaFIFO for other use cases.
type PriorityQueue struct {
	lock sync.RWMutex
	cond sync.Cond
	// We depend on the property that items in the set are in the queue and vice versa.
	items map[string]interface{}
	queue []string

	// populated is true if the first batch of items inserted by Replace() has been populated
	// or Delete/Add/Update was called first.
	populated bool
	// initialPopulationCount is the number of items inserted by the first call of Replace()
	initialPopulationCount int

	// keyFunc is used to make the key used for queued item insertion and retrieval, and
	// should be deterministic.
	keyFunc KeyFunc
}

var (
	_ = Queue(&PriorityQueue{}) // PriorityQueue is a Queue
)

// Return true if an Add/Update/Delete/AddIfNotPresent are called first,
// or an Update called first but the first batch of items inserted by Replace() has been popped
func (f *PriorityQueue) HasSynced() bool {
	f.lock.Lock()
	defer f.lock.Unlock()
	return f.populated && f.initialPopulationCount == 0
}

// Add inserts an item, and puts it in the queue. The item is only enqueued
// if it doesn't already exist in the set.
// TODO: extract the priority, and add to the heap properly
func (f *PriorityQueue) Add(obj interface{}) error {
	id, err := f.keyFunc(obj)
	if err != nil {
		return KeyError{obj, err}
	}
	f.lock.Lock()
	defer f.lock.Unlock()
	f.populated = true
	if _, exists := f.items[id]; !exists {
		f.queue = append(f.queue, id)
	}
	f.items[id] = obj
	f.cond.Broadcast()
	return nil
}

// AddIfNotPresent inserts an item, and puts it in the queue. If the item is already
// present in the set, it is neither enqueued nor added to the set.
//
// This is useful in a single producer/consumer scenario so that the consumer can
// safely retry items without contending with the producer and potentially enqueueing
// stale items.
func (f *PriorityQueue) AddIfNotPresent(obj interface{}) error {
	id, err := f.keyFunc(obj)
	if err != nil {
		return KeyError{obj, err}
	}
	f.lock.Lock()
	defer f.lock.Unlock()
	f.addIfNotPresent(id, obj)
	return nil
}

// addIfNotPresent assumes the PriorityQueue lock is already held and adds the the provided
// item to the queue under id if it does not already exist.
// TODO: extract the priority
// TODO: use a heap and insert based on priority
func (f *PriorityQueue) addIfNotPresent(id string, obj interface{}) {
	f.populated = true
	if _, exists := f.items[id]; exists {
		return
	}

	f.queue = append(f.queue, id)
	f.items[id] = obj
	f.cond.Broadcast()
}

// Update is the same as Add in this implementation.
// what is this for?
func (f *PriorityQueue) Update(obj interface{}) error {
	return f.Add(obj)
}

// Delete removes an item. It doesn't add it to the queue, because
// this implementation assumes the consumer only cares about the objects,
// not the order in which they were created/added.
// what is this for?
func (f *PriorityQueue) Delete(obj interface{}) error {
	id, err := f.keyFunc(obj)
	if err != nil {
		return KeyError{obj, err}
	}
	f.lock.Lock()
	defer f.lock.Unlock()
	f.populated = true
	delete(f.items, id)
	return err
}

// List returns a list of all the items.
// TODO: return sorted list
func (f *PriorityQueue) List() []interface{} {
	f.lock.RLock()
	defer f.lock.RUnlock()
	list := make([]interface{}, 0, len(f.items))
	for _, item := range f.items {
		list = append(list, item)
	}
	return list
}

// ListKeys returns a list of all the keys of the objects currently
// in the PriorityQueue.
// TODO: return sorted list
func (f *PriorityQueue) ListKeys() []string {
	f.lock.RLock()
	defer f.lock.RUnlock()
	list := make([]string, 0, len(f.items))
	for key := range f.items {
		list = append(list, key)
	}
	return list
}

// Get returns the requested item, or sets exists=false.
func (f *PriorityQueue) Get(obj interface{}) (item interface{}, exists bool, err error) {
	key, err := f.keyFunc(obj)
	if err != nil {
		return nil, false, KeyError{obj, err}
	}
	return f.GetByKey(key)
}

// GetByKey returns the requested item, or sets exists=false.
func (f *PriorityQueue) GetByKey(key string) (item interface{}, exists bool, err error) {
	f.lock.RLock()
	defer f.lock.RUnlock()
	item, exists = f.items[key]
	return item, exists, nil
}

// Pop waits until an item is ready and processes it. If multiple items are
// ready, they are returned in the priority order.
// The item is removed from the queue (and the store) before it is processed,
// so if you don't successfully process it, it should be added back with
// AddIfNotPresent(). process function is called under lock, so it is safe
// update data structures in it that need to be in sync with the queue.
// TODO: check this forever loop to make sure it does the priority queue...
func (f *PriorityQueue) Pop(process PopProcessFunc) (interface{}, error) {
	f.lock.Lock()
	defer f.lock.Unlock()
	for {
		for len(f.queue) == 0 {
			f.cond.Wait()
		}
		id := f.queue[0]
		f.queue = f.queue[1:]
		if f.initialPopulationCount > 0 {
			f.initialPopulationCount--
		}
		item, ok := f.items[id]
		if !ok {
			// Item may have been deleted subsequently.
			continue
		}
		delete(f.items, id)
		err := process(item)
		if e, ok := err.(ErrRequeue); ok {
			f.addIfNotPresent(id, item)
			err = e.Err
		}
		return item, err
	}
}

// Replace will delete the contents of 'f', using instead the given map.
// 'f' takes ownership of the map, you should not reference the map again
// after calling this function. f's queue is reset, too; upon return, it
// will contain the items in the map, in no particular order.
// this is needed because the Store interface specifies it
func (f *PriorityQueue) Replace(list []interface{}, resourceVersion string) error {
	items := map[string]interface{}{}
	for _, item := range list {
		key, err := f.keyFunc(item)
		if err != nil {
			return KeyError{item, err}
		}
		items[key] = item
	}

	f.lock.Lock()
	defer f.lock.Unlock()

	if !f.populated {
		f.populated = true
		f.initialPopulationCount = len(items)
	}

	f.items = items
	f.queue = f.queue[:0]
	for id := range items {
		f.queue = append(f.queue, id)
	}
	if len(f.queue) > 0 {
		f.cond.Broadcast()
	}
	return nil
}

// Resync will touch all objects to put them into the processing queue
// TODO: make this a heap
func (f *PriorityQueue) Resync() error {
	f.lock.Lock()
	defer f.lock.Unlock()

	inQueue := sets.NewString()
	for _, id := range f.queue {
		inQueue.Insert(id)
	}
	for id := range f.items {
		if !inQueue.Has(id) {
			f.queue = append(f.queue, id)
		}
	}
	if len(f.queue) > 0 {
		f.cond.Broadcast()
	}
	return nil
}

// NewPriorityQueue returns a Store which can be used to queue up items to
// process.
func NewPriorityQueue(keyFunc KeyFunc) *PriorityQueue {
	f := &PriorityQueue{
		items:   map[string]interface{}{},
		queue:   []string{},
		keyFunc: keyFunc,
	}
	f.cond.L = &f.lock
	return f
}

// extracts the priority annotation of an object
// if the priority is not set, then set priority to -1
// TODO: add test
func MetaPriorityFunc(obj interface{}) (int, error) {
    if key, ok := obj.(ExplicitKey); ok {
        return string(key), nil
    }
    meta, err := meta.Accessor(obj)
    if err != nil {
        return -1, fmt.Errorf("object has no meta: %v", err)
    }
    var priority = -1
    if len(meta.GetAnnotations()) > 0 {
        annotations, err := strconv.Atoi(meta.GetAnnotations())
        if err != nil {
            return -1, fmt.Errorf("object does not have annotations", err) 
        }
        if p, ok := annotations["k8s_priority"]; ok {
            priority, err := strconv.Atoi(p)
            if err != nil {
                return -1, fmt.Errorf("priority is not an integer", err)
            }
            return priority, nil
        }
    }
    return -1, nil
}
