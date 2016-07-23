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
    "fmt"
    "strconv"
    "container/heap"
    "errors"
)

// Priority recieves adds and updates from a Reflector, and puts them in a queue for
// processing based on priorty. If multiple adds/updates of a single item happen while an
// item is in the queue before it has been processed, it will only be processed once, and
// when it is processed, the most recent version will be processed.
//
// Priority solves this use case:
//  * You want to process every object (exactly) once.
//  * You want to process the most recent version of the object when you process it.
//  * You do not want to process deleted objects, they should be removed from the queue.
//  * You need to periodically reorder the queue based on a priority function.
//  Contrast with FIFO or DeltaFIFO for other use cases.

type Priority struct {
	lock sync.RWMutex
	cond sync.Cond
	// We depend on the property that items in the set are in the queue and vice versa.
    //items stores the actual objects in the queue
	items map[string]interface{}
    //queue keeps track of the order of the items
	queue []PriorityObject

	// populated is true if the first batch of items inserted by Replace() has been populated
	// or Delete/Add/Update was called first.
	populated bool
	// initialPopulationCount is the number of items inserted by the first call of Replace()
	initialPopulationCount int

	// keyFunc is used to make the key used for queued item insertion and retrieval, and
	// should be deterministic.
	keyFunc KeyFunc
}

type PriorityKey struct {
    key         string
    priority    int
    index       int
}

type PriorityQueue struct {
    //this is the underlying priority queue object
    //TODO: refactor everything to use this!!!
}

//Helper functions

// NewPriority returns a Store which can be used to queue up items to
// process.
func NewPriority(keyFunc KeyFunc) *Priority {
	pq := &Priority{
		items:   map[string]interface{}{},
		queue:   []PriorityKey,
		keyFunc: keyFunc,
	}
    heap.Init(&pq.queue)
	pq.cond.L = &f.lock
	return pq
}

func (pq Priority) GetPlaceInQueue(key string) (int, error) {
    for i, pk := range *pq.queue {
        if pk.key = key {
            return i, nil
        }
    }
    return -1, errors.New("key not found in queue")
}

//implement these func for heap:
//Len
func (pq Priority) Len() int {
    return len(pq.queue)
}
//Less
func (pq Priority) Less(i, j int) bool {
    //Pop should give us the highest priority item
    return pq.queue[i].priority > pq.queue[j].priority
}
//Swap
func (pq Priority) Swap(i, j int) {
    pq.queue[i], pq.queue[j] = pq.queue[j], pq.queue[i]
    pq.queue[i].index = i
    pq.queue[j].index = j
}
//Push
//adds an item to the end of the queue
func (pq *Priority) Push(obj interface{}) {
	key, err := pq.keyFunc(obj)
    priority, err := MetaPriorityFunc(obj)
    n := *pq.Len()
    pk := PriorityKey{
        key:        key,
        priority:   priority,
        index:      n
    }

    *pq.items[key] = obj
    *pq.queue = append(*pq.queue, pk)
    //heap.Fix(*pq.queue, pk.index) //this function is called by heap, so don't do this?
}
//Pop
//grabs the last item in the queue
func (pq *Priority) Pop() interface{} {
    //grab the queue
    old := *pq.queue
    n := len(old)
    item := old(n-1)
    item.index = -1 //for safety
    *pw.queue = old[0:n-1]

    //delete from items
    delete(*pq.items, item.key)

    return item
}
// Pop for queue...
// Pop waits until an item is ready and processes it. If multiple items are
// ready, they are returned in the priority order.
// The item is removed from the queue (and the store) before it is processed,
// so if you don't successfully process it, it should be added back with
// AddIfNotPresent(). process function is called under lock, so it is safe
// update data structures in it that need to be in sync with the queue.
// TODO: check this forever loop to make sure it does the priority queue...
//!!! omg how do I fix two conflicting interfaces?
//... I could rewrite a bunch of stuff so that the Priority merely has a pq inside it...
// this could have naming implications...
//maybe if I make this a private struct?
func (f *Priority) Pop(process PopProcessFunc) (interface{}, error) {
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

////update
////isn't actually needed by the interface
//func (pq *Priority) update(item *PriorityKey, priority int) {
//    item.priority = priority
//    heap.Fix(pq.queue, item.index)
//}
//implement these for Store
//Add
// Add inserts an item, and puts it in the queue. The item is only enqueued
// if it doesn't already exist in the set.
func (pq *Priority) Add(obj interface{}) error {
    return pq.AddIfNotPresent(obj)
}
//Update
//Update can modify any part of the object, especially it's priority
//However, if the keyfunc is not identical to the original object, then
//this acts like Add. The controller currently creates the key based on
//GetNamespace() and GetLabels(), so it should be safe to update anything
//else.
func (pq *Priority) Update(obj interface{}) error {
	key, err := pq.keyFunc(obj)

    //if it already exists, then update the object and don't add a new key
	if _, exists := pq.items[key]; exists {
        *pq.items[key] = obj
        //the item is already indexed, but might need a new priority
        index, err :=  pq.GetPlaceInQueue(key)
        priority, err := MetaPriorityFunc(obj)

        *pq.queue[index].priority = priority
        heap.Fix(pq, pk.index)
    } else {
        //if it doesn't already exist (or it has a new key), then add it
        heap.Push(&pq, obj) //I hope this actually works...
	}
    //TODO: fix error handling
    return err
}
//Delete
// Delete removes an item from the queue
func (pq *Priority) Delete(obj interface{}) error {
	id, err := pq.keyFunc(obj)
	if err != nil {
		return KeyError{obj, err}
	}
	pq.lock.Lock()
	defer pq.lock.Unlock()
	pq.populated = true

	delete(pq.items, id)
    i, err :=  pq.GetPlaceInQueue(key)
    pq.queue = append(pq.queue[:i], pq.queue[i+1:]...)

	return err
}

//List
// List returns a list of all the items in key order.
// This is NOT sorted by priority. //TODO: Should it be?
func (pq *Priority) List() []interface{} {
	pq.lock.RLock()
	defer pq.lock.RUnlock()
	list := make([]interface{}, 0, len(pq.items))
	for _, item := range pq.items {
		list = append(list, item)
	}
	return list
}

//ListKeys
// ListKeys returns a list of all the keys of the objects currently
// in the Priority. This is NOT sorted py priority. //TODO: Should it be?
func (pq *Priority) ListKeys() []string {
	pq.lock.RLock()
	defer pq.lock.RUnlock()
	list := make([]string, 0, len(pq.items))
	for item := range pq.items {
		list = append(list, item.key)
	}
	return list
}

//Get
// Get returns the requested item, or sets exists=false.
func (pq *Priority) Get(obj interface{}) (item interface{}, exists bool, err error) {
	key, err := pq.keyFunc(obj)
	if err != nil {
		return nil, false, KeyError{obj, err}
	}
	return pq.GetByKey(key)
}

//GetByKey
// GetByKey returns the requested item, or sets exists=false.
func (pq *Priority) GetByKey(key string) (item interface{}, exists bool, err error) {
	pq.lock.RLock()
	defer pq.lock.RUnlock()
	item, exists = pq.items[key]
	return item, exists, nil
}

//Replace
// Replace will delete the contents of 'pq', using instead the given map.
// 'pq' takes ownership of the map, you should not reference the map again
// after calling this function. pq's queue is reset, too; upon return, it
// will contain the items in the map, in priority order.
func (pq *Priority) Replace(list []interface{}, resourceVersion string) error {

    holder := NewPriority(pq.keyFunc)
	for _, item := range list {
       holder.Add(item)
    } 

	pq.lock.Lock()
	defer pq.lock.Unlock()

	if !pq.populated {
		pq.populated = true
		pq.initialPopulationCount = len(items)
	}

	pq.items = *holder.items
	pq.queue = *holder.queue

	if len(pq.queue) > 0 {
		pq.cond.Broadcast()
	}
	return nil
}

//Resync
// Resync will make sure all the items in the object map are in the queue
// it currently doesn't check if all items in the queue are in the map, so
// there could be dangling items in the queue... //TODO
func (pq *Priority) Resync() error {
    err := nil
	pq.lock.Lock()
	defer pq.lock.Unlock()

	inQueue := sets.NewString()
	for _, pk := range pq.queue {
		inQueue.Insert(pk.key)
	}
	for key, item := range pq.items {
		if !inQueue.Has(key) {
            priority, err := MetaPriorityFunc(item)
            n := len(*pq.queue)
            pk := PriorityKey{
                key:        key,
                priority:   priority,
                index:      n
            }

            heap.Push(*pq.queue, pk)
		}
	}
	if len(f.queue) > 0 {
		f.cond.Broadcast()
	}
	return err
}

//implement these for Queue
//Pop (duplicate)
//AddIfNotPresent
// AddIfNotPresent inserts an item, and puts it in the queue. If the item is already
// present in the set, it is neither enqueued nor added to the set.
//
// This is useful in a single producer/consumer scenario so that the consumer can
// safely retry items without contending with the producer and potentially enqueueing
// stale items.
func (pq *Priority) AddIfNotPresent(obj interface{}) error {
	id, err := pq.keyFunc(obj)
	if err != nil {
		return KeyError{obj, err}
	}
	pq.lock.Lock()
	defer pq.lock.Unlock()
	pq.populated = true
    //here's where the map + array are important...
	if _, exists := pq.items[id]; !exists {
        heap.Push(&pq, obj) //I hope this actually works...
	}
	pq.cond.Broadcast()

	return nil
}

//HasSynced
// Return true if an Add/Update/Delete/AddIfNotPresent are called first,
// or an Update called first but the first batch of items inserted by Replace() has been popped
func (pq *Priority) HasSynced() bool {
	pq.lock.Lock()
	defer pq.lock.Unlock()
	return pq.populated && f.initialPopulationCount == 0
}

var (
	_ = Queue(&Priority{}) // Priority is a Queue
)

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




const annotationKey = "k8s_priority"

// extracts the priority annotation of an object
// if the priority is not set, then set priority to -1
func MetaPriorityFunc(obj interface{}) (int, error) {
    meta, err := meta.Accessor(obj)
    if err != nil {
        return -1, fmt.Errorf("object has no meta: %v", err)
    }
    annotations := meta.GetAnnotations()
    if annotations == nil {
        return -1, fmt.Errorf("object does not have annotations") 
    }

    if p, ok := annotations[annotationKey]; ok {
        priority, err := strconv.Atoi(p)
        if err != nil {
            return -1, fmt.Errorf("priority is not an integer: %q", p)
        }
        return priority, nil
    }
    return -1, nil
}

//inherited from FIFO
// PopProcessFunc is passed to Pop() method of Queue interface.
// It is supposed to process the element popped from the queue.
// pulled in from FIFO
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

