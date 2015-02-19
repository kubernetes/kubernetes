/*
Copyright 2014 Google Inc. All rights reserved.

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
)

// FIFO receives adds and updates from a Reflector, and puts them in a queue for
// FIFO order processing. If multiple adds/updates of a single item happen while
// an item is in the queue before it has been processed, it will only be
// processed once, and when it is processed, the most recent version will be
// processed. This can't be done with a channel.
type FIFO struct {
	lock sync.RWMutex
	cond sync.Cond
	// We depend on the property that items in the set are in the queue and vice versa.
	items map[string]interface{}
	queue []string
	// keyFunc is used to make the key used for queued item insertion and retrieval, and
	// should be deterministic.
	keyFunc KeyFunc
}

// Handler processes obj in the context of HandleNext and responds with an outcome
// dictating whether the FIFO should pop or retry the entry.
type Handler func(obj interface{}) HandlerOutcome

// HandlerOutcome informs HandleNext what to do with the result of a Handler execution.
type HandlerOutcome string

const (
	// PopOutcome means the next entry should be popped from the queue. For example,
	// if the entry was successfully handled, or if the entry has been retried more
	// than a maximum amount of times.
	PopOutcome HandlerOutcome = "Pop"
	// RetryHeadOutcome means the entry should be requeued at the head of the queue
	// so that it is handled during the next HandleNext invocation.
	RetryHeadOutcome HandlerOutcome = "RetryHead"
	// RetryHeadOutcome means the entry should be moved to the tail of the queue to
	// give any other queued entries a chance to be handled.
	RetryTailOutcome HandlerOutcome = "RetryTail"
)

// Add inserts an item, and puts it in the queue. The item is only enqueued
// if it doesn't already exist in the set.
func (f *FIFO) Add(obj interface{}) error {
	id, err := f.keyFunc(obj)
	if err != nil {
		return fmt.Errorf("couldn't create key for object: %v", err)
	}
	f.lock.Lock()
	defer f.lock.Unlock()
	if _, exists := f.items[id]; !exists {
		f.queue = append(f.queue, id)
	}
	f.items[id] = obj
	f.cond.Broadcast()
	return nil
}

// Update is the same as Add in this implementation.
func (f *FIFO) Update(obj interface{}) error {
	return f.Add(obj)
}

// Delete removes an item. It doesn't add it to the queue, because
// this implementation assumes the consumer only cares about the objects,
// not the order in which they were created/added.
func (f *FIFO) Delete(obj interface{}) error {
	id, err := f.keyFunc(obj)
	if err != nil {
		return fmt.Errorf("couldn't create key for object: %v", err)
	}
	f.lock.Lock()
	defer f.lock.Unlock()
	delete(f.items, id)
	return err
}

// List returns a list of all the items.
func (f *FIFO) List() []interface{} {
	f.lock.RLock()
	defer f.lock.RUnlock()
	list := make([]interface{}, 0, len(f.items))
	for _, item := range f.items {
		list = append(list, item)
	}
	return list
}

// Get returns the requested item, or sets exists=false.
func (f *FIFO) Get(obj interface{}) (item interface{}, exists bool, err error) {
	key, err := f.keyFunc(obj)
	if err != nil {
		return nil, false, fmt.Errorf("couldn't create key for object: %v", err)
	}
	return f.GetByKey(key)
}

// GetByKey returns the requested item, or sets exists=false.
func (f *FIFO) GetByKey(key string) (item interface{}, exists bool, err error) {
	f.lock.RLock()
	defer f.lock.RUnlock()
	item, exists = f.items[key]
	return item, exists, nil
}

// Pop waits until an item is ready and returns it. If multiple items are
// ready, they are returned in the order in which they were added/updated.
// The item is removed from the queue (and the store) before it is returned,
// so if you don't succesfully process it, you need to add it back with Add().
func (f *FIFO) Pop() interface{} {
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
		return item
	}
}

// Replace will delete the contents of 'f', using instead the given map.
// 'f' takes ownersip of the map, you should not reference the map again
// after calling this function. f's queue is reset, too; upon return, it
// will contain the items in the map, in no particular order.
func (f *FIFO) Replace(list []interface{}) error {
	items := map[string]interface{}{}
	for _, item := range list {
		key, err := f.keyFunc(item)
		if err != nil {
			return fmt.Errorf("couldn't create key for object: %v", err)
		}
		items[key] = item
	}

	f.lock.Lock()
	defer f.lock.Unlock()
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

// HandleNext gives a Handler an opportunity to do work with the next queue
// entry in the context of a queue lock. The HandlerOutcome returned from the
// Handler is used to decide whether to pop or safely re-requeue the entry at
// either the head or tail of the queue.
//
// The re-queue operations are considered safe as long as the FIFO user takes
// care to write to the queue with only a single producer: otherwise, there are
// no guarantees about event ordering.
func (f *FIFO) HandleNext(handler Handler) interface{} {
	obj := f.Pop()

	id, err := f.keyFunc(obj)
	if err != nil {
		// TODO: error handling
		return nil
	}

	outcome := handler(obj)

	f.lock.Lock()
	defer f.lock.Unlock()

	switch outcome {
	case PopOutcome:
		// Accept the already-peformed pop
	case RetryHeadOutcome:
		// Requeue at head
		if _, exists := f.items[id]; !exists {
			f.queue = append([]string{id}, f.queue...)
			f.items[id] = obj
			f.cond.Broadcast()
		}
	case RetryTailOutcome:
		// Requeue at tail
		if _, exists := f.items[id]; !exists {
			f.queue = append(f.queue, id)
			f.items[id] = obj
			f.cond.Broadcast()
		}
	default:
		// TODO: what should the default behavior be?
	}
	return obj
}

// NewFIFO returns a Store which can be used to queue up items to
// process.
func NewFIFO(keyFunc KeyFunc) *FIFO {
	f := &FIFO{
		items:   map[string]interface{}{},
		queue:   []string{},
		keyFunc: keyFunc,
	}
	f.cond.L = &f.lock
	return f
}
