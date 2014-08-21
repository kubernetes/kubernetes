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
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// FIFO receives adds and updates from a Reflector, and puts them in a queue for
// FIFO order processing. If multiple adds/updates of a single item happen while
// an item is in the queue before it has been processed, it will only be
// processed once, and when it is processed, the most recent version will be
// processed. This can't be done with a channel.
type FIFO struct {
	lock  sync.RWMutex
	cond  sync.Cond
	items map[string]interface{}
	queue []string
}

// Add inserts an item, and puts it in the queue.
func (f *FIFO) Add(id string, obj interface{}) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.items[id] = obj
	f.queue = append(f.queue, id)
	f.cond.Broadcast()
}

// Update updates an item, and adds it to the queue.
func (f *FIFO) Update(id string, obj interface{}) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.items[id] = obj
	f.queue = append(f.queue, id)
	f.cond.Broadcast()
}

// Delete removes an item. It doesn't add it to the queue, because
// this implementation assumes the consumer only cares about the objects,
// not the order in which they were created/added.
func (f *FIFO) Delete(id string) {
	f.lock.Lock()
	defer f.lock.Unlock()
	delete(f.items, id)
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

// Contains returns a util.StringSet containing all IDs of stored the items.
// This is a snapshot of a moment in time, and one should keep in mind that
// other go routines can add or remove items after you call this.
func (c *FIFO) Contains() util.StringSet {
	c.lock.RLock()
	defer c.lock.RUnlock()
	set := util.StringSet{}
	for id := range c.items {
		set.Insert(id)
	}
	return set
}

// Get returns the requested item, or sets exists=false.
func (f *FIFO) Get(id string) (item interface{}, exists bool) {
	f.lock.RLock()
	defer f.lock.RUnlock()
	item, exists = f.items[id]
	return item, exists
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

// NewFIFO returns a Store which can be used to queue up items to
// process.
func NewFIFO() *FIFO {
	f := &FIFO{
		items: map[string]interface{}{},
		queue: []string{},
	}
	f.cond.L = &f.lock
	return f
}
