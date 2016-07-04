/*
Copyright 2015 The Kubernetes Authors.

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

package queue

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/util/sets"
)

type entry struct {
	value UniqueCopyable
	event EventType
}

type deletedEntry struct {
	*entry
	expiration time.Time
}

func (e *entry) Value() UniqueCopyable {
	return e.value
}

func (e *entry) Copy() Copyable {
	if e == nil {
		return nil
	}
	return &entry{e.value.Copy().(UniqueCopyable), e.event}
}

func (e *entry) Is(types EventType) bool {
	return types&e.event != 0
}

func (e *deletedEntry) Copy() Copyable {
	if e == nil {
		return nil
	}
	return &deletedEntry{e.entry.Copy().(*entry), e.expiration}
}

// deliver a message
type pigeon func(msg Entry)

func dead(msg Entry) {
	// intentionally blank
}

// HistoricalFIFO receives adds and updates from a Reflector, and puts them in a queue for
// FIFO order processing. If multiple adds/updates of a single item happen while
// an item is in the queue before it has been processed, it will only be
// processed once, and when it is processed, the most recent version will be
// processed. This can't be done with a channel.
type HistoricalFIFO struct {
	lock      sync.RWMutex
	cond      sync.Cond
	items     map[string]Entry // We depend on the property that items in the queue are in the set.
	queue     []string
	carrier   pigeon // may be dead, but never nil
	gcc       int
	lingerTTL time.Duration
}

// panics if obj doesn't implement UniqueCopyable; otherwise returns the same, typecast object
func checkType(obj interface{}) UniqueCopyable {
	if v, ok := obj.(UniqueCopyable); !ok {
		panic(fmt.Sprintf("Illegal object type, expected UniqueCopyable: %T", obj))
	} else {
		return v
	}
}

// Add inserts an item, and puts it in the queue. The item is only enqueued
// if it doesn't already exist in the set.
func (f *HistoricalFIFO) Add(v interface{}) error {
	obj := checkType(v)
	notifications := []Entry(nil)
	defer func() {
		for _, e := range notifications {
			f.carrier(e)
		}
	}()

	f.lock.Lock()
	defer f.lock.Unlock()

	id := obj.GetUID()
	if entry, exists := f.items[id]; !exists {
		f.queue = append(f.queue, id)
	} else {
		if entry.Is(DELETE_EVENT | POP_EVENT) {
			f.queue = append(f.queue, id)
		}
	}
	notifications = f.merge(id, obj)
	f.cond.Broadcast()
	return nil
}

// Update is the same as Add in this implementation.
func (f *HistoricalFIFO) Update(obj interface{}) error {
	return f.Add(obj)
}

// Delete removes an item. It doesn't add it to the queue, because
// this implementation assumes the consumer only cares about the objects,
// not the order in which they were created/added.
func (f *HistoricalFIFO) Delete(v interface{}) error {
	obj := checkType(v)
	deleteEvent := (Entry)(nil)
	defer func() {
		f.carrier(deleteEvent)
	}()

	f.lock.Lock()
	defer f.lock.Unlock()
	id := obj.GetUID()
	item, exists := f.items[id]
	if exists && !item.Is(DELETE_EVENT) {
		e := item.(*entry)
		e.event = DELETE_EVENT
		deleteEvent = &deletedEntry{e, time.Now().Add(f.lingerTTL)}
		f.items[id] = deleteEvent
	}
	return nil
}

// List returns a list of all the items.
func (f *HistoricalFIFO) List() []interface{} {
	f.lock.RLock()
	defer f.lock.RUnlock()

	// TODO(jdef): slightly overallocates b/c of deleted items
	list := make([]interface{}, 0, len(f.queue))

	for _, entry := range f.items {
		if entry.Is(DELETE_EVENT | POP_EVENT) {
			continue
		}
		list = append(list, entry.Value().Copy())
	}
	return list
}

// List returns a list of all the items.
func (f *HistoricalFIFO) ListKeys() []string {
	f.lock.RLock()
	defer f.lock.RUnlock()

	// TODO(jdef): slightly overallocates b/c of deleted items
	list := make([]string, 0, len(f.queue))

	for key, entry := range f.items {
		if entry.Is(DELETE_EVENT | POP_EVENT) {
			continue
		}
		list = append(list, key)
	}
	return list
}

// ContainedIDs returns a stringset.StringSet containing all IDs of the stored items.
// This is a snapshot of a moment in time, and one should keep in mind that
// other go routines can add or remove items after you call this.
func (c *HistoricalFIFO) ContainedIDs() sets.String {
	c.lock.RLock()
	defer c.lock.RUnlock()
	set := sets.String{}
	for id, entry := range c.items {
		if entry.Is(DELETE_EVENT | POP_EVENT) {
			continue
		}
		set.Insert(id)
	}
	return set
}

// Get returns the requested item, or sets exists=false.
func (f *HistoricalFIFO) Get(v interface{}) (interface{}, bool, error) {
	obj := checkType(v)
	return f.GetByKey(obj.GetUID())
}

// Get returns the requested item, or sets exists=false.
func (f *HistoricalFIFO) GetByKey(id string) (interface{}, bool, error) {
	f.lock.RLock()
	defer f.lock.RUnlock()
	entry, exists := f.items[id]
	if exists && !entry.Is(DELETE_EVENT|POP_EVENT) {
		return entry.Value().Copy(), true, nil
	}
	return nil, false, nil
}

// Get returns the requested item, or sets exists=false.
func (f *HistoricalFIFO) Poll(id string, t EventType) bool {
	f.lock.RLock()
	defer f.lock.RUnlock()
	entry, exists := f.items[id]
	return exists && entry.Is(t)
}

// Variant of DelayQueue.Pop() for UniqueDelayed items
func (q *HistoricalFIFO) Await(timeout time.Duration) interface{} {
	var (
		cancel = make(chan struct{})
		ch     = make(chan interface{}, 1)
		t      = time.NewTimer(timeout)
	)
	defer t.Stop()

	go func() { ch <- q.Pop(cancel) }()

	select {
	case <-t.C:
		close(cancel)
		return <-ch
	case x := <-ch:
		return x
	}
}

// Pop blocks until either there is an item available to dequeue or else the specified
// cancel chan is closed. Callers that have no interest in providing a cancel chan
// should specify nil, or else WithoutCancel() (for readability).
func (f *HistoricalFIFO) Pop(cancel <-chan struct{}) interface{} {
	popEvent := (Entry)(nil)
	defer func() {
		f.carrier(popEvent)
	}()

	f.lock.Lock()
	defer f.lock.Unlock()
	for {
		for len(f.queue) == 0 {
			signal := make(chan struct{})
			go func() {
				defer close(signal)
				f.cond.Wait()
			}()
			select {
			case <-cancel:
				// we may not have the lock yet, so
				// broadcast to abort Wait, then
				// return after lock re-acquisition
				f.cond.Broadcast()
				<-signal
				return nil
			case <-signal:
				// we have the lock, re-check
				// the queue for data...
			}
		}
		id := f.queue[0]
		f.queue = f.queue[1:]
		item, ok := f.items[id]
		if !ok || item.Is(DELETE_EVENT|POP_EVENT) {
			// Item may have been deleted subsequently.
			continue
		}
		value := item.Value()
		popEvent = &entry{value, POP_EVENT}
		f.items[id] = popEvent
		return value.Copy()
	}
}

func (f *HistoricalFIFO) Replace(objs []interface{}, resourceVersion string) error {
	notifications := make([]Entry, 0, len(objs))
	defer func() {
		for _, e := range notifications {
			f.carrier(e)
		}
	}()

	idToObj := make(map[string]interface{})
	for _, v := range objs {
		obj := checkType(v)
		idToObj[obj.GetUID()] = v
	}

	f.lock.Lock()
	defer f.lock.Unlock()

	f.queue = f.queue[:0]
	now := time.Now()
	for id, v := range f.items {
		if _, exists := idToObj[id]; !exists && !v.Is(DELETE_EVENT) {
			// a non-deleted entry in the items list that doesn't show up in the
			// new list: mark it as deleted
			ent := v.(*entry)
			ent.event = DELETE_EVENT
			e := &deletedEntry{ent, now.Add(f.lingerTTL)}
			f.items[id] = e
			notifications = append(notifications, e)
		}
	}
	for id, v := range idToObj {
		obj := checkType(v)
		f.queue = append(f.queue, id)
		n := f.merge(id, obj)
		notifications = append(notifications, n...)
	}
	if len(f.queue) > 0 {
		f.cond.Broadcast()
	}
	return nil
}

// garbage collect DELETEd items whose TTL has expired; the IDs of such items are removed
// from the queue. This impl assumes that caller has acquired state lock.
func (f *HistoricalFIFO) gc() {
	now := time.Now()
	deleted := make(map[string]struct{})
	for id, v := range f.items {
		if v.Is(DELETE_EVENT) {
			ent := v.(*deletedEntry)
			if ent.expiration.Before(now) {
				delete(f.items, id)
				deleted[id] = struct{}{}
			}
		}
	}
	// remove deleted items from the queue, will likely (slightly) overallocate here
	queue := make([]string, 0, len(f.queue))
	for _, id := range f.queue {
		if _, exists := deleted[id]; !exists {
			queue = append(queue, id)
		}
	}
	f.queue = queue
}

// Assumes that the caller has acquired the state lock.
func (f *HistoricalFIFO) merge(id string, obj UniqueCopyable) (notifications []Entry) {
	item, exists := f.items[id]
	if !exists || item.Is(POP_EVENT|DELETE_EVENT) {
		// no prior history for this UID, or else it was popped/removed by the client.
		e := &entry{obj.Copy().(UniqueCopyable), ADD_EVENT}
		f.items[id] = e
		notifications = append(notifications, e)
	} else if item.Value().GetUID() != obj.GetUID() {
		// sanity check, please
		panic(fmt.Sprintf("historical UID %q != current UID %v", item.Value().GetUID(), obj.GetUID()))
	} else {
		// exists && !(popped | deleted). so either the prior event was an add or an
		// update. reflect.DeepEqual is expensive. it won't help us determine if
		// we missed a hidden delete along the way.
		e := &entry{obj.Copy().(UniqueCopyable), UPDATE_EVENT}
		f.items[id] = e
		notifications = append(notifications, e)
		// else objects are the same, no work to do.
	}
	// check for garbage collection
	f.gcc++
	if f.gcc%256 == 0 { //TODO(jdef): extract constant
		f.gcc = 0
		f.gc()
	}
	return
}

// Resync will touch all objects to put them into the processing queue
func (f *HistoricalFIFO) Resync() error {
	// Nothing to do
	return nil
}

// NewHistorical returns a Store which can be used to queue up items to
// process. If a non-nil Mux is provided, then modifications to the
// the FIFO are delivered on a channel specific to this fifo.
func NewHistorical(ch chan<- Entry) *HistoricalFIFO {
	carrier := dead
	if ch != nil {
		carrier = func(msg Entry) {
			if msg != nil {
				ch <- msg.Copy().(Entry)
			}
		}
	}
	f := &HistoricalFIFO{
		items:     map[string]Entry{},
		queue:     []string{},
		carrier:   carrier,
		lingerTTL: 5 * time.Minute, // TODO(jdef): extract constant
	}
	f.cond.L = &f.lock
	return f
}
