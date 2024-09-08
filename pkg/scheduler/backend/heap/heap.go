/*
Copyright 2018 The Kubernetes Authors.

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

// Below is the implementation of the a heap. The logic is pretty much the same
// as cache.heap, however, this heap does not perform synchronization. It leaves
// synchronization to the SchedulingQueue.

package heap

import (
	"container/heap"
	"fmt"

	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// KeyFunc is a function type to get the key from an object.
type KeyFunc[T any] func(obj T) string

type heapItem[T any] struct {
	obj   T   // The object which is stored in the heap.
	index int // The index of the object's key in the Heap.queue.
}

type itemKeyValue[T any] struct {
	key string
	obj T
}

// data is an internal struct that implements the standard heap interface
// and keeps the data stored in the heap.
type data[T any] struct {
	// items is a map from key of the objects to the objects and their index.
	// We depend on the property that items in the map are in the queue and vice versa.
	items map[string]*heapItem[T]
	// queue implements a heap data structure and keeps the order of elements
	// according to the heap invariant. The queue keeps the keys of objects stored
	// in "items".
	queue []string

	// keyFunc is used to make the key used for queued item insertion and retrieval, and
	// should be deterministic.
	keyFunc KeyFunc[T]
	// lessFunc is used to compare two objects in the heap.
	lessFunc LessFunc[T]
}

var (
	_ = heap.Interface(&data[any]{}) // heapData is a standard heap
)

// Less compares two objects and returns true if the first one should go
// in front of the second one in the heap.
func (h *data[T]) Less(i, j int) bool {
	if i > len(h.queue) || j > len(h.queue) {
		return false
	}
	itemi, ok := h.items[h.queue[i]]
	if !ok {
		return false
	}
	itemj, ok := h.items[h.queue[j]]
	if !ok {
		return false
	}
	return h.lessFunc(itemi.obj, itemj.obj)
}

// Len returns the number of items in the Heap.
func (h *data[T]) Len() int { return len(h.queue) }

// Swap implements swapping of two elements in the heap. This is a part of standard
// heap interface and should never be called directly.
func (h *data[T]) Swap(i, j int) {
	if i < 0 || j < 0 {
		return
	}
	h.queue[i], h.queue[j] = h.queue[j], h.queue[i]
	item := h.items[h.queue[i]]
	item.index = i
	item = h.items[h.queue[j]]
	item.index = j
}

// Push is supposed to be called by container/heap.Push only.
func (h *data[T]) Push(kv interface{}) {
	keyValue := kv.(*itemKeyValue[T])
	n := len(h.queue)
	h.items[keyValue.key] = &heapItem[T]{keyValue.obj, n}
	h.queue = append(h.queue, keyValue.key)
}

// Pop is supposed to be called by container/heap.Pop only.
func (h *data[T]) Pop() interface{} {
	if len(h.queue) == 0 {
		return nil
	}
	key := h.queue[len(h.queue)-1]
	h.queue = h.queue[0 : len(h.queue)-1]
	item, ok := h.items[key]
	if !ok {
		// This is an error
		return nil
	}
	delete(h.items, key)
	return item.obj
}

// Peek returns the head of the heap without removing it.
func (h *data[T]) Peek() (T, bool) {
	if len(h.queue) > 0 {
		return h.items[h.queue[0]].obj, true
	}
	var zero T
	return zero, false
}

// Heap is a producer/consumer queue that implements a heap data structure.
// It can be used to implement priority queues and similar data structures.
type Heap[T any] struct {
	// data stores objects and has a queue that keeps their ordering according
	// to the heap invariant.
	data *data[T]
	// metricRecorder updates the counter when elements of a heap get added or
	// removed, and it does nothing if it's nil
	metricRecorder metrics.MetricRecorder
}

// AddOrUpdate inserts an item, and puts it in the queue. The item is updated if it
// already exists.
func (h *Heap[T]) AddOrUpdate(obj T) {
	key := h.data.keyFunc(obj)
	if _, exists := h.data.items[key]; exists {
		h.data.items[key].obj = obj
		heap.Fix(h.data, h.data.items[key].index)
	} else {
		heap.Push(h.data, &itemKeyValue[T]{key, obj})
		if h.metricRecorder != nil {
			h.metricRecorder.Inc()
		}
	}
}

// Delete removes an item.
func (h *Heap[T]) Delete(obj T) error {
	key := h.data.keyFunc(obj)
	if item, ok := h.data.items[key]; ok {
		heap.Remove(h.data, item.index)
		if h.metricRecorder != nil {
			h.metricRecorder.Dec()
		}
		return nil
	}
	return fmt.Errorf("object not found")
}

// Peek returns the head of the heap without removing it.
func (h *Heap[T]) Peek() (T, bool) {
	return h.data.Peek()
}

// Pop returns the head of the heap and removes it.
func (h *Heap[T]) Pop() (T, error) {
	obj := heap.Pop(h.data)
	if obj != nil {
		if h.metricRecorder != nil {
			h.metricRecorder.Dec()
		}
		return obj.(T), nil
	}
	var zero T
	return zero, fmt.Errorf("heap is empty")
}

// Get returns the requested item, or sets exists=false.
func (h *Heap[T]) Get(obj T) (T, bool) {
	key := h.data.keyFunc(obj)
	return h.GetByKey(key)
}

// GetByKey returns the requested item, or sets exists=false.
func (h *Heap[T]) GetByKey(key string) (T, bool) {
	item, exists := h.data.items[key]
	if !exists {
		var zero T
		return zero, false
	}
	return item.obj, true
}

func (h *Heap[T]) Has(obj T) bool {
	key := h.data.keyFunc(obj)
	_, ok := h.GetByKey(key)
	return ok
}

// List returns a list of all the items.
func (h *Heap[T]) List() []T {
	list := make([]T, 0, len(h.data.items))
	for _, item := range h.data.items {
		list = append(list, item.obj)
	}
	return list
}

// Len returns the number of items in the heap.
func (h *Heap[T]) Len() int {
	return len(h.data.queue)
}

// New returns a Heap which can be used to queue up items to process.
func New[T any](keyFn KeyFunc[T], lessFn LessFunc[T]) *Heap[T] {
	return NewWithRecorder(keyFn, lessFn, nil)
}

// NewWithRecorder wraps an optional metricRecorder to compose a Heap object.
func NewWithRecorder[T any](keyFn KeyFunc[T], lessFn LessFunc[T], metricRecorder metrics.MetricRecorder) *Heap[T] {
	return &Heap[T]{
		data: &data[T]{
			items:    map[string]*heapItem[T]{},
			queue:    []string{},
			keyFunc:  keyFn,
			lessFunc: lessFn,
		},
		metricRecorder: metricRecorder,
	}
}

// LessFunc is a function that receives two items and returns true if the first
// item should be placed before the second one when the list is sorted.
type LessFunc[T any] func(item1, item2 T) bool
