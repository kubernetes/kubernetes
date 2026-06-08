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

// Below is the implementation of a generic heap. It does not perform
// synchronization. It leaves synchronization to the SchedulingQueue.

package heap

import (
	"container/heap"
	"fmt"

	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// Item is an interface that should be implemented by all items in the heap.
type Item interface {
	// Size returns the size of the item, used for metrics.
	Size() int
}

// KeyFunc is a function type to get the key from an object.
type KeyFunc[T Item] func(obj T) string

type heapItem[T Item] struct {
	// obj is the object which is stored in the heap.
	obj T
	// key is the key of the object for identity/lookup.
	key string
}

// data is an internal struct that implements the standard heap interface
// and keeps the data stored in the heap.
type data[T Item] struct {
	// queue is a heap-ordered slice of item pointers.
	queue []*heapItem[T]
	// keyIndex maps object keys to their index in queue for O(1) lookups.
	keyIndex map[string]int

	// keyFunc is used to make the key used for queued item insertion and retrieval, and
	// should be deterministic.
	keyFunc KeyFunc[T]
	// lessFunc is used to compare two objects in the heap.
	lessFunc LessFunc[T]
}

var (
	_ = heap.Interface(&data[Item]{}) // heapData is a standard heap
)

// Less compares two objects and returns true if the first one should go
// in front of the second one in the heap.
func (h *data[T]) Less(i, j int) bool {
	return h.lessFunc(h.queue[i].obj, h.queue[j].obj)
}

// Len returns the number of items in the Heap.
func (h *data[T]) Len() int { return len(h.queue) }

// Swap implements swapping of two elements in the heap. This is a part of standard
// heap interface and should never be called directly.
func (h *data[T]) Swap(i, j int) {
	h.queue[i], h.queue[j] = h.queue[j], h.queue[i]
	h.keyIndex[h.queue[i].key] = i
	h.keyIndex[h.queue[j].key] = j
}

// Push is supposed to be called by container/heap.Push only.
func (h *data[T]) Push(x interface{}) {
	item := x.(*heapItem[T])
	h.keyIndex[item.key] = len(h.queue)
	h.queue = append(h.queue, item)
}

// Pop is supposed to be called by container/heap.Pop only.
func (h *data[T]) Pop() interface{} {
	n := len(h.queue)
	if n == 0 {
		return nil
	}
	item := h.queue[n-1]
	h.queue[n-1] = nil // avoid memory leak
	h.queue = h.queue[:n-1]
	delete(h.keyIndex, item.key)
	return item.obj
}

// Peek returns the head of the heap without removing it.
func (h *data[T]) Peek() (T, bool) {
	if len(h.queue) > 0 {
		return h.queue[0].obj, true
	}
	return *new(T), false
}

// Heap is a producer/consumer queue that implements a heap data structure.
// It can be used to implement priority queues and similar data structures.
type Heap[T Item] struct {
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
	if idx, exists := h.data.keyIndex[key]; exists {
		h.data.queue[idx].obj = obj
		heap.Fix(h.data, idx)
	} else {
		heap.Push(h.data, &heapItem[T]{obj: obj, key: key})
		if h.metricRecorder != nil {
			h.metricRecorder.Add(obj.Size())
		}
	}
}

// Delete removes an item and returns the deleted obj.
func (h *Heap[T]) Delete(obj T) T {
	key := h.data.keyFunc(obj)
	if idx, ok := h.data.keyIndex[key]; ok {
		removed := heap.Remove(h.data, idx).(T)
		if h.metricRecorder != nil {
			h.metricRecorder.Add(-obj.Size())
		}
		return removed
	}
	var zero T
	return zero
}

// Peek returns the head of the heap without removing it.
func (h *Heap[T]) Peek() (T, bool) {
	return h.data.Peek()
}

// Pop returns the head of the heap and removes it.
func (h *Heap[T]) Pop() (T, error) {
	if h.data.Len() == 0 {
		return *new(T), fmt.Errorf("heap is empty")
	}
	obj := heap.Pop(h.data)
	typedObj := obj.(T)
	if h.metricRecorder != nil {
		h.metricRecorder.Add(-typedObj.Size())
	}
	return typedObj, nil
}

// Get returns the requested item, or sets exists=false.
func (h *Heap[T]) Get(obj T) (T, bool) {
	key := h.data.keyFunc(obj)
	return h.GetByKey(key)
}

// GetByKey returns the requested item, or sets exists=false.
func (h *Heap[T]) GetByKey(key string) (T, bool) {
	idx, exists := h.data.keyIndex[key]
	if !exists {
		return *new(T), false
	}
	return h.data.queue[idx].obj, true
}

func (h *Heap[T]) Has(obj T) bool {
	key := h.data.keyFunc(obj)
	_, ok := h.data.keyIndex[key]
	return ok
}

// List returns a list of all the items.
func (h *Heap[T]) List() []T {
	list := make([]T, 0, len(h.data.queue))
	for _, item := range h.data.queue {
		list = append(list, item.obj)
	}
	return list
}

// Len returns the number of items in the heap.
func (h *Heap[T]) Len() int {
	return len(h.data.queue)
}

// New returns a Heap which can be used to queue up items to process.
func New[T Item](keyFn KeyFunc[T], lessFn LessFunc[T]) *Heap[T] {
	return NewWithRecorder(keyFn, lessFn, nil)
}

// NewWithRecorder wraps an optional metricRecorder to compose a Heap object.
func NewWithRecorder[T Item](keyFn KeyFunc[T], lessFn LessFunc[T], metricRecorder metrics.MetricRecorder) *Heap[T] {
	return &Heap[T]{
		data: &data[T]{
			queue:    []*heapItem[T]{},
			keyIndex: map[string]int{},
			keyFunc:  keyFn,
			lessFunc: lessFn,
		},
		metricRecorder: metricRecorder,
	}
}

// LessFunc is a function that receives two items and returns true if the first
// item should be placed before the second one when the list is sorted.
type LessFunc[T Item] func(item1, item2 T) bool
