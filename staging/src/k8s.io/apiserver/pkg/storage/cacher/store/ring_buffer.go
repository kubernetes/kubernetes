/*
Copyright The Kubernetes Authors.

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

package store

// ringBuffer stores entries in revision order using a circular buffer.
// Its layout follows the etcd cache ring buffer:
// https://github.com/etcd-io/etcd/blob/main/cache/ringbuffer.go
// TODO: Consider moving this generic implementation to k8s.io/utils
type ringBuffer[T any] struct {
	buffer []entry[T]
	// head is the index immediately after the last non-empty entry in the buffer (i.e., the next write position).
	head, tail, size int
	revisionOf       RevisionOf[T]
}

type entry[T any] struct {
	revision int64
	item     T
}

type (
	KeyPredicate      = func([]byte) bool
	RevisionOf[T any] func(T) int64
	IterFunc[T any]   func(rev int64, item T) bool
)

func newRingBuffer[T any](capacity int, revisionOf RevisionOf[T]) *ringBuffer[T] {
	// assume capacity > 0 – validated by Cache
	return &ringBuffer[T]{
		buffer:     make([]entry[T], capacity),
		revisionOf: revisionOf,
	}
}

func (r *ringBuffer[T]) Append(item T) {
	entry := entry[T]{revision: r.revisionOf(item), item: item}
	if r.full() {
		r.tail = (r.tail + 1) % len(r.buffer)
	} else {
		r.size++
	}
	r.buffer[r.head] = entry
	r.head = (r.head + 1) % len(r.buffer)
}

func (r *ringBuffer[T]) full() bool {
	return r.size == len(r.buffer)
}

// AscendGreaterOrEqual iterates through entries in ascending order starting from the first entry with revision >= pivot.
func (r *ringBuffer[T]) AscendGreaterOrEqual(pivot int64, iter IterFunc[T]) {
	for i := r.findFirstIndexGreaterOrEqual(pivot); i < r.size; i++ {
		entry := r.at(i)
		if !iter(entry.revision, entry.item) {
			return
		}
	}
}

// AscendLessThan iterates in ascending order over entries with revision < pivot.
func (r *ringBuffer[T]) AscendLessThan(pivot int64, iter IterFunc[T]) {
	for i := 0; i < r.findFirstIndexGreaterOrEqual(pivot); i++ {
		entry := r.at(i)
		if !iter(entry.revision, entry.item) {
			return
		}
	}
}

// DescendGreaterThan iterates in descending order over entries with revision > pivot.
func (r *ringBuffer[T]) DescendGreaterThan(pivot int64, iter IterFunc[T]) {
	for i := r.size - 1; i > r.findLastIndexLessOrEqual(pivot); i-- {
		entry := r.at(i)
		if !iter(entry.revision, entry.item) {
			return
		}
	}
}

// DescendLessOrEqual iterates in descending order over entries with revision <= pivot.
func (r *ringBuffer[T]) DescendLessOrEqual(pivot int64, iter IterFunc[T]) {
	for i := r.findLastIndexLessOrEqual(pivot); i >= 0; i-- {
		entry := r.at(i)
		if !iter(entry.revision, entry.item) {
			return
		}
	}
}

// PeekLatest returns the most recently-appended revision (or 0 if empty).
func (r *ringBuffer[T]) PeekLatest() int64 {
	if r.size == 0 {
		return 0
	}
	idx := (r.head - 1 + len(r.buffer)) % len(r.buffer)
	return r.buffer[idx].revision
}

// PeekOldest returns the oldest revision currently stored (or 0 if empty).
func (r *ringBuffer[T]) PeekOldest() int64 {
	if r.size == 0 {
		return 0
	}
	return r.buffer[r.tail].revision
}

func (r *ringBuffer[T]) RebaseHistory() {
	r.head, r.tail, r.size = 0, 0, 0
	for i := range r.buffer {
		r.buffer[i] = entry[T]{}
	}
}

func (r *ringBuffer[T]) moduloIndex(index int) int {
	return (index + len(r.buffer)) % len(r.buffer)
}

func (r *ringBuffer[T]) at(logicalIndex int) entry[T] {
	return r.buffer[r.moduloIndex(r.tail+logicalIndex)]
}

func (r *ringBuffer[T]) findFirstIndexGreaterOrEqual(pivot int64) int {
	left, right := 0, r.size-1
	for left <= right {
		// Prevent overflow; see https://github.com/golang/go/blob/master/src/sort/search.go#L105.
		mid := int(uint(left+right) >> 1)
		if r.at(mid).revision >= pivot {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return left
}

func (r *ringBuffer[T]) findLastIndexLessOrEqual(pivot int64) int {
	left, right := 0, r.size-1
	for left <= right {
		// Prevent overflow; see https://github.com/golang/go/blob/master/src/sort/search.go#L105.
		mid := int(uint(left+right) >> 1)
		if r.at(mid).revision <= pivot {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return right
}

// Len returns the number of entries currently stored in the buffer.
func (r *ringBuffer[T]) Len() int {
	return r.size
}

// ReplaceLatest overwrites the most recently appended entry with the given item.
// Panics if the buffer is empty.
func (r *ringBuffer[T]) ReplaceLatest(item T) {
	if r.size == 0 {
		panic("ringBuffer.ReplaceLatest called on empty buffer")
	}
	r.buffer[r.moduloIndex(r.head-1)] = entry[T]{revision: r.revisionOf(item), item: item}
}

// RemoveLess drops all entries whose revision is strictly less than rv.
func (r *ringBuffer[T]) RemoveLess(rv int64) {
	if r.size == 0 {
		return
	}
	idx := r.findFirstIndexGreaterOrEqual(rv)
	if idx <= 0 {
		return
	}
	if idx == r.size {
		r.RebaseHistory()
		return
	}
	for i := range idx {
		r.buffer[r.moduloIndex(r.tail+i)] = entry[T]{}
	}
	r.tail = r.moduloIndex(r.tail + idx)
	r.size -= idx
}

// ensureCapacity grows the underlying slice (doubling) until it can hold at
// least minCapacity entries. After expansion the logical entries are laid out
// contiguously starting at index 0.
func (r *ringBuffer[T]) ensureCapacity(minCapacity int) {
	if len(r.buffer) >= minCapacity {
		return
	}
	newCap := len(r.buffer)
	if newCap == 0 {
		newCap = 1
	}
	for newCap < minCapacity {
		newCap *= 2
	}
	newBuffer := make([]entry[T], newCap)
	for i := 0; i < r.size; i++ {
		newBuffer[i] = r.at(i)
	}
	r.buffer = newBuffer
	r.tail = 0
	r.head = r.size
}
