/*
Copyright 2017 The Kubernetes Authors.

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

package buffer

// defaultRingSize defines the default ring size if not specified
const defaultRingSize = 16

// RingGrowingOptions sets parameters for [RingGrowing] and
// [TypedRingGrowing].
type RingGrowingOptions struct {
	// InitialSize is the number of pre-allocated elements in the
	// initial underlying storage buffer.
	InitialSize int
}

// RingGrowing is a growing ring buffer.
// Not thread safe.
//
// Deprecated: Use TypedRingGrowing[any] instead.
type RingGrowing = TypedRingGrowing[any]

// NewRingGrowing constructs a new RingGrowing instance with provided parameters.
//
// Deprecated: Use NewTypedRingGrowing[any] instead.
func NewRingGrowing(initialSize int) *RingGrowing {
	return NewTypedRingGrowing[any](RingGrowingOptions{InitialSize: initialSize})
}

// TypedRingGrowing is a growing ring buffer.
// The zero value has an initial size of 0 and is ready to use.
// Not thread safe.
type TypedRingGrowing[T any] struct {
	data     []T
	n        int // Size of Data
	beg      int // First available element
	readable int // Number of data items available
}

// NewTypedRingGrowing constructs a new TypedRingGrowing instance with provided parameters.
func NewTypedRingGrowing[T any](opts RingGrowingOptions) *TypedRingGrowing[T] {
	return &TypedRingGrowing[T]{
		data: make([]T, opts.InitialSize),
		n:    opts.InitialSize,
	}
}

// ReadOne reads (consumes) first item from the buffer if it is available, otherwise returns false.
func (r *TypedRingGrowing[T]) ReadOne() (data T, ok bool) {
	if r.readable == 0 {
		return
	}
	r.readable--
	element := r.data[r.beg]
	var zero T
	r.data[r.beg] = zero // Remove reference to the object to help GC
	if r.beg == r.n-1 {
		// Was the last element
		r.beg = 0
	} else {
		r.beg++
	}
	return element, true
}

// WriteOne adds an item to the end of the buffer, growing it if it is full.
func (r *TypedRingGrowing[T]) WriteOne(data T) {
	if r.readable == r.n {
		// Time to grow
		newN := r.n * 2
		if newN == 0 {
			newN = defaultRingSize
		}
		newData := make([]T, newN)
		to := r.beg + r.readable
		if to <= r.n {
			copy(newData, r.data[r.beg:to])
		} else {
			copied := copy(newData, r.data[r.beg:])
			copy(newData[copied:], r.data[:(to%r.n)])
		}
		r.beg = 0
		r.data = newData
		r.n = newN
	}
	r.data[(r.readable+r.beg)%r.n] = data
	r.readable++
}

// Len returns the number of items in the buffer.
func (r *TypedRingGrowing[T]) Len() int {
	return r.readable
}

// Cap returns the capacity of the buffer.
func (r *TypedRingGrowing[T]) Cap() int {
	return r.n
}

// RingOptions sets parameters for [Ring].
type RingOptions struct {
	// InitialSize is the number of pre-allocated elements in the
	// initial underlying storage buffer.
	InitialSize int
	// NormalSize is the number of elements to allocate for new storage
	// buffers once the Ring is consumed and
	// can shrink again.
	NormalSize int
}

// Ring is a dynamically-sized ring buffer which can grow and shrink as-needed.
// The zero value has an initial size and normal size of 0 and is ready to use.
// Not thread safe.
type Ring[T any] struct {
	growing    TypedRingGrowing[T]
	normalSize int // Limits the size of the buffer that is kept for reuse. Read-only.
}

// NewRing constructs a new Ring instance with provided parameters.
func NewRing[T any](opts RingOptions) *Ring[T] {
	return &Ring[T]{
		growing:    *NewTypedRingGrowing[T](RingGrowingOptions{InitialSize: opts.InitialSize}),
		normalSize: opts.NormalSize,
	}
}

// ReadOne reads (consumes) first item from the buffer if it is available,
// otherwise returns false. When the buffer has been totally consumed and has
// grown in size beyond its normal size, it shrinks down to its normal size again.
func (r *Ring[T]) ReadOne() (data T, ok bool) {
	element, ok := r.growing.ReadOne()

	if r.growing.readable == 0 && r.growing.n > r.normalSize {
		// The buffer is empty. Reallocate a new buffer so the old one can be
		// garbage collected.
		r.growing.data = make([]T, r.normalSize)
		r.growing.n = r.normalSize
		r.growing.beg = 0
	}

	return element, ok
}

// WriteOne adds an item to the end of the buffer, growing it if it is full.
func (r *Ring[T]) WriteOne(data T) {
	r.growing.WriteOne(data)
}

// Len returns the number of items in the buffer.
func (r *Ring[T]) Len() int {
	return r.growing.Len()
}

// Cap returns the capacity of the buffer.
func (r *Ring[T]) Cap() int {
	return r.growing.Cap()
}
