/*
Copyright 2025 The Kubernetes Authors.

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

import (
	"errors"
	"io"
)

// Compile-time check that *TypedRingFixed[byte] implements io.Writer.
var _ io.Writer = (*TypedRingFixed[byte])(nil)

// ErrInvalidSize indicates size must be > 0
var ErrInvalidSize = errors.New("size must be positive")

// TypedRingFixed is a fixed-size circular buffer for elements of type T.
// Writes overwrite older data, keeping only the last N elements.
// Not thread safe.
type TypedRingFixed[T any] struct {
	data        []T
	size        int
	writeCursor int
	written     int64
}

// NewTypedRingFixed creates a circular buffer with the given capacity (must be > 0).
func NewTypedRingFixed[T any](size int) (*TypedRingFixed[T], error) {
	if size <= 0 {
		return nil, ErrInvalidSize
	}
	return &TypedRingFixed[T]{
		data: make([]T, size),
		size: size,
	}, nil
}

// Write writes p to the buffer, overwriting old data if needed.
func (r *TypedRingFixed[T]) Write(p []T) (int, error) {
	originalLen := len(p)
	r.written += int64(originalLen)

	// If the input is larger than our buffer, only keep the last 'size' elements
	if originalLen > r.size {
		p = p[originalLen-r.size:]
	}

	// Copy data, handling wrap-around
	n := len(p)
	remain := r.size - r.writeCursor
	if n <= remain {
		copy(r.data[r.writeCursor:], p)
	} else {
		copy(r.data[r.writeCursor:], p[:remain])
		copy(r.data, p[remain:])
	}

	r.writeCursor = (r.writeCursor + n) % r.size
	return originalLen, nil
}

// Slice returns buffer contents in write order. Don't modify the returned slice.
func (r *TypedRingFixed[T]) Slice() []T {
	if r.written == 0 {
		return nil
	}

	// Buffer hasn't wrapped yet
	if r.written < int64(r.size) {
		return r.data[:r.writeCursor]
	}

	// Buffer has wrapped - need to return data in correct order
	// Data from writeCursor to end is oldest, data from 0 to writeCursor is newest
	if r.writeCursor == 0 {
		return r.data
	}

	out := make([]T, r.size)
	copy(out, r.data[r.writeCursor:])
	copy(out[r.size-r.writeCursor:], r.data[:r.writeCursor])
	return out
}

// Size returns the buffer capacity.
func (r *TypedRingFixed[T]) Size() int {
	return r.size
}

// Len returns how many elements are currently in the buffer.
func (r *TypedRingFixed[T]) Len() int {
	if r.written < int64(r.size) {
		return int(r.written)
	}
	return r.size
}

// TotalWritten returns total elements ever written (including overwritten ones).
func (r *TypedRingFixed[T]) TotalWritten() int64 {
	return r.written
}

// Reset clears the buffer.
func (r *TypedRingFixed[T]) Reset() {
	r.writeCursor = 0
	r.written = 0
}
