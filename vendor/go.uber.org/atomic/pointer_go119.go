// Copyright (c) 2022 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//go:build go1.19
// +build go1.19

package atomic

import "sync/atomic"

// Pointer is an atomic pointer of type *T.
type Pointer[T any] struct {
	_ nocmp // disallow non-atomic comparison
	p atomic.Pointer[T]
}

// NewPointer creates a new Pointer.
func NewPointer[T any](v *T) *Pointer[T] {
	var p Pointer[T]
	if v != nil {
		p.p.Store(v)
	}
	return &p
}

// Load atomically loads the wrapped value.
func (p *Pointer[T]) Load() *T {
	return p.p.Load()
}

// Store atomically stores the passed value.
func (p *Pointer[T]) Store(val *T) {
	p.p.Store(val)
}

// Swap atomically swaps the wrapped pointer and returns the old value.
func (p *Pointer[T]) Swap(val *T) (old *T) {
	return p.p.Swap(val)
}

// CompareAndSwap is an atomic compare-and-swap.
func (p *Pointer[T]) CompareAndSwap(old, new *T) (swapped bool) {
	return p.p.CompareAndSwap(old, new)
}
