// Copyright (c) 2021-2022 Uber Technologies, Inc.
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

package atomic

import (
	"sync/atomic"
	"unsafe"
)

// UnsafePointer is an atomic wrapper around unsafe.Pointer.
type UnsafePointer struct {
	_ nocmp // disallow non-atomic comparison

	v unsafe.Pointer
}

// NewUnsafePointer creates a new UnsafePointer.
func NewUnsafePointer(val unsafe.Pointer) *UnsafePointer {
	return &UnsafePointer{v: val}
}

// Load atomically loads the wrapped value.
func (p *UnsafePointer) Load() unsafe.Pointer {
	return atomic.LoadPointer(&p.v)
}

// Store atomically stores the passed value.
func (p *UnsafePointer) Store(val unsafe.Pointer) {
	atomic.StorePointer(&p.v, val)
}

// Swap atomically swaps the wrapped unsafe.Pointer and returns the old value.
func (p *UnsafePointer) Swap(val unsafe.Pointer) (old unsafe.Pointer) {
	return atomic.SwapPointer(&p.v, val)
}

// CAS is an atomic compare-and-swap.
//
// Deprecated: Use CompareAndSwap
func (p *UnsafePointer) CAS(old, new unsafe.Pointer) (swapped bool) {
	return p.CompareAndSwap(old, new)
}

// CompareAndSwap is an atomic compare-and-swap.
func (p *UnsafePointer) CompareAndSwap(old, new unsafe.Pointer) (swapped bool) {
	return atomic.CompareAndSwapPointer(&p.v, old, new)
}
