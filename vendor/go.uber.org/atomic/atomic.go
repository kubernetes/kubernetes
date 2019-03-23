// Copyright (c) 2016 Uber Technologies, Inc.
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

// Package atomic provides simple wrappers around numerics to enforce atomic
// access.
package atomic

import (
	"math"
	"sync/atomic"
	"time"
)

// Int32 is an atomic wrapper around an int32.
type Int32 struct{ v int32 }

// NewInt32 creates an Int32.
func NewInt32(i int32) *Int32 {
	return &Int32{i}
}

// Load atomically loads the wrapped value.
func (i *Int32) Load() int32 {
	return atomic.LoadInt32(&i.v)
}

// Add atomically adds to the wrapped int32 and returns the new value.
func (i *Int32) Add(n int32) int32 {
	return atomic.AddInt32(&i.v, n)
}

// Sub atomically subtracts from the wrapped int32 and returns the new value.
func (i *Int32) Sub(n int32) int32 {
	return atomic.AddInt32(&i.v, -n)
}

// Inc atomically increments the wrapped int32 and returns the new value.
func (i *Int32) Inc() int32 {
	return i.Add(1)
}

// Dec atomically decrements the wrapped int32 and returns the new value.
func (i *Int32) Dec() int32 {
	return i.Sub(1)
}

// CAS is an atomic compare-and-swap.
func (i *Int32) CAS(old, new int32) bool {
	return atomic.CompareAndSwapInt32(&i.v, old, new)
}

// Store atomically stores the passed value.
func (i *Int32) Store(n int32) {
	atomic.StoreInt32(&i.v, n)
}

// Swap atomically swaps the wrapped int32 and returns the old value.
func (i *Int32) Swap(n int32) int32 {
	return atomic.SwapInt32(&i.v, n)
}

// Int64 is an atomic wrapper around an int64.
type Int64 struct{ v int64 }

// NewInt64 creates an Int64.
func NewInt64(i int64) *Int64 {
	return &Int64{i}
}

// Load atomically loads the wrapped value.
func (i *Int64) Load() int64 {
	return atomic.LoadInt64(&i.v)
}

// Add atomically adds to the wrapped int64 and returns the new value.
func (i *Int64) Add(n int64) int64 {
	return atomic.AddInt64(&i.v, n)
}

// Sub atomically subtracts from the wrapped int64 and returns the new value.
func (i *Int64) Sub(n int64) int64 {
	return atomic.AddInt64(&i.v, -n)
}

// Inc atomically increments the wrapped int64 and returns the new value.
func (i *Int64) Inc() int64 {
	return i.Add(1)
}

// Dec atomically decrements the wrapped int64 and returns the new value.
func (i *Int64) Dec() int64 {
	return i.Sub(1)
}

// CAS is an atomic compare-and-swap.
func (i *Int64) CAS(old, new int64) bool {
	return atomic.CompareAndSwapInt64(&i.v, old, new)
}

// Store atomically stores the passed value.
func (i *Int64) Store(n int64) {
	atomic.StoreInt64(&i.v, n)
}

// Swap atomically swaps the wrapped int64 and returns the old value.
func (i *Int64) Swap(n int64) int64 {
	return atomic.SwapInt64(&i.v, n)
}

// Uint32 is an atomic wrapper around an uint32.
type Uint32 struct{ v uint32 }

// NewUint32 creates a Uint32.
func NewUint32(i uint32) *Uint32 {
	return &Uint32{i}
}

// Load atomically loads the wrapped value.
func (i *Uint32) Load() uint32 {
	return atomic.LoadUint32(&i.v)
}

// Add atomically adds to the wrapped uint32 and returns the new value.
func (i *Uint32) Add(n uint32) uint32 {
	return atomic.AddUint32(&i.v, n)
}

// Sub atomically subtracts from the wrapped uint32 and returns the new value.
func (i *Uint32) Sub(n uint32) uint32 {
	return atomic.AddUint32(&i.v, ^(n - 1))
}

// Inc atomically increments the wrapped uint32 and returns the new value.
func (i *Uint32) Inc() uint32 {
	return i.Add(1)
}

// Dec atomically decrements the wrapped int32 and returns the new value.
func (i *Uint32) Dec() uint32 {
	return i.Sub(1)
}

// CAS is an atomic compare-and-swap.
func (i *Uint32) CAS(old, new uint32) bool {
	return atomic.CompareAndSwapUint32(&i.v, old, new)
}

// Store atomically stores the passed value.
func (i *Uint32) Store(n uint32) {
	atomic.StoreUint32(&i.v, n)
}

// Swap atomically swaps the wrapped uint32 and returns the old value.
func (i *Uint32) Swap(n uint32) uint32 {
	return atomic.SwapUint32(&i.v, n)
}

// Uint64 is an atomic wrapper around a uint64.
type Uint64 struct{ v uint64 }

// NewUint64 creates a Uint64.
func NewUint64(i uint64) *Uint64 {
	return &Uint64{i}
}

// Load atomically loads the wrapped value.
func (i *Uint64) Load() uint64 {
	return atomic.LoadUint64(&i.v)
}

// Add atomically adds to the wrapped uint64 and returns the new value.
func (i *Uint64) Add(n uint64) uint64 {
	return atomic.AddUint64(&i.v, n)
}

// Sub atomically subtracts from the wrapped uint64 and returns the new value.
func (i *Uint64) Sub(n uint64) uint64 {
	return atomic.AddUint64(&i.v, ^(n - 1))
}

// Inc atomically increments the wrapped uint64 and returns the new value.
func (i *Uint64) Inc() uint64 {
	return i.Add(1)
}

// Dec atomically decrements the wrapped uint64 and returns the new value.
func (i *Uint64) Dec() uint64 {
	return i.Sub(1)
}

// CAS is an atomic compare-and-swap.
func (i *Uint64) CAS(old, new uint64) bool {
	return atomic.CompareAndSwapUint64(&i.v, old, new)
}

// Store atomically stores the passed value.
func (i *Uint64) Store(n uint64) {
	atomic.StoreUint64(&i.v, n)
}

// Swap atomically swaps the wrapped uint64 and returns the old value.
func (i *Uint64) Swap(n uint64) uint64 {
	return atomic.SwapUint64(&i.v, n)
}

// Bool is an atomic Boolean.
type Bool struct{ v uint32 }

// NewBool creates a Bool.
func NewBool(initial bool) *Bool {
	return &Bool{boolToInt(initial)}
}

// Load atomically loads the Boolean.
func (b *Bool) Load() bool {
	return truthy(atomic.LoadUint32(&b.v))
}

// CAS is an atomic compare-and-swap.
func (b *Bool) CAS(old, new bool) bool {
	return atomic.CompareAndSwapUint32(&b.v, boolToInt(old), boolToInt(new))
}

// Store atomically stores the passed value.
func (b *Bool) Store(new bool) {
	atomic.StoreUint32(&b.v, boolToInt(new))
}

// Swap sets the given value and returns the previous value.
func (b *Bool) Swap(new bool) bool {
	return truthy(atomic.SwapUint32(&b.v, boolToInt(new)))
}

// Toggle atomically negates the Boolean and returns the previous value.
func (b *Bool) Toggle() bool {
	return truthy(atomic.AddUint32(&b.v, 1) - 1)
}

func truthy(n uint32) bool {
	return n&1 == 1
}

func boolToInt(b bool) uint32 {
	if b {
		return 1
	}
	return 0
}

// Float64 is an atomic wrapper around float64.
type Float64 struct {
	v uint64
}

// NewFloat64 creates a Float64.
func NewFloat64(f float64) *Float64 {
	return &Float64{math.Float64bits(f)}
}

// Load atomically loads the wrapped value.
func (f *Float64) Load() float64 {
	return math.Float64frombits(atomic.LoadUint64(&f.v))
}

// Store atomically stores the passed value.
func (f *Float64) Store(s float64) {
	atomic.StoreUint64(&f.v, math.Float64bits(s))
}

// Add atomically adds to the wrapped float64 and returns the new value.
func (f *Float64) Add(s float64) float64 {
	for {
		old := f.Load()
		new := old + s
		if f.CAS(old, new) {
			return new
		}
	}
}

// Sub atomically subtracts from the wrapped float64 and returns the new value.
func (f *Float64) Sub(s float64) float64 {
	return f.Add(-s)
}

// CAS is an atomic compare-and-swap.
func (f *Float64) CAS(old, new float64) bool {
	return atomic.CompareAndSwapUint64(&f.v, math.Float64bits(old), math.Float64bits(new))
}

// Duration is an atomic wrapper around time.Duration
// https://godoc.org/time#Duration
type Duration struct {
	v Int64
}

// NewDuration creates a Duration.
func NewDuration(d time.Duration) *Duration {
	return &Duration{v: *NewInt64(int64(d))}
}

// Load atomically loads the wrapped value.
func (d *Duration) Load() time.Duration {
	return time.Duration(d.v.Load())
}

// Store atomically stores the passed value.
func (d *Duration) Store(n time.Duration) {
	d.v.Store(int64(n))
}

// Add atomically adds to the wrapped time.Duration and returns the new value.
func (d *Duration) Add(n time.Duration) time.Duration {
	return time.Duration(d.v.Add(int64(n)))
}

// Sub atomically subtracts from the wrapped time.Duration and returns the new value.
func (d *Duration) Sub(n time.Duration) time.Duration {
	return time.Duration(d.v.Sub(int64(n)))
}

// Swap atomically swaps the wrapped time.Duration and returns the old value.
func (d *Duration) Swap(n time.Duration) time.Duration {
	return time.Duration(d.v.Swap(int64(n)))
}

// CAS is an atomic compare-and-swap.
func (d *Duration) CAS(old, new time.Duration) bool {
	return d.v.CAS(int64(old), int64(new))
}

// Value shadows the type of the same name from sync/atomic
// https://godoc.org/sync/atomic#Value
type Value struct{ atomic.Value }
