// Copyright (c) 2020-2022 Uber Technologies, Inc.
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
	"math"
	"strconv"
)

//go:generate bin/gen-atomicwrapper -name=Float64 -type=float64 -wrapped=Uint64 -pack=math.Float64bits -unpack=math.Float64frombits -swap -json -imports math -file=float64.go

// Add atomically adds to the wrapped float64 and returns the new value.
func (f *Float64) Add(delta float64) float64 {
	for {
		old := f.Load()
		new := old + delta
		if f.CAS(old, new) {
			return new
		}
	}
}

// Sub atomically subtracts from the wrapped float64 and returns the new value.
func (f *Float64) Sub(delta float64) float64 {
	return f.Add(-delta)
}

// CAS is an atomic compare-and-swap for float64 values.
//
// Deprecated: Use CompareAndSwap
func (f *Float64) CAS(old, new float64) (swapped bool) {
	return f.CompareAndSwap(old, new)
}

// CompareAndSwap is an atomic compare-and-swap for float64 values.
//
// Note: CompareAndSwap handles NaN incorrectly. NaN != NaN using Go's inbuilt operators
// but CompareAndSwap allows a stored NaN to compare equal to a passed in NaN.
// This avoids typical CompareAndSwap loops from blocking forever, e.g.,
//
//	for {
//	  old := atom.Load()
//	  new = f(old)
//	  if atom.CompareAndSwap(old, new) {
//	    break
//	  }
//	}
//
// If CompareAndSwap did not match NaN to match, then the above would loop forever.
func (f *Float64) CompareAndSwap(old, new float64) (swapped bool) {
	return f.v.CompareAndSwap(math.Float64bits(old), math.Float64bits(new))
}

// String encodes the wrapped value as a string.
func (f *Float64) String() string {
	// 'g' is the behavior for floats with %v.
	return strconv.FormatFloat(f.Load(), 'g', -1, 64)
}
