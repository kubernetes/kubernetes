// Copyright 2014 The sortutil Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sortutil provides utilities supplementing the standard 'sort' package.
package sortutil

import "sort"

// ByteSlice attaches the methods of sort.Interface to []byte, sorting in increasing order.
type ByteSlice []byte

func (s ByteSlice) Len() int           { return len(s) }
func (s ByteSlice) Less(i, j int) bool { return s[i] < s[j] }
func (s ByteSlice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// Sort is a convenience method.
func (s ByteSlice) Sort() {
	sort.Sort(s)
}

// SearchBytes searches for x in a sorted slice of bytes and returns the index
// as specified by sort.Search. The slice must be sorted in ascending order.
func SearchBytes(a []byte, x byte) int {
	return sort.Search(len(a), func(i int) bool { return a[i] >= x })
}

// Float32Slice attaches the methods of sort.Interface to []float32, sorting in increasing order.
type Float32Slice []float32

func (s Float32Slice) Len() int           { return len(s) }
func (s Float32Slice) Less(i, j int) bool { return s[i] < s[j] }
func (s Float32Slice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// Sort is a convenience method.
func (s Float32Slice) Sort() {
	sort.Sort(s)
}

// SearchFloat32s searches for x in a sorted slice of float32 and returns the index
// as specified by sort.Search. The slice must be sorted in ascending order.
func SearchFloat32s(a []float32, x float32) int {
	return sort.Search(len(a), func(i int) bool { return a[i] >= x })
}

// Int8Slice attaches the methods of sort.Interface to []int8, sorting in increasing order.
type Int8Slice []int8

func (s Int8Slice) Len() int           { return len(s) }
func (s Int8Slice) Less(i, j int) bool { return s[i] < s[j] }
func (s Int8Slice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// Sort is a convenience method.
func (s Int8Slice) Sort() {
	sort.Sort(s)
}

// SearchInt8s searches for x in a sorted slice of int8 and returns the index
// as specified by sort.Search. The slice must be sorted in ascending order.
func SearchInt8s(a []int8, x int8) int {
	return sort.Search(len(a), func(i int) bool { return a[i] >= x })
}

// Int16Slice attaches the methods of sort.Interface to []int16, sorting in increasing order.
type Int16Slice []int16

func (s Int16Slice) Len() int           { return len(s) }
func (s Int16Slice) Less(i, j int) bool { return s[i] < s[j] }
func (s Int16Slice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// Sort is a convenience method.
func (s Int16Slice) Sort() {
	sort.Sort(s)
}

// SearchInt16s searches for x in a sorted slice of int16 and returns the index
// as specified by sort.Search. The slice must be sorted in ascending order.
func SearchInt16s(a []int16, x int16) int {
	return sort.Search(len(a), func(i int) bool { return a[i] >= x })
}

// Int32Slice attaches the methods of sort.Interface to []int32, sorting in increasing order.
type Int32Slice []int32

func (s Int32Slice) Len() int           { return len(s) }
func (s Int32Slice) Less(i, j int) bool { return s[i] < s[j] }
func (s Int32Slice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// Sort is a convenience method.
func (s Int32Slice) Sort() {
	sort.Sort(s)
}

// SearchInt32s searches for x in a sorted slice of int32 and returns the index
// as specified by sort.Search. The slice must be sorted in ascending order.
func SearchInt32s(a []int32, x int32) int {
	return sort.Search(len(a), func(i int) bool { return a[i] >= x })
}

// Int64Slice attaches the methods of sort.Interface to []int64, sorting in increasing order.
type Int64Slice []int64

func (s Int64Slice) Len() int           { return len(s) }
func (s Int64Slice) Less(i, j int) bool { return s[i] < s[j] }
func (s Int64Slice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// Sort is a convenience method.
func (s Int64Slice) Sort() {
	sort.Sort(s)
}

// SearchInt64s searches for x in a sorted slice of int64 and returns the index
// as specified by sort.Search. The slice must be sorted in ascending order.
func SearchInt64s(a []int64, x int64) int {
	return sort.Search(len(a), func(i int) bool { return a[i] >= x })
}

// UintSlice attaches the methods of sort.Interface to []uint, sorting in increasing order.
type UintSlice []uint

func (s UintSlice) Len() int           { return len(s) }
func (s UintSlice) Less(i, j int) bool { return s[i] < s[j] }
func (s UintSlice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// Sort is a convenience method.
func (s UintSlice) Sort() {
	sort.Sort(s)
}

// SearchUints searches for x in a sorted slice of uints and returns the index
// as specified by sort.Search. The slice must be sorted in ascending order.
func SearchUints(a []uint, x uint) int {
	return sort.Search(len(a), func(i int) bool { return a[i] >= x })
}

// Uint16Slice attaches the methods of sort.Interface to []uint16, sorting in increasing order.
type Uint16Slice []uint16

func (s Uint16Slice) Len() int           { return len(s) }
func (s Uint16Slice) Less(i, j int) bool { return s[i] < s[j] }
func (s Uint16Slice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// Sort is a convenience method.
func (s Uint16Slice) Sort() {
	sort.Sort(s)
}

// SearchUint16s searches for x in a sorted slice of uint16 and returns the index
// as specified by sort.Search. The slice must be sorted in ascending order.
func SearchUint16s(a []uint16, x uint16) int {
	return sort.Search(len(a), func(i int) bool { return a[i] >= x })
}

// Uint32Slice attaches the methods of sort.Interface to []uint32, sorting in increasing order.
type Uint32Slice []uint32

func (s Uint32Slice) Len() int           { return len(s) }
func (s Uint32Slice) Less(i, j int) bool { return s[i] < s[j] }
func (s Uint32Slice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// Sort is a convenience method.
func (s Uint32Slice) Sort() {
	sort.Sort(s)
}

// SearchUint32s searches for x in a sorted slice of uint32 and returns the index
// as specified by sort.Search. The slice must be sorted in ascending order.
func SearchUint32s(a []uint32, x uint32) int {
	return sort.Search(len(a), func(i int) bool { return a[i] >= x })
}

// Uint64Slice attaches the methods of sort.Interface to []uint64, sorting in increasing order.
type Uint64Slice []uint64

func (s Uint64Slice) Len() int           { return len(s) }
func (s Uint64Slice) Less(i, j int) bool { return s[i] < s[j] }
func (s Uint64Slice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// Sort is a convenience method.
func (s Uint64Slice) Sort() {
	sort.Sort(s)
}

// SearchUint64s searches for x in a sorted slice of uint64 and returns the index
// as specified by sort.Search. The slice must be sorted in ascending order.
func SearchUint64s(a []uint64, x uint64) int {
	return sort.Search(len(a), func(i int) bool { return a[i] >= x })
}

// RuneSlice attaches the methods of sort.Interface to []rune, sorting in increasing order.
type RuneSlice []rune

func (s RuneSlice) Len() int           { return len(s) }
func (s RuneSlice) Less(i, j int) bool { return s[i] < s[j] }
func (s RuneSlice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// Sort is a convenience method.
func (s RuneSlice) Sort() {
	sort.Sort(s)
}

// SearchRunes searches for x in a sorted slice of uint64 and returns the index
// as specified by sort.Search. The slice must be sorted in ascending order.
func SearchRunes(a []rune, x rune) int {
	return sort.Search(len(a), func(i int) bool { return a[i] >= x })
}

// Dedupe returns n, the number of distinct elements in data. The resulting
// elements are sorted in elements [0, n) or data[:n] for a slice.
func Dedupe(data sort.Interface) (n int) {
	if n = data.Len(); n < 2 {
		return n
	}

	sort.Sort(data)
	a, b := 0, 1
	for b < n {
		if data.Less(a, b) {
			a++
			if a != b {
				data.Swap(a, b)
			}
		}
		b++
	}
	return a + 1
}
