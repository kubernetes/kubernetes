// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slices

import (
	"math/bits"

	"golang.org/x/exp/constraints"
)

// Sort sorts a slice of any ordered type in ascending order.
// Sort may fail to sort correctly when sorting slices of floating-point
// numbers containing Not-a-number (NaN) values.
// Use slices.SortFunc(x, func(a, b float64) bool {return a < b || (math.IsNaN(a) && !math.IsNaN(b))})
// instead if the input may contain NaNs.
func Sort[E constraints.Ordered](x []E) {
	n := len(x)
	pdqsortOrdered(x, 0, n, bits.Len(uint(n)))
}

// SortFunc sorts the slice x in ascending order as determined by the less function.
// This sort is not guaranteed to be stable.
//
// SortFunc requires that less is a strict weak ordering.
// See https://en.wikipedia.org/wiki/Weak_ordering#Strict_weak_orderings.
func SortFunc[E any](x []E, less func(a, b E) bool) {
	n := len(x)
	pdqsortLessFunc(x, 0, n, bits.Len(uint(n)), less)
}

// SortStable sorts the slice x while keeping the original order of equal
// elements, using less to compare elements.
func SortStableFunc[E any](x []E, less func(a, b E) bool) {
	stableLessFunc(x, len(x), less)
}

// IsSorted reports whether x is sorted in ascending order.
func IsSorted[E constraints.Ordered](x []E) bool {
	for i := len(x) - 1; i > 0; i-- {
		if x[i] < x[i-1] {
			return false
		}
	}
	return true
}

// IsSortedFunc reports whether x is sorted in ascending order, with less as the
// comparison function.
func IsSortedFunc[E any](x []E, less func(a, b E) bool) bool {
	for i := len(x) - 1; i > 0; i-- {
		if less(x[i], x[i-1]) {
			return false
		}
	}
	return true
}

// BinarySearch searches for target in a sorted slice and returns the position
// where target is found, or the position where target would appear in the
// sort order; it also returns a bool saying whether the target is really found
// in the slice. The slice must be sorted in increasing order.
func BinarySearch[E constraints.Ordered](x []E, target E) (int, bool) {
	// search returns the leftmost position where f returns true, or len(x) if f
	// returns false for all x. This is the insertion position for target in x,
	// and could point to an element that's either == target or not.
	pos := search(len(x), func(i int) bool { return x[i] >= target })
	if pos >= len(x) || x[pos] != target {
		return pos, false
	} else {
		return pos, true
	}
}

// BinarySearchFunc works like BinarySearch, but uses a custom comparison
// function. The slice must be sorted in increasing order, where "increasing" is
// defined by cmp. cmp(a, b) is expected to return an integer comparing the two
// parameters: 0 if a == b, a negative number if a < b and a positive number if
// a > b.
func BinarySearchFunc[E any](x []E, target E, cmp func(E, E) int) (int, bool) {
	pos := search(len(x), func(i int) bool { return cmp(x[i], target) >= 0 })
	if pos >= len(x) || cmp(x[pos], target) != 0 {
		return pos, false
	} else {
		return pos, true
	}
}

func search(n int, f func(int) bool) int {
	// Define f(-1) == false and f(n) == true.
	// Invariant: f(i-1) == false, f(j) == true.
	i, j := 0, n
	for i < j {
		h := int(uint(i+j) >> 1) // avoid overflow when computing h
		// i ≤ h < j
		if !f(h) {
			i = h + 1 // preserves f(i-1) == false
		} else {
			j = h // preserves f(j) == true
		}
	}
	// i == j, f(i-1) == false, and f(j) (= f(i)) == true  =>  answer is i.
	return i
}

type sortedHint int // hint for pdqsort when choosing the pivot

const (
	unknownHint sortedHint = iota
	increasingHint
	decreasingHint
)

// xorshift paper: https://www.jstatsoft.org/article/view/v008i14/xorshift.pdf
type xorshift uint64

func (r *xorshift) Next() uint64 {
	*r ^= *r << 13
	*r ^= *r >> 17
	*r ^= *r << 5
	return uint64(*r)
}

func nextPowerOfTwo(length int) uint {
	return 1 << bits.Len(uint(length))
}
