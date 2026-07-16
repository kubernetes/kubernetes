// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slices

import (
	"cmp"
	"slices"
)

// Sort sorts a slice of any ordered type in ascending order.
// When sorting floating-point numbers, NaNs are ordered before other values.
//
//go:fix inline
func Sort[S ~[]E, E cmp.Ordered](x S) {
	slices.Sort(x)
}

// SortFunc sorts the slice x in ascending order as determined by the cmp
// function. This sort is not guaranteed to be stable.
// cmp(a, b) should return a negative number when a < b, a positive number when
// a > b and zero when a == b or when a is not comparable to b in the sense
// of the formal definition of Strict Weak Ordering.
//
// SortFunc requires that cmp is a strict weak ordering.
// See https://en.wikipedia.org/wiki/Weak_ordering#Strict_weak_orderings.
// To indicate 'uncomparable', return 0 from the function.
//
//go:fix inline
func SortFunc[S ~[]E, E any](x S, cmp func(a, b E) int) {
	slices.SortFunc(x, cmp)
}

// SortStableFunc sorts the slice x while keeping the original order of equal
// elements, using cmp to compare elements in the same way as [SortFunc].
//
//go:fix inline
func SortStableFunc[S ~[]E, E any](x S, cmp func(a, b E) int) {
	slices.SortStableFunc(x, cmp)
}

// IsSorted reports whether x is sorted in ascending order.
//
//go:fix inline
func IsSorted[S ~[]E, E cmp.Ordered](x S) bool {
	return slices.IsSorted(x)
}

// IsSortedFunc reports whether x is sorted in ascending order, with cmp as the
// comparison function as defined by [SortFunc].
//
//go:fix inline
func IsSortedFunc[S ~[]E, E any](x S, cmp func(a, b E) int) bool {
	return slices.IsSortedFunc(x, cmp)
}

// Min returns the minimal value in x. It panics if x is empty.
// For floating-point numbers, Min propagates NaNs (any NaN value in x
// forces the output to be NaN).
//
//go:fix inline
func Min[S ~[]E, E cmp.Ordered](x S) E {
	return slices.Min(x)
}

// MinFunc returns the minimal value in x, using cmp to compare elements.
// It panics if x is empty. If there is more than one minimal element
// according to the cmp function, MinFunc returns the first one.
//
//go:fix inline
func MinFunc[S ~[]E, E any](x S, cmp func(a, b E) int) E {
	return slices.MinFunc(x, cmp)
}

// Max returns the maximal value in x. It panics if x is empty.
// For floating-point E, Max propagates NaNs (any NaN value in x
// forces the output to be NaN).
//
//go:fix inline
func Max[S ~[]E, E cmp.Ordered](x S) E {
	return slices.Max(x)
}

// MaxFunc returns the maximal value in x, using cmp to compare elements.
// It panics if x is empty. If there is more than one maximal element
// according to the cmp function, MaxFunc returns the first one.
//
//go:fix inline
func MaxFunc[S ~[]E, E any](x S, cmp func(a, b E) int) E {
	return slices.MaxFunc(x, cmp)
}

// BinarySearch searches for target in a sorted slice and returns the position
// where target is found, or the position where target would appear in the
// sort order; it also returns a bool saying whether the target is really found
// in the slice. The slice must be sorted in increasing order.
//
//go:fix inline
func BinarySearch[S ~[]E, E cmp.Ordered](x S, target E) (int, bool) {
	return slices.BinarySearch(x, target)
}

// BinarySearchFunc works like [BinarySearch], but uses a custom comparison
// function. The slice must be sorted in increasing order, where "increasing"
// is defined by cmp. cmp should return 0 if the slice element matches
// the target, a negative number if the slice element precedes the target,
// or a positive number if the slice element follows the target.
// cmp must implement the same ordering as the slice, such that if
// cmp(a, t) < 0 and cmp(b, t) >= 0, then a must precede b in the slice.
//
//go:fix inline
func BinarySearchFunc[S ~[]E, E, T any](x S, target T, cmp func(E, T) int) (int, bool) {
	return slices.BinarySearchFunc(x, target, cmp)
}
