// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE.md file.

// Package cmpopts provides common options for the cmp package.
package cmpopts

import (
	"math"
	"reflect"

	"github.com/google/go-cmp/cmp"
)

func equateAlways(_, _ interface{}) bool { return true }

// EquateEmpty returns a Comparer option that determines all maps and slices
// with a length of zero to be equal, regardless of whether they are nil.
//
// EquateEmpty can be used in conjunction with SortSlices and SortMaps.
func EquateEmpty() cmp.Option {
	return cmp.FilterValues(isEmpty, cmp.Comparer(equateAlways))
}

func isEmpty(x, y interface{}) bool {
	vx, vy := reflect.ValueOf(x), reflect.ValueOf(y)
	return (x != nil && y != nil && vx.Type() == vy.Type()) &&
		(vx.Kind() == reflect.Slice || vx.Kind() == reflect.Map) &&
		(vx.Len() == 0 && vy.Len() == 0)
}

// EquateApprox returns a Comparer option that determines float32 or float64
// values to be equal if they are within a relative fraction or absolute margin.
// This option is not used when either x or y is NaN or infinite.
//
// The fraction determines that the difference of two values must be within the
// smaller fraction of the two values, while the margin determines that the two
// values must be within some absolute margin.
// To express only a fraction or only a margin, use 0 for the other parameter.
// The fraction and margin must be non-negative.
//
// The mathematical expression used is equivalent to:
//	|x-y| ≤ max(fraction*min(|x|, |y|), margin)
//
// EquateApprox can be used in conjunction with EquateNaNs.
func EquateApprox(fraction, margin float64) cmp.Option {
	if margin < 0 || fraction < 0 || math.IsNaN(margin) || math.IsNaN(fraction) {
		panic("margin or fraction must be a non-negative number")
	}
	a := approximator{fraction, margin}
	return cmp.Options{
		cmp.FilterValues(areRealF64s, cmp.Comparer(a.compareF64)),
		cmp.FilterValues(areRealF32s, cmp.Comparer(a.compareF32)),
	}
}

type approximator struct{ frac, marg float64 }

func areRealF64s(x, y float64) bool {
	return !math.IsNaN(x) && !math.IsNaN(y) && !math.IsInf(x, 0) && !math.IsInf(y, 0)
}
func areRealF32s(x, y float32) bool {
	return areRealF64s(float64(x), float64(y))
}
func (a approximator) compareF64(x, y float64) bool {
	relMarg := a.frac * math.Min(math.Abs(x), math.Abs(y))
	return math.Abs(x-y) <= math.Max(a.marg, relMarg)
}
func (a approximator) compareF32(x, y float32) bool {
	return a.compareF64(float64(x), float64(y))
}

// EquateNaNs returns a Comparer option that determines float32 and float64
// NaN values to be equal.
//
// EquateNaNs can be used in conjunction with EquateApprox.
func EquateNaNs() cmp.Option {
	return cmp.Options{
		cmp.FilterValues(areNaNsF64s, cmp.Comparer(equateAlways)),
		cmp.FilterValues(areNaNsF32s, cmp.Comparer(equateAlways)),
	}
}

func areNaNsF64s(x, y float64) bool {
	return math.IsNaN(x) && math.IsNaN(y)
}
func areNaNsF32s(x, y float32) bool {
	return areNaNsF64s(float64(x), float64(y))
}
