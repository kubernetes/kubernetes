// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cmpopts provides common options for the cmp package.
package cmpopts

import (
	"errors"
	"math"
	"reflect"
	"time"

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
//
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

// EquateApproxTime returns a Comparer option that determines two non-zero
// time.Time values to be equal if they are within some margin of one another.
// If both times have a monotonic clock reading, then the monotonic time
// difference will be used. The margin must be non-negative.
func EquateApproxTime(margin time.Duration) cmp.Option {
	if margin < 0 {
		panic("margin must be a non-negative number")
	}
	a := timeApproximator{margin}
	return cmp.FilterValues(areNonZeroTimes, cmp.Comparer(a.compare))
}

func areNonZeroTimes(x, y time.Time) bool {
	return !x.IsZero() && !y.IsZero()
}

type timeApproximator struct {
	margin time.Duration
}

func (a timeApproximator) compare(x, y time.Time) bool {
	// Avoid subtracting times to avoid overflow when the
	// difference is larger than the largest representable duration.
	if x.After(y) {
		// Ensure x is always before y
		x, y = y, x
	}
	// We're within the margin if x+margin >= y.
	// Note: time.Time doesn't have AfterOrEqual method hence the negation.
	return !x.Add(a.margin).Before(y)
}

// AnyError is an error that matches any non-nil error.
var AnyError anyError

type anyError struct{}

func (anyError) Error() string     { return "any error" }
func (anyError) Is(err error) bool { return err != nil }

// EquateErrors returns a Comparer option that determines errors to be equal
// if errors.Is reports them to match. The AnyError error can be used to
// match any non-nil error.
func EquateErrors() cmp.Option {
	return cmp.FilterValues(areConcreteErrors, cmp.Comparer(compareErrors))
}

// areConcreteErrors reports whether x and y are types that implement error.
// The input types are deliberately of the interface{} type rather than the
// error type so that we can handle situations where the current type is an
// interface{}, but the underlying concrete types both happen to implement
// the error interface.
func areConcreteErrors(x, y interface{}) bool {
	_, ok1 := x.(error)
	_, ok2 := y.(error)
	return ok1 && ok2
}

func compareErrors(x, y interface{}) bool {
	xe := x.(error)
	ye := y.(error)
	return errors.Is(xe, ye) || errors.Is(ye, xe)
}
