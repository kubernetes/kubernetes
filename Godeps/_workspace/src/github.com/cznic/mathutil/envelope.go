// Copyright (c) 2014 The mathutil Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mathutil

import (
	"math"
)

// Approximation type determines approximation methods used by e.g. Envelope.
type Approximation int

// Specific approximation method tags
const (
	_          Approximation = iota
	Linear                   // As named
	Sinusoidal               // Smooth for all derivations
)

// Envelope is an utility for defining simple curves using a small (usually)
// set of data points.  Envelope returns a value defined by x, points and
// approximation.  The value of x must be in [0,1) otherwise the result is
// undefined or the function may panic. Points are interpreted as dividing the
// [0,1) interval in len(points)-1 sections, so len(points) must be > 1 or the
// function may panic. According to the left and right points closing/adjacent
// to the section the resulting value is interpolated using the chosen
// approximation method.  Unsupported values of approximation are silently
// interpreted as 'Linear'.
func Envelope(x float64, points []float64, approximation Approximation) float64 {
	step := 1 / float64(len(points)-1)
	fslot := math.Floor(x / step)
	mod := x - fslot*step
	slot := int(fslot)
	l, r := points[slot], points[slot+1]
	rmod := mod / step
	switch approximation {
	case Sinusoidal:
		k := (math.Sin(math.Pi*(rmod-0.5)) + 1) / 2
		return l + (r-l)*k
	case Linear:
		fallthrough
	default:
		return l + (r-l)*rmod
	}
}
