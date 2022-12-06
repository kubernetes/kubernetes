// Copyright (c) 2015 Björn Rabenstein
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// The code in this package is copy/paste to avoid a dependency. Hence this file
// carries the copyright of the original repo.
// https://github.com/beorn7/floats
package internal

import (
	"math"
)

// minNormalFloat64 is the smallest positive normal value of type float64.
var minNormalFloat64 = math.Float64frombits(0x0010000000000000)

// AlmostEqualFloat64 returns true if a and b are equal within a relative error
// of epsilon. See http://floating-point-gui.de/errors/comparison/ for the
// details of the applied method.
func AlmostEqualFloat64(a, b, epsilon float64) bool {
	if a == b {
		return true
	}
	absA := math.Abs(a)
	absB := math.Abs(b)
	diff := math.Abs(a - b)
	if a == 0 || b == 0 || absA+absB < minNormalFloat64 {
		return diff < epsilon*minNormalFloat64
	}
	return diff/math.Min(absA+absB, math.MaxFloat64) < epsilon
}

// AlmostEqualFloat64s is the slice form of AlmostEqualFloat64.
func AlmostEqualFloat64s(a, b []float64, epsilon float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !AlmostEqualFloat64(a[i], b[i], epsilon) {
			return false
		}
	}
	return true
}
