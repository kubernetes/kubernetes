// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package cost provides the saturating arithmetic shared by cost estimation and cost tracking.
//
// Costs and sizes are unsigned 64-bit values where math.MaxUint64 doubles as the representation
// of an unbounded, or unknown, quantity. Every operation in this package saturates at
// math.MaxUint64 rather than wrapping so that an unbounded input remains unbounded through any
// sequence of operations.
package cost

import "math"

// maxUint64AsFloat is the smallest float64 value greater than math.MaxUint64.
//
// Conversion of a float64 to a uint64 is undefined when the value is out of range, so float
// results are compared against this bound before conversion.
var maxUint64AsFloat = math.Ldexp(1.0, 64)

// SafeAdd returns the sum of the input values, saturating at math.MaxUint64.
func SafeAdd(x, y uint64, rest ...uint64) uint64 {
	sum := x
	if y > 0 && sum > math.MaxUint64-y {
		return math.MaxUint64
	}
	sum += y
	for _, r := range rest {
		if r > 0 && sum > math.MaxUint64-r {
			return math.MaxUint64
		}
		sum += r
	}
	return sum
}

// SafeMultiply returns the product of the input values, saturating at math.MaxUint64.
func SafeMultiply(x, y uint64) uint64 {
	if y != 0 && x > math.MaxUint64/y {
		return math.MaxUint64
	}
	return x * y
}

// SafeMultiplyByFactor multiplies a value by a cost factor and returns the nearest integer
// value, rounded up, saturating at math.MaxUint64.
func SafeMultiplyByFactor(x uint64, factor float64) uint64 {
	xFloat := float64(x)
	if xFloat > 0 && factor > 0 && xFloat > math.MaxUint64/factor {
		return math.MaxUint64
	}
	return SafeCeil(xFloat * factor)
}

// SafeCeil returns the smallest integer value greater than or equal to the input, saturating at
// math.MaxUint64 and flooring at zero.
//
// Negative and NaN inputs return zero.
func SafeCeil(x float64) uint64 {
	if math.IsNaN(x) || x <= 0 {
		return 0
	}
	ceil := math.Ceil(x)
	if ceil >= maxUint64AsFloat {
		return math.MaxUint64
	}
	return uint64(ceil)
}
