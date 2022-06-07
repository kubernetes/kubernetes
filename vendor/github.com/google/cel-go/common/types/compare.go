// Copyright 2021 Google LLC
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

package types

import (
	"math"

	"github.com/google/cel-go/common/types/ref"
)

func compareDoubleInt(d Double, i Int) Int {
	if d < math.MinInt64 {
		return IntNegOne
	}
	if d > math.MaxInt64 {
		return IntOne
	}
	return compareDouble(d, Double(i))
}

func compareIntDouble(i Int, d Double) Int {
	return -compareDoubleInt(d, i)
}

func compareDoubleUint(d Double, u Uint) Int {
	if d < 0 {
		return IntNegOne
	}
	if d > math.MaxUint64 {
		return IntOne
	}
	return compareDouble(d, Double(u))
}

func compareUintDouble(u Uint, d Double) Int {
	return -compareDoubleUint(d, u)
}

func compareIntUint(i Int, u Uint) Int {
	if i < 0 || u > math.MaxInt64 {
		return IntNegOne
	}
	cmp := i - Int(u)
	if cmp < 0 {
		return IntNegOne
	}
	if cmp > 0 {
		return IntOne
	}
	return IntZero
}

func compareUintInt(u Uint, i Int) Int {
	return -compareIntUint(i, u)
}

func compareDouble(a, b Double) Int {
	if a < b {
		return IntNegOne
	}
	if a > b {
		return IntOne
	}
	return IntZero
}

func compareInt(a, b Int) ref.Val {
	if a < b {
		return IntNegOne
	}
	if a > b {
		return IntOne
	}
	return IntZero
}

func compareUint(a, b Uint) ref.Val {
	if a < b {
		return IntNegOne
	}
	if a > b {
		return IntOne
	}
	return IntZero
}
