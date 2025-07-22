// Copyright 2018 Google LLC
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

package traits

import "github.com/google/cel-go/common/types/ref"

// Adder interface to support '+' operator overloads.
type Adder interface {
	// Add returns a combination of the current value and other value.
	//
	// If the other value is an unsupported type, an error is returned.
	Add(other ref.Val) ref.Val
}

// Divider interface to support '/' operator overloads.
type Divider interface {
	// Divide returns the result of dividing the current value by the input
	// denominator.
	//
	// A denominator value of zero results in an error.
	Divide(denominator ref.Val) ref.Val
}

// Modder interface to support '%' operator overloads.
type Modder interface {
	// Modulo returns the result of taking the modulus of the current value
	// by the denominator.
	//
	// A denominator value of zero results in an error.
	Modulo(denominator ref.Val) ref.Val
}

// Multiplier interface to support '*' operator overloads.
type Multiplier interface {
	// Multiply returns the result of multiplying the current and input value.
	Multiply(other ref.Val) ref.Val
}

// Negater interface to support unary '-' and '!' operator overloads.
type Negater interface {
	// Negate returns the complement of the current value.
	Negate() ref.Val
}

// Subtractor interface to support binary '-' operator overloads.
type Subtractor interface {
	// Subtract returns the result of subtracting the input from the current
	// value.
	Subtract(subtrahend ref.Val) ref.Val
}
