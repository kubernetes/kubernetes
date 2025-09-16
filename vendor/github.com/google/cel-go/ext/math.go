// Copyright 2022 Google LLC
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

package ext

import (
	"fmt"
	"math"
	"strings"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

// Math returns a cel.EnvOption to configure namespaced math helper macros and
// functions.
//
// Note, all macros use the 'math' namespace; however, at the time of macro
// expansion the namespace looks just like any other identifier. If you are
// currently using a variable named 'math', the macro will likely work just as
// intended; however, there is some chance for collision.
//
// # Math.Greatest
//
// Returns the greatest valued number present in the arguments to the macro.
//
// Greatest is a variable argument count macro which must take at least one
// argument. Simple numeric and list literals are supported as valid argument
// types; however, other literals will be flagged as errors during macro
// expansion. If the argument expression does not resolve to a numeric or
// list(numeric) type during type-checking, or during runtime then an error
// will be produced. If a list argument is empty, this too will produce an
// error.
//
//	math.greatest(<arg>, ...) -> <double|int|uint>
//
// Examples:
//
//	math.greatest(1)      // 1
//	math.greatest(1u, 2u) // 2u
//	math.greatest(-42.0, -21.5, -100.0)   // -21.5
//	math.greatest([-42.0, -21.5, -100.0]) // -21.5
//	math.greatest(numbers) // numbers must be list(numeric)
//
//	math.greatest()         // parse error
//	math.greatest('string') // parse error
//	math.greatest(a, b)     // check-time error if a or b is non-numeric
//	math.greatest(dyn('string')) // runtime error
//
// # Math.Least
//
// Returns the least valued number present in the arguments to the macro.
//
// Least is a variable argument count macro which must take at least one
// argument. Simple numeric and list literals are supported as valid argument
// types; however, other literals will be flagged as errors during macro
// expansion. If the argument expression does not resolve to a numeric or
// list(numeric) type during type-checking, or during runtime then an error
// will be produced. If a list argument is empty, this too will produce an
// error.
//
//	math.least(<arg>, ...) -> <double|int|uint>
//
// Examples:
//
//	math.least(1)      // 1
//	math.least(1u, 2u) // 1u
//	math.least(-42.0, -21.5, -100.0)   // -100.0
//	math.least([-42.0, -21.5, -100.0]) // -100.0
//	math.least(numbers) // numbers must be list(numeric)
//
//	math.least()         // parse error
//	math.least('string') // parse error
//	math.least(a, b)     // check-time error if a or b is non-numeric
//	math.least(dyn('string')) // runtime error
//
// # Math.BitOr
//
// Introduced at version: 1
//
// Performs a bitwise-OR operation over two int or uint values.
//
//	math.bitOr(<int>, <int>) -> <int>
//	math.bitOr(<uint>, <uint>) -> <uint>
//
// Examples:
//
//	math.bitOr(1u, 2u)    // returns 3u
//	math.bitOr(-2, -4)    // returns -2
//
// # Math.BitAnd
//
// Introduced at version: 1
//
// Performs a bitwise-AND operation over two int or uint values.
//
//	math.bitAnd(<int>, <int>) -> <int>
//	math.bitAnd(<uint>, <uint>) -> <uint>
//
// Examples:
//
//	math.bitAnd(3u, 2u)   // return 2u
//	math.bitAnd(3, 5)     // returns 3
//	math.bitAnd(-3, -5)   // returns -7
//
// # Math.BitXor
//
// Introduced at version: 1
//
//	math.bitXor(<int>, <int>) -> <int>
//	math.bitXor(<uint>, <uint>) -> <uint>
//
// Performs a bitwise-XOR operation over two int or uint values.
//
// Examples:
//
//	math.bitXor(3u, 5u) // returns 6u
//	math.bitXor(1, 3)   // returns 2
//
// # Math.BitNot
//
// Introduced at version: 1
//
// Function which accepts a single int or uint and performs a bitwise-NOT
// ones-complement of the given binary value.
//
//	math.bitNot(<int>) -> <int>
//	math.bitNot(<uint>) -> <uint>
//
// Examples
//
//	math.bitNot(1)  // returns -1
//	math.bitNot(-1) // return 0
//	math.bitNot(0u) // returns 18446744073709551615u
//
// # Math.BitShiftLeft
//
// Introduced at version: 1
//
// Perform a left shift of bits on the first parameter, by the amount of bits
// specified in the second parameter. The first parameter is either a uint or
// an int. The second parameter must be an int.
//
// When the second parameter is 64 or greater, 0 will be always be returned
// since the number of bits shifted is greater than or equal to the total bit
// length of the number being shifted. Negative valued bit shifts will result
// in a runtime error.
//
//	math.bitShiftLeft(<int>, <int>) -> <int>
//	math.bitShiftLeft(<uint>, <int>) -> <uint>
//
// Examples
//
//	math.bitShiftLeft(1, 2)    // returns 4
//	math.bitShiftLeft(-1, 2)   // returns -4
//	math.bitShiftLeft(1u, 2)   // return 4u
//	math.bitShiftLeft(1u, 200) // returns 0u
//
// # Math.BitShiftRight
//
// Introduced at version: 1
//
// Perform a right shift of bits on the first parameter, by the amount of bits
// specified in the second parameter. The first parameter is either a uint or
// an int. The second parameter must be an int.
//
// When the second parameter is 64 or greater, 0 will always be returned since
// the number of bits shifted is greater than or equal to the total bit length
// of the number being shifted. Negative valued bit shifts will result in a
// runtime error.
//
// The sign bit extension will not be preserved for this operation: vacant bits
// on the left are filled with 0.
//
//	math.bitShiftRight(<int>, <int>) -> <int>
//	math.bitShiftRight(<uint>, <int>) -> <uint>
//
// Examples
//
//	math.bitShiftRight(1024, 2)    // returns 256
//	math.bitShiftRight(1024u, 2)   // returns 256u
//	math.bitShiftRight(1024u, 64)  // returns 0u
//
// # Math.Ceil
//
// Introduced at version: 1
//
// Compute the ceiling of a double value.
//
//	math.ceil(<double>) -> <double>
//
// Examples:
//
//	math.ceil(1.2)   // returns 2.0
//	math.ceil(-1.2)  // returns -1.0
//
// # Math.Floor
//
// Introduced at version: 1
//
// Compute the floor of a double value.
//
//	math.floor(<double>) -> <double>
//
// Examples:
//
//	math.floor(1.2)   // returns 1.0
//	math.floor(-1.2)  // returns -2.0
//
// # Math.Round
//
// Introduced at version: 1
//
// Rounds the double value to the nearest whole number with ties rounding away
// from zero, e.g. 1.5 -> 2.0, -1.5 -> -2.0.
//
//	math.round(<double>) -> <double>
//
// Examples:
//
//	math.round(1.2)  // returns 1.0
//	math.round(1.5)  // returns 2.0
//	math.round(-1.5) // returns -2.0
//
// # Math.Trunc
//
// Introduced at version: 1
//
// Truncates the fractional portion of the double value.
//
//	math.trunc(<double>) -> <double>
//
// Examples:
//
//	math.trunc(-1.3)  // returns -1.0
//	math.trunc(1.3)   // returns 1.0
//
// # Math.Abs
//
// Introduced at version: 1
//
// Returns the absolute value of the numeric type provided as input. If the
// value is NaN, the output is NaN. If the input is int64 min, the function
// will result in an overflow error.
//
//	math.abs(<double>) -> <double>
//	math.abs(<int>) -> <int>
//	math.abs(<uint>) -> <uint>
//
// Examples:
//
//	math.abs(-1)  // returns 1
//	math.abs(1)   // returns 1
//	math.abs(-9223372036854775808) // overflow error
//
// # Math.Sign
//
// Introduced at version: 1
//
// Returns the sign of the numeric type, either -1, 0, 1 as an int, double, or
// uint depending on the overload. For floating point values, if NaN is
// provided as input, the output is also NaN. The implementation does not
// differentiate between positive and negative zero.
//
//	math.sign(<double>) -> <double>
//	math.sign(<int>) -> <int>
//	math.sign(<uint>) -> <uint>
//
// Examples:
//
//	math.sign(-42) // returns -1
//	math.sign(0)   // returns 0
//	math.sign(42)  // returns 1
//
// # Math.IsInf
//
// Introduced at version: 1
//
// Returns true if the input double value is -Inf or +Inf.
//
//	math.isInf(<double>) -> <bool>
//
// Examples:
//
//	math.isInf(1.0/0.0)  // returns true
//	math.isInf(1.2)      // returns false
//
// # Math.IsNaN
//
// Introduced at version: 1
//
// Returns true if the input double value is NaN, false otherwise.
//
//	math.isNaN(<double>) -> <bool>
//
// Examples:
//
//	math.isNaN(0.0/0.0)  // returns true
//	math.isNaN(1.2)      // returns false
//
// # Math.IsFinite
//
// Introduced at version: 1
//
// Returns true if the value is a finite number. Equivalent in behavior to:
// !math.isNaN(double) && !math.isInf(double)
//
//	math.isFinite(<double>) -> <bool>
//
// Examples:
//
//	math.isFinite(0.0/0.0)  // returns false
//	math.isFinite(1.2)      // returns true
//
// # Math.Sqrt
//
// Introduced at version: 2
//
// Returns the square root of the given input as double
// Throws error for negative or non-numeric inputs
//
//	math.sqrt(<double>) -> <double>
//	math.sqrt(<int>) -> <double>
//	math.sqrt(<uint>) -> <double>
//
// Examples:
//
//	math.sqrt(81) // returns 9.0
//	math.sqrt(985.25)   // returns 31.388692231439016
//      math.sqrt(-15)  // returns NaN
func Math(options ...MathOption) cel.EnvOption {
	m := &mathLib{version: math.MaxUint32}
	for _, o := range options {
		m = o(m)
	}
	return cel.Lib(m)
}

const (
	mathNamespace = "math"
	leastMacro    = "least"
	greatestMacro = "greatest"

	// Min-max functions
	minFunc = "math.@min"
	maxFunc = "math.@max"

	// Rounding functions
	ceilFunc  = "math.ceil"
	floorFunc = "math.floor"
	roundFunc = "math.round"
	truncFunc = "math.trunc"

	// Floating point helper functions
	isInfFunc    = "math.isInf"
	isNanFunc    = "math.isNaN"
	isFiniteFunc = "math.isFinite"

	// Signedness functions
	absFunc  = "math.abs"
	signFunc = "math.sign"

	// SquareRoot function
	sqrtFunc = "math.sqrt"

	// Bitwise functions
	bitAndFunc        = "math.bitAnd"
	bitOrFunc         = "math.bitOr"
	bitXorFunc        = "math.bitXor"
	bitNotFunc        = "math.bitNot"
	bitShiftLeftFunc  = "math.bitShiftLeft"
	bitShiftRightFunc = "math.bitShiftRight"
)

var (
	errIntOverflow = types.NewErr("integer overflow")
)

// MathOption declares a functional operator for configuring math extensions.
type MathOption func(*mathLib) *mathLib

// MathVersion sets the library version for math extensions.
func MathVersion(version uint32) MathOption {
	return func(lib *mathLib) *mathLib {
		lib.version = version
		return lib
	}
}

type mathLib struct {
	version uint32
}

// LibraryName implements the SingletonLibrary interface method.
func (*mathLib) LibraryName() string {
	return "cel.lib.ext.math"
}

// CompileOptions implements the Library interface method.
func (lib *mathLib) CompileOptions() []cel.EnvOption {
	opts := []cel.EnvOption{
		cel.Macros(
			// math.least(num, ...)
			cel.ReceiverVarArgMacro(leastMacro, mathLeast),
			// math.greatest(num, ...)
			cel.ReceiverVarArgMacro(greatestMacro, mathGreatest),
		),
		cel.Function(minFunc,
			cel.Overload("math_@min_double", []*cel.Type{cel.DoubleType}, cel.DoubleType,
				cel.UnaryBinding(identity)),
			cel.Overload("math_@min_int", []*cel.Type{cel.IntType}, cel.IntType,
				cel.UnaryBinding(identity)),
			cel.Overload("math_@min_uint", []*cel.Type{cel.UintType}, cel.UintType,
				cel.UnaryBinding(identity)),
			cel.Overload("math_@min_double_double", []*cel.Type{cel.DoubleType, cel.DoubleType}, cel.DoubleType,
				cel.BinaryBinding(minPair)),
			cel.Overload("math_@min_int_int", []*cel.Type{cel.IntType, cel.IntType}, cel.IntType,
				cel.BinaryBinding(minPair)),
			cel.Overload("math_@min_uint_uint", []*cel.Type{cel.UintType, cel.UintType}, cel.UintType,
				cel.BinaryBinding(minPair)),
			cel.Overload("math_@min_int_uint", []*cel.Type{cel.IntType, cel.UintType}, cel.DynType,
				cel.BinaryBinding(minPair)),
			cel.Overload("math_@min_int_double", []*cel.Type{cel.IntType, cel.DoubleType}, cel.DynType,
				cel.BinaryBinding(minPair)),
			cel.Overload("math_@min_double_int", []*cel.Type{cel.DoubleType, cel.IntType}, cel.DynType,
				cel.BinaryBinding(minPair)),
			cel.Overload("math_@min_double_uint", []*cel.Type{cel.DoubleType, cel.UintType}, cel.DynType,
				cel.BinaryBinding(minPair)),
			cel.Overload("math_@min_uint_int", []*cel.Type{cel.UintType, cel.IntType}, cel.DynType,
				cel.BinaryBinding(minPair)),
			cel.Overload("math_@min_uint_double", []*cel.Type{cel.UintType, cel.DoubleType}, cel.DynType,
				cel.BinaryBinding(minPair)),
			cel.Overload("math_@min_list_double", []*cel.Type{cel.ListType(cel.DoubleType)}, cel.DoubleType,
				cel.UnaryBinding(minList)),
			cel.Overload("math_@min_list_int", []*cel.Type{cel.ListType(cel.IntType)}, cel.IntType,
				cel.UnaryBinding(minList)),
			cel.Overload("math_@min_list_uint", []*cel.Type{cel.ListType(cel.UintType)}, cel.UintType,
				cel.UnaryBinding(minList)),
		),
		cel.Function(maxFunc,
			cel.Overload("math_@max_double", []*cel.Type{cel.DoubleType}, cel.DoubleType,
				cel.UnaryBinding(identity)),
			cel.Overload("math_@max_int", []*cel.Type{cel.IntType}, cel.IntType,
				cel.UnaryBinding(identity)),
			cel.Overload("math_@max_uint", []*cel.Type{cel.UintType}, cel.UintType,
				cel.UnaryBinding(identity)),
			cel.Overload("math_@max_double_double", []*cel.Type{cel.DoubleType, cel.DoubleType}, cel.DoubleType,
				cel.BinaryBinding(maxPair)),
			cel.Overload("math_@max_int_int", []*cel.Type{cel.IntType, cel.IntType}, cel.IntType,
				cel.BinaryBinding(maxPair)),
			cel.Overload("math_@max_uint_uint", []*cel.Type{cel.UintType, cel.UintType}, cel.UintType,
				cel.BinaryBinding(maxPair)),
			cel.Overload("math_@max_int_uint", []*cel.Type{cel.IntType, cel.UintType}, cel.DynType,
				cel.BinaryBinding(maxPair)),
			cel.Overload("math_@max_int_double", []*cel.Type{cel.IntType, cel.DoubleType}, cel.DynType,
				cel.BinaryBinding(maxPair)),
			cel.Overload("math_@max_double_int", []*cel.Type{cel.DoubleType, cel.IntType}, cel.DynType,
				cel.BinaryBinding(maxPair)),
			cel.Overload("math_@max_double_uint", []*cel.Type{cel.DoubleType, cel.UintType}, cel.DynType,
				cel.BinaryBinding(maxPair)),
			cel.Overload("math_@max_uint_int", []*cel.Type{cel.UintType, cel.IntType}, cel.DynType,
				cel.BinaryBinding(maxPair)),
			cel.Overload("math_@max_uint_double", []*cel.Type{cel.UintType, cel.DoubleType}, cel.DynType,
				cel.BinaryBinding(maxPair)),
			cel.Overload("math_@max_list_double", []*cel.Type{cel.ListType(cel.DoubleType)}, cel.DoubleType,
				cel.UnaryBinding(maxList)),
			cel.Overload("math_@max_list_int", []*cel.Type{cel.ListType(cel.IntType)}, cel.IntType,
				cel.UnaryBinding(maxList)),
			cel.Overload("math_@max_list_uint", []*cel.Type{cel.ListType(cel.UintType)}, cel.UintType,
				cel.UnaryBinding(maxList)),
		),
	}
	if lib.version >= 1 {
		opts = append(opts,
			// Rounding function declarations
			cel.Function(ceilFunc,
				cel.Overload("math_ceil_double", []*cel.Type{cel.DoubleType}, cel.DoubleType,
					cel.UnaryBinding(ceil))),
			cel.Function(floorFunc,
				cel.Overload("math_floor_double", []*cel.Type{cel.DoubleType}, cel.DoubleType,
					cel.UnaryBinding(floor))),
			cel.Function(roundFunc,
				cel.Overload("math_round_double", []*cel.Type{cel.DoubleType}, cel.DoubleType,
					cel.UnaryBinding(round))),
			cel.Function(truncFunc,
				cel.Overload("math_trunc_double", []*cel.Type{cel.DoubleType}, cel.DoubleType,
					cel.UnaryBinding(trunc))),

			// Floating point helpers
			cel.Function(isInfFunc,
				cel.Overload("math_isInf_double", []*cel.Type{cel.DoubleType}, cel.BoolType,
					cel.UnaryBinding(isInf))),
			cel.Function(isNanFunc,
				cel.Overload("math_isNaN_double", []*cel.Type{cel.DoubleType}, cel.BoolType,
					cel.UnaryBinding(isNaN))),
			cel.Function(isFiniteFunc,
				cel.Overload("math_isFinite_double", []*cel.Type{cel.DoubleType}, cel.BoolType,
					cel.UnaryBinding(isFinite))),

			// Signedness functions
			cel.Function(absFunc,
				cel.Overload("math_abs_double", []*cel.Type{cel.DoubleType}, cel.DoubleType,
					cel.UnaryBinding(absDouble)),
				cel.Overload("math_abs_int", []*cel.Type{cel.IntType}, cel.IntType,
					cel.UnaryBinding(absInt)),
				cel.Overload("math_abs_uint", []*cel.Type{cel.UintType}, cel.UintType,
					cel.UnaryBinding(identity)),
			),
			cel.Function(signFunc,
				cel.Overload("math_sign_double", []*cel.Type{cel.DoubleType}, cel.DoubleType,
					cel.UnaryBinding(sign)),
				cel.Overload("math_sign_int", []*cel.Type{cel.IntType}, cel.IntType,
					cel.UnaryBinding(sign)),
				cel.Overload("math_sign_uint", []*cel.Type{cel.UintType}, cel.UintType,
					cel.UnaryBinding(sign)),
			),

			// Bitwise operator declarations
			cel.Function(bitAndFunc,
				cel.Overload("math_bitAnd_int_int", []*cel.Type{cel.IntType, cel.IntType}, cel.IntType,
					cel.BinaryBinding(bitAndPairInt)),
				cel.Overload("math_bitAnd_uint_uint", []*cel.Type{cel.UintType, cel.UintType}, cel.UintType,
					cel.BinaryBinding(bitAndPairUint)),
			),
			cel.Function(bitOrFunc,
				cel.Overload("math_bitOr_int_int", []*cel.Type{cel.IntType, cel.IntType}, cel.IntType,
					cel.BinaryBinding(bitOrPairInt)),
				cel.Overload("math_bitOr_uint_uint", []*cel.Type{cel.UintType, cel.UintType}, cel.UintType,
					cel.BinaryBinding(bitOrPairUint)),
			),
			cel.Function(bitXorFunc,
				cel.Overload("math_bitXor_int_int", []*cel.Type{cel.IntType, cel.IntType}, cel.IntType,
					cel.BinaryBinding(bitXorPairInt)),
				cel.Overload("math_bitXor_uint_uint", []*cel.Type{cel.UintType, cel.UintType}, cel.UintType,
					cel.BinaryBinding(bitXorPairUint)),
			),
			cel.Function(bitNotFunc,
				cel.Overload("math_bitNot_int_int", []*cel.Type{cel.IntType}, cel.IntType,
					cel.UnaryBinding(bitNotInt)),
				cel.Overload("math_bitNot_uint_uint", []*cel.Type{cel.UintType}, cel.UintType,
					cel.UnaryBinding(bitNotUint)),
			),
			cel.Function(bitShiftLeftFunc,
				cel.Overload("math_bitShiftLeft_int_int", []*cel.Type{cel.IntType, cel.IntType}, cel.IntType,
					cel.BinaryBinding(bitShiftLeftIntInt)),
				cel.Overload("math_bitShiftLeft_uint_int", []*cel.Type{cel.UintType, cel.IntType}, cel.UintType,
					cel.BinaryBinding(bitShiftLeftUintInt)),
			),
			cel.Function(bitShiftRightFunc,
				cel.Overload("math_bitShiftRight_int_int", []*cel.Type{cel.IntType, cel.IntType}, cel.IntType,
					cel.BinaryBinding(bitShiftRightIntInt)),
				cel.Overload("math_bitShiftRight_uint_int", []*cel.Type{cel.UintType, cel.IntType}, cel.UintType,
					cel.BinaryBinding(bitShiftRightUintInt)),
			),
		)
	}
	if lib.version >= 2 {
		opts = append(opts,
			cel.Function(sqrtFunc,
				cel.Overload("math_sqrt_double", []*cel.Type{cel.DoubleType}, cel.DoubleType,
					cel.UnaryBinding(sqrt)),
				cel.Overload("math_sqrt_int", []*cel.Type{cel.IntType}, cel.DoubleType,
					cel.UnaryBinding(sqrt)),
				cel.Overload("math_sqrt_uint", []*cel.Type{cel.UintType}, cel.DoubleType,
					cel.UnaryBinding(sqrt)),
			),
		)
	}
	return opts
}

// ProgramOptions implements the Library interface method.
func (*mathLib) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

func mathLeast(meh cel.MacroExprFactory, target ast.Expr, args []ast.Expr) (ast.Expr, *cel.Error) {
	if !macroTargetMatchesNamespace(mathNamespace, target) {
		return nil, nil
	}
	switch len(args) {
	case 0:
		return nil, meh.NewError(target.ID(), "math.least() requires at least one argument")
	case 1:
		if isListLiteralWithNumericArgs(args[0]) || isNumericArgType(args[0]) {
			return meh.NewCall(minFunc, args[0]), nil
		}
		return nil, meh.NewError(args[0].ID(), "math.least() invalid single argument value")
	case 2:
		err := checkInvalidArgs(meh, "math.least()", args)
		if err != nil {
			return nil, err
		}
		return meh.NewCall(minFunc, args...), nil
	default:
		err := checkInvalidArgs(meh, "math.least()", args)
		if err != nil {
			return nil, err
		}
		return meh.NewCall(minFunc, meh.NewList(args...)), nil
	}
}

func mathGreatest(mef cel.MacroExprFactory, target ast.Expr, args []ast.Expr) (ast.Expr, *cel.Error) {
	if !macroTargetMatchesNamespace(mathNamespace, target) {
		return nil, nil
	}
	switch len(args) {
	case 0:
		return nil, mef.NewError(target.ID(), "math.greatest() requires at least one argument")
	case 1:
		if isListLiteralWithNumericArgs(args[0]) || isNumericArgType(args[0]) {
			return mef.NewCall(maxFunc, args[0]), nil
		}
		return nil, mef.NewError(args[0].ID(), "math.greatest() invalid single argument value")
	case 2:
		err := checkInvalidArgs(mef, "math.greatest()", args)
		if err != nil {
			return nil, err
		}
		return mef.NewCall(maxFunc, args...), nil
	default:
		err := checkInvalidArgs(mef, "math.greatest()", args)
		if err != nil {
			return nil, err
		}
		return mef.NewCall(maxFunc, mef.NewList(args...)), nil
	}
}

func identity(val ref.Val) ref.Val {
	return val
}

func ceil(val ref.Val) ref.Val {
	v := val.(types.Double)
	return types.Double(math.Ceil(float64(v)))
}

func floor(val ref.Val) ref.Val {
	v := val.(types.Double)
	return types.Double(math.Floor(float64(v)))
}

func round(val ref.Val) ref.Val {
	v := val.(types.Double)
	return types.Double(math.Round(float64(v)))
}

func trunc(val ref.Val) ref.Val {
	v := val.(types.Double)
	return types.Double(math.Trunc(float64(v)))
}

func isInf(val ref.Val) ref.Val {
	v := val.(types.Double)
	return types.Bool(math.IsInf(float64(v), 0))
}

func isFinite(val ref.Val) ref.Val {
	v := float64(val.(types.Double))
	return types.Bool(!math.IsInf(v, 0) && !math.IsNaN(v))
}

func isNaN(val ref.Val) ref.Val {
	v := val.(types.Double)
	return types.Bool(math.IsNaN(float64(v)))
}

func absDouble(val ref.Val) ref.Val {
	v := float64(val.(types.Double))
	return types.Double(math.Abs(v))
}

func absInt(val ref.Val) ref.Val {
	v := int64(val.(types.Int))
	if v == math.MinInt64 {
		return errIntOverflow
	}
	if v >= 0 {
		return val
	}
	return -types.Int(v)
}

func sign(val ref.Val) ref.Val {
	switch v := val.(type) {
	case types.Double:
		if isNaN(v) == types.True {
			return v
		}
		zero := types.Double(0)
		if v > zero {
			return types.Double(1)
		}
		if v < zero {
			return types.Double(-1)
		}
		return zero
	case types.Int:
		return v.Compare(types.IntZero)
	case types.Uint:
		if v == types.Uint(0) {
			return types.Uint(0)
		}
		return types.Uint(1)
	default:
		return maybeSuffixError(val, "math.sign")
	}
}


func sqrt(val ref.Val) ref.Val {
	switch v := val.(type) {
	case types.Double:
	  return types.Double(math.Sqrt(float64(v)))
	case types.Int:
	  return types.Double(math.Sqrt(float64(v)))
	case types.Uint:
	  return types.Double(math.Sqrt(float64(v)))
	default:
	  return types.NewErr("no such overload: sqrt")
	}
}


func bitAndPairInt(first, second ref.Val) ref.Val {
	l := first.(types.Int)
	r := second.(types.Int)
	return l & r
}

func bitAndPairUint(first, second ref.Val) ref.Val {
	l := first.(types.Uint)
	r := second.(types.Uint)
	return l & r
}

func bitOrPairInt(first, second ref.Val) ref.Val {
	l := first.(types.Int)
	r := second.(types.Int)
	return l | r
}

func bitOrPairUint(first, second ref.Val) ref.Val {
	l := first.(types.Uint)
	r := second.(types.Uint)
	return l | r
}

func bitXorPairInt(first, second ref.Val) ref.Val {
	l := first.(types.Int)
	r := second.(types.Int)
	return l ^ r
}

func bitXorPairUint(first, second ref.Val) ref.Val {
	l := first.(types.Uint)
	r := second.(types.Uint)
	return l ^ r
}

func bitNotInt(value ref.Val) ref.Val {
	v := value.(types.Int)
	return ^v
}

func bitNotUint(value ref.Val) ref.Val {
	v := value.(types.Uint)
	return ^v
}

func bitShiftLeftIntInt(value, bits ref.Val) ref.Val {
	v := value.(types.Int)
	bs := bits.(types.Int)
	if bs < types.IntZero {
		return types.NewErr("math.bitShiftLeft() negative offset: %d", bs)
	}
	return v << bs
}

func bitShiftLeftUintInt(value, bits ref.Val) ref.Val {
	v := value.(types.Uint)
	bs := bits.(types.Int)
	if bs < types.IntZero {
		return types.NewErr("math.bitShiftLeft() negative offset: %d", bs)
	}
	return v << bs
}

func bitShiftRightIntInt(value, bits ref.Val) ref.Val {
	v := value.(types.Int)
	bs := bits.(types.Int)
	if bs < types.IntZero {
		return types.NewErr("math.bitShiftRight() negative offset: %d", bs)
	}
	return types.Int(types.Uint(v) >> bs)
}

func bitShiftRightUintInt(value, bits ref.Val) ref.Val {
	v := value.(types.Uint)
	bs := bits.(types.Int)
	if bs < types.IntZero {
		return types.NewErr("math.bitShiftRight() negative offset: %d", bs)
	}
	return v >> bs
}

func minPair(first, second ref.Val) ref.Val {
	cmp, ok := first.(traits.Comparer)
	if !ok {
		return types.MaybeNoSuchOverloadErr(first)
	}
	out := cmp.Compare(second)
	if types.IsUnknownOrError(out) {
		return maybeSuffixError(out, "math.@min")
	}
	if out == types.IntOne {
		return second
	}
	return first
}

func minList(numList ref.Val) ref.Val {
	l := numList.(traits.Lister)
	size := l.Size().(types.Int)
	if size == types.IntZero {
		return types.NewErr("math.@min(list) argument must not be empty")
	}
	min := l.Get(types.IntZero)
	for i := types.IntOne; i < size; i++ {
		min = minPair(min, l.Get(i))
	}
	switch min.Type() {
	case types.IntType, types.DoubleType, types.UintType, types.UnknownType:
		return min
	default:
		return types.NewErr("no such overload: math.@min")
	}
}

func maxPair(first, second ref.Val) ref.Val {
	cmp, ok := first.(traits.Comparer)
	if !ok {
		return types.MaybeNoSuchOverloadErr(first)
	}
	out := cmp.Compare(second)
	if types.IsUnknownOrError(out) {
		return maybeSuffixError(out, "math.@max")
	}
	if out == types.IntNegOne {
		return second
	}
	return first
}

func maxList(numList ref.Val) ref.Val {
	l := numList.(traits.Lister)
	size := l.Size().(types.Int)
	if size == types.IntZero {
		return types.NewErr("math.@max(list) argument must not be empty")
	}
	max := l.Get(types.IntZero)
	for i := types.IntOne; i < size; i++ {
		max = maxPair(max, l.Get(i))
	}
	switch max.Type() {
	case types.IntType, types.DoubleType, types.UintType, types.UnknownType:
		return max
	default:
		return types.NewErr("no such overload: math.@max")
	}
}

func checkInvalidArgs(meh cel.MacroExprFactory, funcName string, args []ast.Expr) *cel.Error {
	for _, arg := range args {
		err := checkInvalidArgLiteral(funcName, arg)
		if err != nil {
			return meh.NewError(arg.ID(), err.Error())
		}
	}
	return nil
}

func checkInvalidArgLiteral(funcName string, arg ast.Expr) error {
	if !isNumericArgType(arg) {
		return fmt.Errorf("%s simple literal arguments must be numeric", funcName)
	}
	return nil
}

func isNumericArgType(arg ast.Expr) bool {
	switch arg.Kind() {
	case ast.LiteralKind:
		c := ref.Val(arg.AsLiteral())
		switch c.(type) {
		case types.Double, types.Int, types.Uint:
			return true
		default:
			return false
		}
	case ast.ListKind, ast.MapKind, ast.StructKind:
		return false
	default:
		return true
	}
}

func isListLiteralWithNumericArgs(arg ast.Expr) bool {
	switch arg.Kind() {
	case ast.ListKind:
		list := arg.AsList()
		if list.Size() == 0 {
			return false
		}
		for _, e := range list.Elements() {
			if !isNumericArgType(e) {
				return false
			}
		}
		return true
	}
	return false
}

func maybeSuffixError(val ref.Val, suffix string) ref.Val {
	if types.IsError(val) {
		msg := val.(*types.Err).String()
		if !strings.Contains(msg, suffix) {
			return types.NewErr("%s: %s", msg, suffix)
		}
	}
	return val
}
