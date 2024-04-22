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
func Math() cel.EnvOption {
	return cel.Lib(mathLib{})
}

const (
	mathNamespace = "math"
	leastMacro    = "least"
	greatestMacro = "greatest"
	minFunc       = "math.@min"
	maxFunc       = "math.@max"
)

type mathLib struct{}

// LibraryName implements the SingletonLibrary interface method.
func (mathLib) LibraryName() string {
	return "cel.lib.ext.math"
}

// CompileOptions implements the Library interface method.
func (mathLib) CompileOptions() []cel.EnvOption {
	return []cel.EnvOption{
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
}

// ProgramOptions implements the Library interface method.
func (mathLib) ProgramOptions() []cel.ProgramOption {
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
		if isListLiteralWithValidArgs(args[0]) || isValidArgType(args[0]) {
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
		if isListLiteralWithValidArgs(args[0]) || isValidArgType(args[0]) {
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
	if !isValidArgType(arg) {
		return fmt.Errorf("%s simple literal arguments must be numeric", funcName)
	}
	return nil
}

func isValidArgType(arg ast.Expr) bool {
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

func isListLiteralWithValidArgs(arg ast.Expr) bool {
	switch arg.Kind() {
	case ast.ListKind:
		list := arg.AsList()
		if list.Size() == 0 {
			return false
		}
		for _, e := range list.Elements() {
			if !isValidArgType(e) {
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
