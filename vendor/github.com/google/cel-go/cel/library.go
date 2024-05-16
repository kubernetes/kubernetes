// Copyright 2020 Google LLC
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

package cel

import (
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/stdlib"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/interpreter"
	"github.com/google/cel-go/parser"
)

const (
	optMapMacro                = "optMap"
	optFlatMapMacro            = "optFlatMap"
	hasValueFunc               = "hasValue"
	optionalNoneFunc           = "optional.none"
	optionalOfFunc             = "optional.of"
	optionalOfNonZeroValueFunc = "optional.ofNonZeroValue"
	valueFunc                  = "value"
	unusedIterVar              = "#unused"
)

// Library provides a collection of EnvOption and ProgramOption values used to configure a CEL
// environment for a particular use case or with a related set of functionality.
//
// Note, the ProgramOption values provided by a library are expected to be static and not vary
// between calls to Env.Program(). If there is a need for such dynamic configuration, prefer to
// configure these options outside the Library and within the Env.Program() call directly.
type Library interface {
	// CompileOptions returns a collection of functional options for configuring the Parse / Check
	// environment.
	CompileOptions() []EnvOption

	// ProgramOptions returns a collection of functional options which should be included in every
	// Program generated from the Env.Program() call.
	ProgramOptions() []ProgramOption
}

// SingletonLibrary refines the Library interface to ensure that libraries in this format are only
// configured once within the environment.
type SingletonLibrary interface {
	Library

	// LibraryName provides a namespaced name which is used to check whether the library has already
	// been configured in the environment.
	LibraryName() string
}

// Lib creates an EnvOption out of a Library, allowing libraries to be provided as functional args,
// and to be linked to each other.
func Lib(l Library) EnvOption {
	singleton, isSingleton := l.(SingletonLibrary)
	return func(e *Env) (*Env, error) {
		if isSingleton {
			if e.HasLibrary(singleton.LibraryName()) {
				return e, nil
			}
			e.libraries[singleton.LibraryName()] = true
		}
		var err error
		for _, opt := range l.CompileOptions() {
			e, err = opt(e)
			if err != nil {
				return nil, err
			}
		}
		e.progOpts = append(e.progOpts, l.ProgramOptions()...)
		return e, nil
	}
}

// StdLib returns an EnvOption for the standard library of CEL functions and macros.
func StdLib() EnvOption {
	return Lib(stdLibrary{})
}

// stdLibrary implements the Library interface and provides functional options for the core CEL
// features documented in the specification.
type stdLibrary struct{}

// LibraryName implements the SingletonLibrary interface method.
func (stdLibrary) LibraryName() string {
	return "cel.lib.std"
}

// CompileOptions returns options for the standard CEL function declarations and macros.
func (stdLibrary) CompileOptions() []EnvOption {
	return []EnvOption{
		func(e *Env) (*Env, error) {
			var err error
			for _, fn := range stdlib.Functions() {
				existing, found := e.functions[fn.Name()]
				if found {
					fn, err = existing.Merge(fn)
					if err != nil {
						return nil, err
					}
				}
				e.functions[fn.Name()] = fn
			}
			return e, nil
		},
		func(e *Env) (*Env, error) {
			e.variables = append(e.variables, stdlib.Types()...)
			return e, nil
		},
		Macros(StandardMacros...),
	}
}

// ProgramOptions returns function implementations for the standard CEL functions.
func (stdLibrary) ProgramOptions() []ProgramOption {
	return []ProgramOption{}
}

// OptionalTypes enable support for optional syntax and types in CEL.
//
// The optional value type makes it possible to express whether variables have
// been provided, whether a result has been computed, and in the future whether
// an object field path, map key value, or list index has a value.
//
// # Syntax Changes
//
// OptionalTypes are unlike other CEL extensions because they modify the CEL
// syntax itself, notably through the use of a `?` preceding a field name or
// index value.
//
// ## Field Selection
//
// The optional syntax in field selection is denoted as `obj.?field`. In other
// words, if a field is set, return `optional.of(obj.field)“, else
// `optional.none()`. The optional field selection is viral in the sense that
// after the first optional selection all subsequent selections or indices
// are treated as optional, i.e. the following expressions are equivalent:
//
//	obj.?field.subfield
//	obj.?field.?subfield
//
// ## Indexing
//
// Similar to field selection, the optional syntax can be used in index
// expressions on maps and lists:
//
//	list[?0]
//	map[?key]
//
// ## Optional Field Setting
//
// When creating map or message literals, if a field may be optionally set
// based on its presence, then placing a `?` before the field name or key
// will ensure the type on the right-hand side must be optional(T) where T
// is the type of the field or key-value.
//
// The following returns a map with the key expression set only if the
// subfield is present, otherwise an empty map is created:
//
//	{?key: obj.?field.subfield}
//
// ## Optional Element Setting
//
// When creating list literals, an element in the list may be optionally added
// when the element expression is preceded by a `?`:
//
//	[a, ?b, ?c] // return a list with either [a], [a, b], [a, b, c], or [a, c]
//
// # Optional.Of
//
// Create an optional(T) value of a given value with type T.
//
//	optional.of(10)
//
// # Optional.OfNonZeroValue
//
// Create an optional(T) value of a given value with type T if it is not a
// zero-value. A zero-value the default empty value for any given CEL type,
// including empty protobuf message types. If the value is empty, the result
// of this call will be optional.none().
//
//	optional.ofNonZeroValue([1, 2, 3]) // optional(list(int))
//	optional.ofNonZeroValue([]) // optional.none()
//	optional.ofNonZeroValue(0)  // optional.none()
//	optional.ofNonZeroValue("") // optional.none()
//
// # Optional.None
//
// Create an empty optional value.
//
// # HasValue
//
// Determine whether the optional contains a value.
//
//	optional.of(b'hello').hasValue() // true
//	optional.ofNonZeroValue({}).hasValue() // false
//
// # Value
//
// Get the value contained by the optional. If the optional does not have a
// value, the result will be a CEL error.
//
//	optional.of(b'hello').value() // b'hello'
//	optional.ofNonZeroValue({}).value() // error
//
// # Or
//
// If the value on the left-hand side is optional.none(), the optional value
// on the right hand side is returned. If the value on the left-hand set is
// valued, then it is returned. This operation is short-circuiting and will
// only evaluate as many links in the `or` chain as are needed to return a
// non-empty optional value.
//
//	obj.?field.or(m[?key])
//	l[?index].or(obj.?field.subfield).or(obj.?other)
//
// # OrValue
//
// Either return the value contained within the optional on the left-hand side
// or return the alternative value on the right hand side.
//
//	m[?key].orValue("none")
//
// # OptMap
//
// Apply a transformation to the optional's underlying value if it is not empty
// and return an optional typed result based on the transformation. The
// transformation expression type must return a type T which is wrapped into
// an optional.
//
//	msg.?elements.optMap(e, e.size()).orValue(0)
//
// # OptFlatMap
//
// Introduced in version: 1
//
// Apply a transformation to the optional's underlying value if it is not empty
// and return the result. The transform expression must return an optional(T)
// rather than type T. This can be useful when dealing with zero values and
// conditionally generating an empty or non-empty result in ways which cannot
// be expressed with `optMap`.
//
//	msg.?elements.optFlatMap(e, e[?0]) // return the first element if present.
func OptionalTypes(opts ...OptionalTypesOption) EnvOption {
	lib := &optionalLib{version: math.MaxUint32}
	for _, opt := range opts {
		lib = opt(lib)
	}
	return Lib(lib)
}

type optionalLib struct {
	version uint32
}

// OptionalTypesOption is a functional interface for configuring the strings library.
type OptionalTypesOption func(*optionalLib) *optionalLib

// OptionalTypesVersion configures the version of the optional type library.
//
// The version limits which functions are available. Only functions introduced
// below or equal to the given version included in the library. If this option
// is not set, all functions are available.
//
// See the library documentation to determine which version a function was introduced.
// If the documentation does not state which version a function was introduced, it can
// be assumed to be introduced at version 0, when the library was first created.
func OptionalTypesVersion(version uint32) OptionalTypesOption {
	return func(lib *optionalLib) *optionalLib {
		lib.version = version
		return lib
	}
}

// LibraryName implements the SingletonLibrary interface method.
func (lib *optionalLib) LibraryName() string {
	return "cel.lib.optional"
}

// CompileOptions implements the Library interface method.
func (lib *optionalLib) CompileOptions() []EnvOption {
	paramTypeK := TypeParamType("K")
	paramTypeV := TypeParamType("V")
	optionalTypeV := OptionalType(paramTypeV)
	listTypeV := ListType(paramTypeV)
	mapTypeKV := MapType(paramTypeK, paramTypeV)

	opts := []EnvOption{
		// Enable the optional syntax in the parser.
		enableOptionalSyntax(),

		// Introduce the optional type.
		Types(types.OptionalType),

		// Configure the optMap and optFlatMap macros.
		Macros(ReceiverMacro(optMapMacro, 2, optMap)),

		// Global and member functions for working with optional values.
		Function(optionalOfFunc,
			Overload("optional_of", []*Type{paramTypeV}, optionalTypeV,
				UnaryBinding(func(value ref.Val) ref.Val {
					return types.OptionalOf(value)
				}))),
		Function(optionalOfNonZeroValueFunc,
			Overload("optional_ofNonZeroValue", []*Type{paramTypeV}, optionalTypeV,
				UnaryBinding(func(value ref.Val) ref.Val {
					v, isZeroer := value.(traits.Zeroer)
					if !isZeroer || !v.IsZeroValue() {
						return types.OptionalOf(value)
					}
					return types.OptionalNone
				}))),
		Function(optionalNoneFunc,
			Overload("optional_none", []*Type{}, optionalTypeV,
				FunctionBinding(func(values ...ref.Val) ref.Val {
					return types.OptionalNone
				}))),
		Function(valueFunc,
			MemberOverload("optional_value", []*Type{optionalTypeV}, paramTypeV,
				UnaryBinding(func(value ref.Val) ref.Val {
					opt := value.(*types.Optional)
					return opt.GetValue()
				}))),
		Function(hasValueFunc,
			MemberOverload("optional_hasValue", []*Type{optionalTypeV}, BoolType,
				UnaryBinding(func(value ref.Val) ref.Val {
					opt := value.(*types.Optional)
					return types.Bool(opt.HasValue())
				}))),

		// Implementation of 'or' and 'orValue' are special-cased to support short-circuiting in the
		// evaluation chain.
		Function("or",
			MemberOverload("optional_or_optional", []*Type{optionalTypeV, optionalTypeV}, optionalTypeV)),
		Function("orValue",
			MemberOverload("optional_orValue_value", []*Type{optionalTypeV, paramTypeV}, paramTypeV)),

		// OptSelect is handled specially by the type-checker, so the receiver's field type is used to determine the
		// optput type.
		Function(operators.OptSelect,
			Overload("select_optional_field", []*Type{DynType, StringType}, optionalTypeV)),

		// OptIndex is handled mostly like any other indexing operation on a list or map, so the type-checker can use
		// these signatures to determine type-agreement without any special handling.
		Function(operators.OptIndex,
			Overload("list_optindex_optional_int", []*Type{listTypeV, IntType}, optionalTypeV),
			Overload("optional_list_optindex_optional_int", []*Type{OptionalType(listTypeV), IntType}, optionalTypeV),
			Overload("map_optindex_optional_value", []*Type{mapTypeKV, paramTypeK}, optionalTypeV),
			Overload("optional_map_optindex_optional_value", []*Type{OptionalType(mapTypeKV), paramTypeK}, optionalTypeV)),

		// Index overloads to accommodate using an optional value as the operand.
		Function(operators.Index,
			Overload("optional_list_index_int", []*Type{OptionalType(listTypeV), IntType}, optionalTypeV),
			Overload("optional_map_index_value", []*Type{OptionalType(mapTypeKV), paramTypeK}, optionalTypeV)),
	}
	if lib.version >= 1 {
		opts = append(opts, Macros(ReceiverMacro(optFlatMapMacro, 2, optFlatMap)))
	}
	return opts
}

// ProgramOptions implements the Library interface method.
func (lib *optionalLib) ProgramOptions() []ProgramOption {
	return []ProgramOption{
		CustomDecorator(decorateOptionalOr),
	}
}

func optMap(meh MacroExprFactory, target ast.Expr, args []ast.Expr) (ast.Expr, *Error) {
	varIdent := args[0]
	varName := ""
	switch varIdent.Kind() {
	case ast.IdentKind:
		varName = varIdent.AsIdent()
	default:
		return nil, meh.NewError(varIdent.ID(), "optMap() variable name must be a simple identifier")
	}
	mapExpr := args[1]
	return meh.NewCall(
		operators.Conditional,
		meh.NewMemberCall(hasValueFunc, target),
		meh.NewCall(optionalOfFunc,
			meh.NewComprehension(
				meh.NewList(),
				unusedIterVar,
				varName,
				meh.NewMemberCall(valueFunc, target),
				meh.NewLiteral(types.False),
				meh.NewIdent(varName),
				mapExpr,
			),
		),
		meh.NewCall(optionalNoneFunc),
	), nil
}

func optFlatMap(meh MacroExprFactory, target ast.Expr, args []ast.Expr) (ast.Expr, *Error) {
	varIdent := args[0]
	varName := ""
	switch varIdent.Kind() {
	case ast.IdentKind:
		varName = varIdent.AsIdent()
	default:
		return nil, meh.NewError(varIdent.ID(), "optFlatMap() variable name must be a simple identifier")
	}
	mapExpr := args[1]
	return meh.NewCall(
		operators.Conditional,
		meh.NewMemberCall(hasValueFunc, target),
		meh.NewComprehension(
			meh.NewList(),
			unusedIterVar,
			varName,
			meh.NewMemberCall(valueFunc, target),
			meh.NewLiteral(types.False),
			meh.NewIdent(varName),
			mapExpr,
		),
		meh.NewCall(optionalNoneFunc),
	), nil
}

func enableOptionalSyntax() EnvOption {
	return func(e *Env) (*Env, error) {
		e.prsrOpts = append(e.prsrOpts, parser.EnableOptionalSyntax(true))
		return e, nil
	}
}

func decorateOptionalOr(i interpreter.Interpretable) (interpreter.Interpretable, error) {
	call, ok := i.(interpreter.InterpretableCall)
	if !ok {
		return i, nil
	}
	args := call.Args()
	if len(args) != 2 {
		return i, nil
	}
	switch call.Function() {
	case "or":
		if call.OverloadID() != "" && call.OverloadID() != "optional_or_optional" {
			return i, nil
		}
		return &evalOptionalOr{
			id:  call.ID(),
			lhs: args[0],
			rhs: args[1],
		}, nil
	case "orValue":
		if call.OverloadID() != "" && call.OverloadID() != "optional_orValue_value" {
			return i, nil
		}
		return &evalOptionalOrValue{
			id:  call.ID(),
			lhs: args[0],
			rhs: args[1],
		}, nil
	default:
		return i, nil
	}
}

// evalOptionalOr selects between two optional values, either the first if it has a value, or
// the second optional expression is evaluated and returned.
type evalOptionalOr struct {
	id  int64
	lhs interpreter.Interpretable
	rhs interpreter.Interpretable
}

// ID implements the Interpretable interface method.
func (opt *evalOptionalOr) ID() int64 {
	return opt.id
}

// Eval evaluates the left-hand side optional to determine whether it contains a value, else
// proceeds with the right-hand side evaluation.
func (opt *evalOptionalOr) Eval(ctx interpreter.Activation) ref.Val {
	// short-circuit lhs.
	optLHS := opt.lhs.Eval(ctx)
	optVal, ok := optLHS.(*types.Optional)
	if !ok {
		return optLHS
	}
	if optVal.HasValue() {
		return optVal
	}
	return opt.rhs.Eval(ctx)
}

// evalOptionalOrValue selects between an optional or a concrete value. If the optional has a value,
// its value is returned, otherwise the alternative value expression is evaluated and returned.
type evalOptionalOrValue struct {
	id  int64
	lhs interpreter.Interpretable
	rhs interpreter.Interpretable
}

// ID implements the Interpretable interface method.
func (opt *evalOptionalOrValue) ID() int64 {
	return opt.id
}

// Eval evaluates the left-hand side optional to determine whether it contains a value, else
// proceeds with the right-hand side evaluation.
func (opt *evalOptionalOrValue) Eval(ctx interpreter.Activation) ref.Val {
	// short-circuit lhs.
	optLHS := opt.lhs.Eval(ctx)
	optVal, ok := optLHS.(*types.Optional)
	if !ok {
		return optLHS
	}
	if optVal.HasValue() {
		return optVal.GetValue()
	}
	return opt.rhs.Eval(ctx)
}

type timeUTCLibrary struct{}

func (timeUTCLibrary) CompileOptions() []EnvOption {
	return timeOverloadDeclarations
}

func (timeUTCLibrary) ProgramOptions() []ProgramOption {
	return []ProgramOption{}
}

// Declarations and functions which enable using UTC on time.Time inputs when the timezone is unspecified
// in the CEL expression.
var (
	utcTZ = types.String("UTC")

	timeOverloadDeclarations = []EnvOption{
		Function(overloads.TimeGetHours,
			MemberOverload(overloads.DurationToHours, []*Type{DurationType}, IntType,
				UnaryBinding(types.DurationGetHours))),
		Function(overloads.TimeGetMinutes,
			MemberOverload(overloads.DurationToMinutes, []*Type{DurationType}, IntType,
				UnaryBinding(types.DurationGetMinutes))),
		Function(overloads.TimeGetSeconds,
			MemberOverload(overloads.DurationToSeconds, []*Type{DurationType}, IntType,
				UnaryBinding(types.DurationGetSeconds))),
		Function(overloads.TimeGetMilliseconds,
			MemberOverload(overloads.DurationToMilliseconds, []*Type{DurationType}, IntType,
				UnaryBinding(types.DurationGetMilliseconds))),
		Function(overloads.TimeGetFullYear,
			MemberOverload(overloads.TimestampToYear, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					return timestampGetFullYear(ts, utcTZ)
				}),
			),
			MemberOverload(overloads.TimestampToYearWithTz, []*Type{TimestampType, StringType}, IntType,
				BinaryBinding(timestampGetFullYear),
			),
		),
		Function(overloads.TimeGetMonth,
			MemberOverload(overloads.TimestampToMonth, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					return timestampGetMonth(ts, utcTZ)
				}),
			),
			MemberOverload(overloads.TimestampToMonthWithTz, []*Type{TimestampType, StringType}, IntType,
				BinaryBinding(timestampGetMonth),
			),
		),
		Function(overloads.TimeGetDayOfYear,
			MemberOverload(overloads.TimestampToDayOfYear, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					return timestampGetDayOfYear(ts, utcTZ)
				}),
			),
			MemberOverload(overloads.TimestampToDayOfYearWithTz, []*Type{TimestampType, StringType}, IntType,
				BinaryBinding(func(ts, tz ref.Val) ref.Val {
					return timestampGetDayOfYear(ts, tz)
				}),
			),
		),
		Function(overloads.TimeGetDayOfMonth,
			MemberOverload(overloads.TimestampToDayOfMonthZeroBased, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					return timestampGetDayOfMonthZeroBased(ts, utcTZ)
				}),
			),
			MemberOverload(overloads.TimestampToDayOfMonthZeroBasedWithTz, []*Type{TimestampType, StringType}, IntType,
				BinaryBinding(timestampGetDayOfMonthZeroBased),
			),
		),
		Function(overloads.TimeGetDate,
			MemberOverload(overloads.TimestampToDayOfMonthOneBased, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					return timestampGetDayOfMonthOneBased(ts, utcTZ)
				}),
			),
			MemberOverload(overloads.TimestampToDayOfMonthOneBasedWithTz, []*Type{TimestampType, StringType}, IntType,
				BinaryBinding(timestampGetDayOfMonthOneBased),
			),
		),
		Function(overloads.TimeGetDayOfWeek,
			MemberOverload(overloads.TimestampToDayOfWeek, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					return timestampGetDayOfWeek(ts, utcTZ)
				}),
			),
			MemberOverload(overloads.TimestampToDayOfWeekWithTz, []*Type{TimestampType, StringType}, IntType,
				BinaryBinding(timestampGetDayOfWeek),
			),
		),
		Function(overloads.TimeGetHours,
			MemberOverload(overloads.TimestampToHours, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					return timestampGetHours(ts, utcTZ)
				}),
			),
			MemberOverload(overloads.TimestampToHoursWithTz, []*Type{TimestampType, StringType}, IntType,
				BinaryBinding(timestampGetHours),
			),
		),
		Function(overloads.TimeGetMinutes,
			MemberOverload(overloads.TimestampToMinutes, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					return timestampGetMinutes(ts, utcTZ)
				}),
			),
			MemberOverload(overloads.TimestampToMinutesWithTz, []*Type{TimestampType, StringType}, IntType,
				BinaryBinding(timestampGetMinutes),
			),
		),
		Function(overloads.TimeGetSeconds,
			MemberOverload(overloads.TimestampToSeconds, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					return timestampGetSeconds(ts, utcTZ)
				}),
			),
			MemberOverload(overloads.TimestampToSecondsWithTz, []*Type{TimestampType, StringType}, IntType,
				BinaryBinding(timestampGetSeconds),
			),
		),
		Function(overloads.TimeGetMilliseconds,
			MemberOverload(overloads.TimestampToMilliseconds, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					return timestampGetMilliseconds(ts, utcTZ)
				}),
			),
			MemberOverload(overloads.TimestampToMillisecondsWithTz, []*Type{TimestampType, StringType}, IntType,
				BinaryBinding(timestampGetMilliseconds),
			),
		),
	}
)

func timestampGetFullYear(ts, tz ref.Val) ref.Val {
	t, err := inTimeZone(ts, tz)
	if err != nil {
		return types.NewErr(err.Error())
	}
	return types.Int(t.Year())
}

func timestampGetMonth(ts, tz ref.Val) ref.Val {
	t, err := inTimeZone(ts, tz)
	if err != nil {
		return types.NewErr(err.Error())
	}
	// CEL spec indicates that the month should be 0-based, but the Time value
	// for Month() is 1-based.
	return types.Int(t.Month() - 1)
}

func timestampGetDayOfYear(ts, tz ref.Val) ref.Val {
	t, err := inTimeZone(ts, tz)
	if err != nil {
		return types.NewErr(err.Error())
	}
	return types.Int(t.YearDay() - 1)
}

func timestampGetDayOfMonthZeroBased(ts, tz ref.Val) ref.Val {
	t, err := inTimeZone(ts, tz)
	if err != nil {
		return types.NewErr(err.Error())
	}
	return types.Int(t.Day() - 1)
}

func timestampGetDayOfMonthOneBased(ts, tz ref.Val) ref.Val {
	t, err := inTimeZone(ts, tz)
	if err != nil {
		return types.NewErr(err.Error())
	}
	return types.Int(t.Day())
}

func timestampGetDayOfWeek(ts, tz ref.Val) ref.Val {
	t, err := inTimeZone(ts, tz)
	if err != nil {
		return types.NewErr(err.Error())
	}
	return types.Int(t.Weekday())
}

func timestampGetHours(ts, tz ref.Val) ref.Val {
	t, err := inTimeZone(ts, tz)
	if err != nil {
		return types.NewErr(err.Error())
	}
	return types.Int(t.Hour())
}

func timestampGetMinutes(ts, tz ref.Val) ref.Val {
	t, err := inTimeZone(ts, tz)
	if err != nil {
		return types.NewErr(err.Error())
	}
	return types.Int(t.Minute())
}

func timestampGetSeconds(ts, tz ref.Val) ref.Val {
	t, err := inTimeZone(ts, tz)
	if err != nil {
		return types.NewErr(err.Error())
	}
	return types.Int(t.Second())
}

func timestampGetMilliseconds(ts, tz ref.Val) ref.Val {
	t, err := inTimeZone(ts, tz)
	if err != nil {
		return types.NewErr(err.Error())
	}
	return types.Int(t.Nanosecond() / 1000000)
}

func inTimeZone(ts, tz ref.Val) (time.Time, error) {
	t := ts.(types.Timestamp)
	val := string(tz.(types.String))
	ind := strings.Index(val, ":")
	if ind == -1 {
		loc, err := time.LoadLocation(val)
		if err != nil {
			return time.Time{}, err
		}
		return t.In(loc), nil
	}

	// If the input is not the name of a timezone (for example, 'US/Central'), it should be a numerical offset from UTC
	// in the format ^(+|-)(0[0-9]|1[0-4]):[0-5][0-9]$. The numerical input is parsed in terms of hours and minutes.
	hr, err := strconv.Atoi(string(val[0:ind]))
	if err != nil {
		return time.Time{}, err
	}
	min, err := strconv.Atoi(string(val[ind+1:]))
	if err != nil {
		return time.Time{}, err
	}
	var offset int
	if string(val[0]) == "-" {
		offset = hr*60 - min
	} else {
		offset = hr*60 + min
	}
	secondsEastOfUTC := int((time.Duration(offset) * time.Minute).Seconds())
	timezone := time.FixedZone("", secondsEastOfUTC)
	return t.In(timezone), nil
}
