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
	"strconv"
	"strings"
	"time"

	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/interpreter"
	"github.com/google/cel-go/interpreter/functions"
	"github.com/google/cel-go/parser"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

const (
	optMapMacro                = "optMap"
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

// EnvOptions returns options for the standard CEL function declarations and macros.
func (stdLibrary) CompileOptions() []EnvOption {
	return []EnvOption{
		Declarations(checker.StandardDeclarations()...),
		Macros(StandardMacros...),
	}
}

// ProgramOptions returns function implementations for the standard CEL functions.
func (stdLibrary) ProgramOptions() []ProgramOption {
	return []ProgramOption{
		Functions(functions.StandardOverloads()...),
	}
}

type optionalLibrary struct{}

// LibraryName implements the SingletonLibrary interface method.
func (optionalLibrary) LibraryName() string {
	return "cel.lib.optional"
}

// CompileOptions implements the Library interface method.
func (optionalLibrary) CompileOptions() []EnvOption {
	paramTypeK := TypeParamType("K")
	paramTypeV := TypeParamType("V")
	optionalTypeV := OptionalType(paramTypeV)
	listTypeV := ListType(paramTypeV)
	mapTypeKV := MapType(paramTypeK, paramTypeV)

	return []EnvOption{
		// Enable the optional syntax in the parser.
		enableOptionalSyntax(),

		// Introduce the optional type.
		Types(types.OptionalType),

		// Configure the optMap macro.
		Macros(NewReceiverMacro(optMapMacro, 2, optMap)),

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
			Overload("optional_map_index_optional_value", []*Type{OptionalType(mapTypeKV), paramTypeK}, optionalTypeV)),
	}
}

func optMap(meh MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	varIdent := args[0]
	varName := ""
	switch varIdent.GetExprKind().(type) {
	case *exprpb.Expr_IdentExpr:
		varName = varIdent.GetIdentExpr().GetName()
	default:
		return nil, &common.Error{
			Message:  "optMap() variable name must be a simple identifier",
			Location: meh.OffsetLocation(varIdent.GetId()),
		}
	}
	mapExpr := args[1]
	return meh.GlobalCall(
		operators.Conditional,
		meh.ReceiverCall(hasValueFunc, target),
		meh.GlobalCall(optionalOfFunc,
			meh.Fold(
				unusedIterVar,
				meh.NewList(),
				varName,
				meh.ReceiverCall(valueFunc, target),
				meh.LiteralBool(false),
				meh.Ident(varName),
				mapExpr,
			),
		),
		meh.GlobalCall(optionalNoneFunc),
	), nil
}

// ProgramOptions implements the Library interface method.
func (optionalLibrary) ProgramOptions() []ProgramOption {
	return []ProgramOption{
		CustomDecorator(decorateOptionalOr),
	}
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
				UnaryBinding(func(dur ref.Val) ref.Val {
					d := dur.(types.Duration)
					return types.Int(d.Hours())
				}))),
		Function(overloads.TimeGetMinutes,
			MemberOverload(overloads.DurationToMinutes, []*Type{DurationType}, IntType,
				UnaryBinding(func(dur ref.Val) ref.Val {
					d := dur.(types.Duration)
					return types.Int(d.Minutes())
				}))),
		Function(overloads.TimeGetSeconds,
			MemberOverload(overloads.DurationToSeconds, []*Type{DurationType}, IntType,
				UnaryBinding(func(dur ref.Val) ref.Val {
					d := dur.(types.Duration)
					return types.Int(d.Seconds())
				}))),
		Function(overloads.TimeGetMilliseconds,
			MemberOverload(overloads.DurationToMilliseconds, []*Type{DurationType}, IntType,
				UnaryBinding(func(dur ref.Val) ref.Val {
					d := dur.(types.Duration)
					return types.Int(d.Milliseconds())
				}))),
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
