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

// Package stdlib contains all of the standard library function declarations and definitions for CEL.
package stdlib

import (
	"github.com/google/cel-go/common/decls"
	"github.com/google/cel-go/common/functions"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

var (
	stdFunctions []*decls.FunctionDecl
	stdTypes     []*decls.VariableDecl
)

func init() {
	paramA := types.NewTypeParamType("A")
	paramB := types.NewTypeParamType("B")
	listOfA := types.NewListType(paramA)
	mapOfAB := types.NewMapType(paramA, paramB)

	stdTypes = []*decls.VariableDecl{
		decls.TypeVariable(types.BoolType),
		decls.TypeVariable(types.BytesType),
		decls.TypeVariable(types.DoubleType),
		decls.TypeVariable(types.DurationType),
		decls.TypeVariable(types.IntType),
		decls.TypeVariable(listOfA),
		decls.TypeVariable(mapOfAB),
		decls.TypeVariable(types.NullType),
		decls.TypeVariable(types.StringType),
		decls.TypeVariable(types.TimestampType),
		decls.TypeVariable(types.TypeType),
		decls.TypeVariable(types.UintType),
	}

	stdFunctions = []*decls.FunctionDecl{
		// Logical operators. Special-cased within the interpreter.
		// Note, the singleton binding prevents extensions from overriding the operator behavior.
		function(operators.Conditional,
			decls.Overload(overloads.Conditional, argTypes(types.BoolType, paramA, paramA), paramA,
				decls.OverloadIsNonStrict()),
			decls.SingletonFunctionBinding(noFunctionOverrides)),
		function(operators.LogicalAnd,
			decls.Overload(overloads.LogicalAnd, argTypes(types.BoolType, types.BoolType), types.BoolType,
				decls.OverloadIsNonStrict()),
			decls.SingletonBinaryBinding(noBinaryOverrides)),
		function(operators.LogicalOr,
			decls.Overload(overloads.LogicalOr, argTypes(types.BoolType, types.BoolType), types.BoolType,
				decls.OverloadIsNonStrict()),
			decls.SingletonBinaryBinding(noBinaryOverrides)),
		function(operators.LogicalNot,
			decls.Overload(overloads.LogicalNot, argTypes(types.BoolType), types.BoolType),
			decls.SingletonUnaryBinding(func(val ref.Val) ref.Val {
				b, ok := val.(types.Bool)
				if !ok {
					return types.MaybeNoSuchOverloadErr(val)
				}
				return b.Negate()
			})),

		// Comprehension short-circuiting related function
		function(operators.NotStrictlyFalse,
			decls.Overload(overloads.NotStrictlyFalse, argTypes(types.BoolType), types.BoolType,
				decls.OverloadIsNonStrict(),
				decls.UnaryBinding(notStrictlyFalse))),
		// Deprecated: __not_strictly_false__
		function(operators.OldNotStrictlyFalse,
			decls.DisableDeclaration(true), // safe deprecation
			decls.Overload(operators.OldNotStrictlyFalse, argTypes(types.BoolType), types.BoolType,
				decls.OverloadIsNonStrict(),
				decls.UnaryBinding(notStrictlyFalse))),

		// Equality / inequality. Special-cased in the interpreter
		function(operators.Equals,
			decls.Overload(overloads.Equals, argTypes(paramA, paramA), types.BoolType),
			decls.SingletonBinaryBinding(noBinaryOverrides)),
		function(operators.NotEquals,
			decls.Overload(overloads.NotEquals, argTypes(paramA, paramA), types.BoolType),
			decls.SingletonBinaryBinding(noBinaryOverrides)),

		// Mathematical operators
		function(operators.Add,
			decls.Overload(overloads.AddBytes,
				argTypes(types.BytesType, types.BytesType), types.BytesType),
			decls.Overload(overloads.AddDouble,
				argTypes(types.DoubleType, types.DoubleType), types.DoubleType),
			decls.Overload(overloads.AddDurationDuration,
				argTypes(types.DurationType, types.DurationType), types.DurationType),
			decls.Overload(overloads.AddDurationTimestamp,
				argTypes(types.DurationType, types.TimestampType), types.TimestampType),
			decls.Overload(overloads.AddTimestampDuration,
				argTypes(types.TimestampType, types.DurationType), types.TimestampType),
			decls.Overload(overloads.AddInt64,
				argTypes(types.IntType, types.IntType), types.IntType),
			decls.Overload(overloads.AddList,
				argTypes(listOfA, listOfA), listOfA),
			decls.Overload(overloads.AddString,
				argTypes(types.StringType, types.StringType), types.StringType),
			decls.Overload(overloads.AddUint64,
				argTypes(types.UintType, types.UintType), types.UintType),
			decls.SingletonBinaryBinding(func(lhs, rhs ref.Val) ref.Val {
				return lhs.(traits.Adder).Add(rhs)
			}, traits.AdderType)),
		function(operators.Divide,
			decls.Overload(overloads.DivideDouble,
				argTypes(types.DoubleType, types.DoubleType), types.DoubleType),
			decls.Overload(overloads.DivideInt64,
				argTypes(types.IntType, types.IntType), types.IntType),
			decls.Overload(overloads.DivideUint64,
				argTypes(types.UintType, types.UintType), types.UintType),
			decls.SingletonBinaryBinding(func(lhs, rhs ref.Val) ref.Val {
				return lhs.(traits.Divider).Divide(rhs)
			}, traits.DividerType)),
		function(operators.Modulo,
			decls.Overload(overloads.ModuloInt64,
				argTypes(types.IntType, types.IntType), types.IntType),
			decls.Overload(overloads.ModuloUint64,
				argTypes(types.UintType, types.UintType), types.UintType),
			decls.SingletonBinaryBinding(func(lhs, rhs ref.Val) ref.Val {
				return lhs.(traits.Modder).Modulo(rhs)
			}, traits.ModderType)),
		function(operators.Multiply,
			decls.Overload(overloads.MultiplyDouble,
				argTypes(types.DoubleType, types.DoubleType), types.DoubleType),
			decls.Overload(overloads.MultiplyInt64,
				argTypes(types.IntType, types.IntType), types.IntType),
			decls.Overload(overloads.MultiplyUint64,
				argTypes(types.UintType, types.UintType), types.UintType),
			decls.SingletonBinaryBinding(func(lhs, rhs ref.Val) ref.Val {
				return lhs.(traits.Multiplier).Multiply(rhs)
			}, traits.MultiplierType)),
		function(operators.Negate,
			decls.Overload(overloads.NegateDouble, argTypes(types.DoubleType), types.DoubleType),
			decls.Overload(overloads.NegateInt64, argTypes(types.IntType), types.IntType),
			decls.SingletonUnaryBinding(func(val ref.Val) ref.Val {
				if types.IsBool(val) {
					return types.MaybeNoSuchOverloadErr(val)
				}
				return val.(traits.Negater).Negate()
			}, traits.NegatorType)),
		function(operators.Subtract,
			decls.Overload(overloads.SubtractDouble,
				argTypes(types.DoubleType, types.DoubleType), types.DoubleType),
			decls.Overload(overloads.SubtractDurationDuration,
				argTypes(types.DurationType, types.DurationType), types.DurationType),
			decls.Overload(overloads.SubtractInt64,
				argTypes(types.IntType, types.IntType), types.IntType),
			decls.Overload(overloads.SubtractTimestampDuration,
				argTypes(types.TimestampType, types.DurationType), types.TimestampType),
			decls.Overload(overloads.SubtractTimestampTimestamp,
				argTypes(types.TimestampType, types.TimestampType), types.DurationType),
			decls.Overload(overloads.SubtractUint64,
				argTypes(types.UintType, types.UintType), types.UintType),
			decls.SingletonBinaryBinding(func(lhs, rhs ref.Val) ref.Val {
				return lhs.(traits.Subtractor).Subtract(rhs)
			}, traits.SubtractorType)),

		// Relations operators

		function(operators.Less,
			decls.Overload(overloads.LessBool,
				argTypes(types.BoolType, types.BoolType), types.BoolType),
			decls.Overload(overloads.LessInt64,
				argTypes(types.IntType, types.IntType), types.BoolType),
			decls.Overload(overloads.LessInt64Double,
				argTypes(types.IntType, types.DoubleType), types.BoolType),
			decls.Overload(overloads.LessInt64Uint64,
				argTypes(types.IntType, types.UintType), types.BoolType),
			decls.Overload(overloads.LessUint64,
				argTypes(types.UintType, types.UintType), types.BoolType),
			decls.Overload(overloads.LessUint64Double,
				argTypes(types.UintType, types.DoubleType), types.BoolType),
			decls.Overload(overloads.LessUint64Int64,
				argTypes(types.UintType, types.IntType), types.BoolType),
			decls.Overload(overloads.LessDouble,
				argTypes(types.DoubleType, types.DoubleType), types.BoolType),
			decls.Overload(overloads.LessDoubleInt64,
				argTypes(types.DoubleType, types.IntType), types.BoolType),
			decls.Overload(overloads.LessDoubleUint64,
				argTypes(types.DoubleType, types.UintType), types.BoolType),
			decls.Overload(overloads.LessString,
				argTypes(types.StringType, types.StringType), types.BoolType),
			decls.Overload(overloads.LessBytes,
				argTypes(types.BytesType, types.BytesType), types.BoolType),
			decls.Overload(overloads.LessTimestamp,
				argTypes(types.TimestampType, types.TimestampType), types.BoolType),
			decls.Overload(overloads.LessDuration,
				argTypes(types.DurationType, types.DurationType), types.BoolType),
			decls.SingletonBinaryBinding(func(lhs, rhs ref.Val) ref.Val {
				cmp := lhs.(traits.Comparer).Compare(rhs)
				if cmp == types.IntNegOne {
					return types.True
				}
				if cmp == types.IntOne || cmp == types.IntZero {
					return types.False
				}
				return cmp
			}, traits.ComparerType)),

		function(operators.LessEquals,
			decls.Overload(overloads.LessEqualsBool,
				argTypes(types.BoolType, types.BoolType), types.BoolType),
			decls.Overload(overloads.LessEqualsInt64,
				argTypes(types.IntType, types.IntType), types.BoolType),
			decls.Overload(overloads.LessEqualsInt64Double,
				argTypes(types.IntType, types.DoubleType), types.BoolType),
			decls.Overload(overloads.LessEqualsInt64Uint64,
				argTypes(types.IntType, types.UintType), types.BoolType),
			decls.Overload(overloads.LessEqualsUint64,
				argTypes(types.UintType, types.UintType), types.BoolType),
			decls.Overload(overloads.LessEqualsUint64Double,
				argTypes(types.UintType, types.DoubleType), types.BoolType),
			decls.Overload(overloads.LessEqualsUint64Int64,
				argTypes(types.UintType, types.IntType), types.BoolType),
			decls.Overload(overloads.LessEqualsDouble,
				argTypes(types.DoubleType, types.DoubleType), types.BoolType),
			decls.Overload(overloads.LessEqualsDoubleInt64,
				argTypes(types.DoubleType, types.IntType), types.BoolType),
			decls.Overload(overloads.LessEqualsDoubleUint64,
				argTypes(types.DoubleType, types.UintType), types.BoolType),
			decls.Overload(overloads.LessEqualsString,
				argTypes(types.StringType, types.StringType), types.BoolType),
			decls.Overload(overloads.LessEqualsBytes,
				argTypes(types.BytesType, types.BytesType), types.BoolType),
			decls.Overload(overloads.LessEqualsTimestamp,
				argTypes(types.TimestampType, types.TimestampType), types.BoolType),
			decls.Overload(overloads.LessEqualsDuration,
				argTypes(types.DurationType, types.DurationType), types.BoolType),
			decls.SingletonBinaryBinding(func(lhs, rhs ref.Val) ref.Val {
				cmp := lhs.(traits.Comparer).Compare(rhs)
				if cmp == types.IntNegOne || cmp == types.IntZero {
					return types.True
				}
				if cmp == types.IntOne {
					return types.False
				}
				return cmp
			}, traits.ComparerType)),

		function(operators.Greater,
			decls.Overload(overloads.GreaterBool,
				argTypes(types.BoolType, types.BoolType), types.BoolType),
			decls.Overload(overloads.GreaterInt64,
				argTypes(types.IntType, types.IntType), types.BoolType),
			decls.Overload(overloads.GreaterInt64Double,
				argTypes(types.IntType, types.DoubleType), types.BoolType),
			decls.Overload(overloads.GreaterInt64Uint64,
				argTypes(types.IntType, types.UintType), types.BoolType),
			decls.Overload(overloads.GreaterUint64,
				argTypes(types.UintType, types.UintType), types.BoolType),
			decls.Overload(overloads.GreaterUint64Double,
				argTypes(types.UintType, types.DoubleType), types.BoolType),
			decls.Overload(overloads.GreaterUint64Int64,
				argTypes(types.UintType, types.IntType), types.BoolType),
			decls.Overload(overloads.GreaterDouble,
				argTypes(types.DoubleType, types.DoubleType), types.BoolType),
			decls.Overload(overloads.GreaterDoubleInt64,
				argTypes(types.DoubleType, types.IntType), types.BoolType),
			decls.Overload(overloads.GreaterDoubleUint64,
				argTypes(types.DoubleType, types.UintType), types.BoolType),
			decls.Overload(overloads.GreaterString,
				argTypes(types.StringType, types.StringType), types.BoolType),
			decls.Overload(overloads.GreaterBytes,
				argTypes(types.BytesType, types.BytesType), types.BoolType),
			decls.Overload(overloads.GreaterTimestamp,
				argTypes(types.TimestampType, types.TimestampType), types.BoolType),
			decls.Overload(overloads.GreaterDuration,
				argTypes(types.DurationType, types.DurationType), types.BoolType),
			decls.SingletonBinaryBinding(func(lhs, rhs ref.Val) ref.Val {
				cmp := lhs.(traits.Comparer).Compare(rhs)
				if cmp == types.IntOne {
					return types.True
				}
				if cmp == types.IntNegOne || cmp == types.IntZero {
					return types.False
				}
				return cmp
			}, traits.ComparerType)),

		function(operators.GreaterEquals,
			decls.Overload(overloads.GreaterEqualsBool,
				argTypes(types.BoolType, types.BoolType), types.BoolType),
			decls.Overload(overloads.GreaterEqualsInt64,
				argTypes(types.IntType, types.IntType), types.BoolType),
			decls.Overload(overloads.GreaterEqualsInt64Double,
				argTypes(types.IntType, types.DoubleType), types.BoolType),
			decls.Overload(overloads.GreaterEqualsInt64Uint64,
				argTypes(types.IntType, types.UintType), types.BoolType),
			decls.Overload(overloads.GreaterEqualsUint64,
				argTypes(types.UintType, types.UintType), types.BoolType),
			decls.Overload(overloads.GreaterEqualsUint64Double,
				argTypes(types.UintType, types.DoubleType), types.BoolType),
			decls.Overload(overloads.GreaterEqualsUint64Int64,
				argTypes(types.UintType, types.IntType), types.BoolType),
			decls.Overload(overloads.GreaterEqualsDouble,
				argTypes(types.DoubleType, types.DoubleType), types.BoolType),
			decls.Overload(overloads.GreaterEqualsDoubleInt64,
				argTypes(types.DoubleType, types.IntType), types.BoolType),
			decls.Overload(overloads.GreaterEqualsDoubleUint64,
				argTypes(types.DoubleType, types.UintType), types.BoolType),
			decls.Overload(overloads.GreaterEqualsString,
				argTypes(types.StringType, types.StringType), types.BoolType),
			decls.Overload(overloads.GreaterEqualsBytes,
				argTypes(types.BytesType, types.BytesType), types.BoolType),
			decls.Overload(overloads.GreaterEqualsTimestamp,
				argTypes(types.TimestampType, types.TimestampType), types.BoolType),
			decls.Overload(overloads.GreaterEqualsDuration,
				argTypes(types.DurationType, types.DurationType), types.BoolType),
			decls.SingletonBinaryBinding(func(lhs, rhs ref.Val) ref.Val {
				cmp := lhs.(traits.Comparer).Compare(rhs)
				if cmp == types.IntOne || cmp == types.IntZero {
					return types.True
				}
				if cmp == types.IntNegOne {
					return types.False
				}
				return cmp
			}, traits.ComparerType)),

		// Indexing
		function(operators.Index,
			decls.Overload(overloads.IndexList, argTypes(listOfA, types.IntType), paramA),
			decls.Overload(overloads.IndexMap, argTypes(mapOfAB, paramA), paramB),
			decls.SingletonBinaryBinding(func(lhs, rhs ref.Val) ref.Val {
				return lhs.(traits.Indexer).Get(rhs)
			}, traits.IndexerType)),

		// Collections operators
		function(operators.In,
			decls.Overload(overloads.InList, argTypes(paramA, listOfA), types.BoolType),
			decls.Overload(overloads.InMap, argTypes(paramA, mapOfAB), types.BoolType),
			decls.SingletonBinaryBinding(inAggregate)),
		function(operators.OldIn,
			decls.DisableDeclaration(true), // safe deprecation
			decls.Overload(overloads.InList, argTypes(paramA, listOfA), types.BoolType),
			decls.Overload(overloads.InMap, argTypes(paramA, mapOfAB), types.BoolType),
			decls.SingletonBinaryBinding(inAggregate)),
		function(overloads.DeprecatedIn,
			decls.DisableDeclaration(true), // safe deprecation
			decls.Overload(overloads.InList, argTypes(paramA, listOfA), types.BoolType),
			decls.Overload(overloads.InMap, argTypes(paramA, mapOfAB), types.BoolType),
			decls.SingletonBinaryBinding(inAggregate)),
		function(overloads.Size,
			decls.Overload(overloads.SizeBytes, argTypes(types.BytesType), types.IntType),
			decls.MemberOverload(overloads.SizeBytesInst, argTypes(types.BytesType), types.IntType),
			decls.Overload(overloads.SizeList, argTypes(listOfA), types.IntType),
			decls.MemberOverload(overloads.SizeListInst, argTypes(listOfA), types.IntType),
			decls.Overload(overloads.SizeMap, argTypes(mapOfAB), types.IntType),
			decls.MemberOverload(overloads.SizeMapInst, argTypes(mapOfAB), types.IntType),
			decls.Overload(overloads.SizeString, argTypes(types.StringType), types.IntType),
			decls.MemberOverload(overloads.SizeStringInst, argTypes(types.StringType), types.IntType),
			decls.SingletonUnaryBinding(func(val ref.Val) ref.Val {
				return val.(traits.Sizer).Size()
			}, traits.SizerType)),

		// Type conversions
		function(overloads.TypeConvertType,
			decls.Overload(overloads.TypeConvertType, argTypes(paramA), types.NewTypeTypeWithParam(paramA)),
			decls.SingletonUnaryBinding(convertToType(types.TypeType))),

		// Bool conversions
		function(overloads.TypeConvertBool,
			decls.Overload(overloads.BoolToBool, argTypes(types.BoolType), types.BoolType,
				decls.UnaryBinding(identity)),
			decls.Overload(overloads.StringToBool, argTypes(types.StringType), types.BoolType,
				decls.UnaryBinding(convertToType(types.BoolType)))),

		// Bytes conversions
		function(overloads.TypeConvertBytes,
			decls.Overload(overloads.BytesToBytes, argTypes(types.BytesType), types.BytesType,
				decls.UnaryBinding(identity)),
			decls.Overload(overloads.StringToBytes, argTypes(types.StringType), types.BytesType,
				decls.UnaryBinding(convertToType(types.BytesType)))),

		// Double conversions
		function(overloads.TypeConvertDouble,
			decls.Overload(overloads.DoubleToDouble, argTypes(types.DoubleType), types.DoubleType,
				decls.UnaryBinding(identity)),
			decls.Overload(overloads.IntToDouble, argTypes(types.IntType), types.DoubleType,
				decls.UnaryBinding(convertToType(types.DoubleType))),
			decls.Overload(overloads.StringToDouble, argTypes(types.StringType), types.DoubleType,
				decls.UnaryBinding(convertToType(types.DoubleType))),
			decls.Overload(overloads.UintToDouble, argTypes(types.UintType), types.DoubleType,
				decls.UnaryBinding(convertToType(types.DoubleType)))),

		// Duration conversions
		function(overloads.TypeConvertDuration,
			decls.Overload(overloads.DurationToDuration, argTypes(types.DurationType), types.DurationType,
				decls.UnaryBinding(identity)),
			decls.Overload(overloads.IntToDuration, argTypes(types.IntType), types.DurationType,
				decls.UnaryBinding(convertToType(types.DurationType))),
			decls.Overload(overloads.StringToDuration, argTypes(types.StringType), types.DurationType,
				decls.UnaryBinding(convertToType(types.DurationType)))),

		// Dyn conversions
		function(overloads.TypeConvertDyn,
			decls.Overload(overloads.ToDyn, argTypes(paramA), types.DynType),
			decls.SingletonUnaryBinding(identity)),

		// Int conversions
		function(overloads.TypeConvertInt,
			decls.Overload(overloads.IntToInt, argTypes(types.IntType), types.IntType,
				decls.UnaryBinding(identity)),
			decls.Overload(overloads.DoubleToInt, argTypes(types.DoubleType), types.IntType,
				decls.UnaryBinding(convertToType(types.IntType))),
			decls.Overload(overloads.DurationToInt, argTypes(types.DurationType), types.IntType,
				decls.UnaryBinding(convertToType(types.IntType))),
			decls.Overload(overloads.StringToInt, argTypes(types.StringType), types.IntType,
				decls.UnaryBinding(convertToType(types.IntType))),
			decls.Overload(overloads.TimestampToInt, argTypes(types.TimestampType), types.IntType,
				decls.UnaryBinding(convertToType(types.IntType))),
			decls.Overload(overloads.UintToInt, argTypes(types.UintType), types.IntType,
				decls.UnaryBinding(convertToType(types.IntType))),
		),

		// String conversions
		function(overloads.TypeConvertString,
			decls.Overload(overloads.StringToString, argTypes(types.StringType), types.StringType,
				decls.UnaryBinding(identity)),
			decls.Overload(overloads.BoolToString, argTypes(types.BoolType), types.StringType,
				decls.UnaryBinding(convertToType(types.StringType))),
			decls.Overload(overloads.BytesToString, argTypes(types.BytesType), types.StringType,
				decls.UnaryBinding(convertToType(types.StringType))),
			decls.Overload(overloads.DoubleToString, argTypes(types.DoubleType), types.StringType,
				decls.UnaryBinding(convertToType(types.StringType))),
			decls.Overload(overloads.DurationToString, argTypes(types.DurationType), types.StringType,
				decls.UnaryBinding(convertToType(types.StringType))),
			decls.Overload(overloads.IntToString, argTypes(types.IntType), types.StringType,
				decls.UnaryBinding(convertToType(types.StringType))),
			decls.Overload(overloads.TimestampToString, argTypes(types.TimestampType), types.StringType,
				decls.UnaryBinding(convertToType(types.StringType))),
			decls.Overload(overloads.UintToString, argTypes(types.UintType), types.StringType,
				decls.UnaryBinding(convertToType(types.StringType)))),

		// Timestamp conversions
		function(overloads.TypeConvertTimestamp,
			decls.Overload(overloads.TimestampToTimestamp, argTypes(types.TimestampType), types.TimestampType,
				decls.UnaryBinding(identity)),
			decls.Overload(overloads.IntToTimestamp, argTypes(types.IntType), types.TimestampType,
				decls.UnaryBinding(convertToType(types.TimestampType))),
			decls.Overload(overloads.StringToTimestamp, argTypes(types.StringType), types.TimestampType,
				decls.UnaryBinding(convertToType(types.TimestampType)))),

		// Uint conversions
		function(overloads.TypeConvertUint,
			decls.Overload(overloads.UintToUint, argTypes(types.UintType), types.UintType,
				decls.UnaryBinding(identity)),
			decls.Overload(overloads.DoubleToUint, argTypes(types.DoubleType), types.UintType,
				decls.UnaryBinding(convertToType(types.UintType))),
			decls.Overload(overloads.IntToUint, argTypes(types.IntType), types.UintType,
				decls.UnaryBinding(convertToType(types.UintType))),
			decls.Overload(overloads.StringToUint, argTypes(types.StringType), types.UintType,
				decls.UnaryBinding(convertToType(types.UintType)))),

		// String functions
		function(overloads.Contains,
			decls.MemberOverload(overloads.ContainsString,
				argTypes(types.StringType, types.StringType), types.BoolType,
				decls.BinaryBinding(types.StringContains)),
			decls.DisableTypeGuards(true)),
		function(overloads.EndsWith,
			decls.MemberOverload(overloads.EndsWithString,
				argTypes(types.StringType, types.StringType), types.BoolType,
				decls.BinaryBinding(types.StringEndsWith)),
			decls.DisableTypeGuards(true)),
		function(overloads.StartsWith,
			decls.MemberOverload(overloads.StartsWithString,
				argTypes(types.StringType, types.StringType), types.BoolType,
				decls.BinaryBinding(types.StringStartsWith)),
			decls.DisableTypeGuards(true)),
		function(overloads.Matches,
			decls.Overload(overloads.Matches, argTypes(types.StringType, types.StringType), types.BoolType),
			decls.MemberOverload(overloads.MatchesString,
				argTypes(types.StringType, types.StringType), types.BoolType),
			decls.SingletonBinaryBinding(func(str, pat ref.Val) ref.Val {
				return str.(traits.Matcher).Match(pat)
			}, traits.MatcherType)),

		// Timestamp / duration functions
		function(overloads.TimeGetFullYear,
			decls.MemberOverload(overloads.TimestampToYear,
				argTypes(types.TimestampType), types.IntType),
			decls.MemberOverload(overloads.TimestampToYearWithTz,
				argTypes(types.TimestampType, types.StringType), types.IntType)),

		function(overloads.TimeGetMonth,
			decls.MemberOverload(overloads.TimestampToMonth,
				argTypes(types.TimestampType), types.IntType),
			decls.MemberOverload(overloads.TimestampToMonthWithTz,
				argTypes(types.TimestampType, types.StringType), types.IntType)),

		function(overloads.TimeGetDayOfYear,
			decls.MemberOverload(overloads.TimestampToDayOfYear,
				argTypes(types.TimestampType), types.IntType),
			decls.MemberOverload(overloads.TimestampToDayOfYearWithTz,
				argTypes(types.TimestampType, types.StringType), types.IntType)),

		function(overloads.TimeGetDayOfMonth,
			decls.MemberOverload(overloads.TimestampToDayOfMonthZeroBased,
				argTypes(types.TimestampType), types.IntType),
			decls.MemberOverload(overloads.TimestampToDayOfMonthZeroBasedWithTz,
				argTypes(types.TimestampType, types.StringType), types.IntType)),

		function(overloads.TimeGetDate,
			decls.MemberOverload(overloads.TimestampToDayOfMonthOneBased,
				argTypes(types.TimestampType), types.IntType),
			decls.MemberOverload(overloads.TimestampToDayOfMonthOneBasedWithTz,
				argTypes(types.TimestampType, types.StringType), types.IntType)),

		function(overloads.TimeGetDayOfWeek,
			decls.MemberOverload(overloads.TimestampToDayOfWeek,
				argTypes(types.TimestampType), types.IntType),
			decls.MemberOverload(overloads.TimestampToDayOfWeekWithTz,
				argTypes(types.TimestampType, types.StringType), types.IntType)),

		function(overloads.TimeGetHours,
			decls.MemberOverload(overloads.TimestampToHours,
				argTypes(types.TimestampType), types.IntType),
			decls.MemberOverload(overloads.TimestampToHoursWithTz,
				argTypes(types.TimestampType, types.StringType), types.IntType),
			decls.MemberOverload(overloads.DurationToHours,
				argTypes(types.DurationType), types.IntType)),

		function(overloads.TimeGetMinutes,
			decls.MemberOverload(overloads.TimestampToMinutes,
				argTypes(types.TimestampType), types.IntType),
			decls.MemberOverload(overloads.TimestampToMinutesWithTz,
				argTypes(types.TimestampType, types.StringType), types.IntType),
			decls.MemberOverload(overloads.DurationToMinutes,
				argTypes(types.DurationType), types.IntType)),

		function(overloads.TimeGetSeconds,
			decls.MemberOverload(overloads.TimestampToSeconds,
				argTypes(types.TimestampType), types.IntType),
			decls.MemberOverload(overloads.TimestampToSecondsWithTz,
				argTypes(types.TimestampType, types.StringType), types.IntType),
			decls.MemberOverload(overloads.DurationToSeconds,
				argTypes(types.DurationType), types.IntType)),

		function(overloads.TimeGetMilliseconds,
			decls.MemberOverload(overloads.TimestampToMilliseconds,
				argTypes(types.TimestampType), types.IntType),
			decls.MemberOverload(overloads.TimestampToMillisecondsWithTz,
				argTypes(types.TimestampType, types.StringType), types.IntType),
			decls.MemberOverload(overloads.DurationToMilliseconds,
				argTypes(types.DurationType), types.IntType)),
	}
}

// Functions returns the set of standard library function declarations and definitions for CEL.
func Functions() []*decls.FunctionDecl {
	return stdFunctions
}

// Types returns the set of standard library types for CEL.
func Types() []*decls.VariableDecl {
	return stdTypes
}

func notStrictlyFalse(value ref.Val) ref.Val {
	if types.IsBool(value) {
		return value
	}
	return types.True
}

func inAggregate(lhs ref.Val, rhs ref.Val) ref.Val {
	if rhs.Type().HasTrait(traits.ContainerType) {
		return rhs.(traits.Container).Contains(lhs)
	}
	return types.ValOrErr(rhs, "no such overload")
}

func function(name string, opts ...decls.FunctionOpt) *decls.FunctionDecl {
	fn, err := decls.NewFunction(name, opts...)
	if err != nil {
		panic(err)
	}
	return fn
}

func argTypes(args ...*types.Type) []*types.Type {
	return args
}

func noBinaryOverrides(rhs, lhs ref.Val) ref.Val {
	return types.NoSuchOverloadErr()
}

func noFunctionOverrides(args ...ref.Val) ref.Val {
	return types.NoSuchOverloadErr()
}

func identity(val ref.Val) ref.Val {
	return val
}

func convertToType(t ref.Type) functions.UnaryOp {
	return func(val ref.Val) ref.Val {
		return val.ConvertToType(t)
	}
}
