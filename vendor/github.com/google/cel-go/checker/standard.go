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

package checker

import (
	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/overloads"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

var (
	standardDeclarations []*exprpb.Decl
)

func init() {
	// Some shortcuts we use when building declarations.
	paramA := decls.NewTypeParamType("A")
	typeParamAList := []string{"A"}
	listOfA := decls.NewListType(paramA)
	paramB := decls.NewTypeParamType("B")
	typeParamABList := []string{"A", "B"}
	mapOfAB := decls.NewMapType(paramA, paramB)

	var idents []*exprpb.Decl
	for _, t := range []*exprpb.Type{
		decls.Int, decls.Uint, decls.Bool,
		decls.Double, decls.Bytes, decls.String} {
		idents = append(idents,
			decls.NewVar(FormatCheckedType(t), decls.NewTypeType(t)))
	}
	idents = append(idents,
		decls.NewVar("list", decls.NewTypeType(listOfA)),
		decls.NewVar("map", decls.NewTypeType(mapOfAB)),
		decls.NewVar("null_type", decls.NewTypeType(decls.Null)),
		decls.NewVar("type", decls.NewTypeType(decls.NewTypeType(nil))))

	standardDeclarations = append(standardDeclarations, idents...)
	standardDeclarations = append(standardDeclarations, []*exprpb.Decl{
		// Booleans
		decls.NewFunction(operators.Conditional,
			decls.NewParameterizedOverload(overloads.Conditional,
				[]*exprpb.Type{decls.Bool, paramA, paramA}, paramA,
				typeParamAList)),

		decls.NewFunction(operators.LogicalAnd,
			decls.NewOverload(overloads.LogicalAnd,
				[]*exprpb.Type{decls.Bool, decls.Bool}, decls.Bool)),

		decls.NewFunction(operators.LogicalOr,
			decls.NewOverload(overloads.LogicalOr,
				[]*exprpb.Type{decls.Bool, decls.Bool}, decls.Bool)),

		decls.NewFunction(operators.LogicalNot,
			decls.NewOverload(overloads.LogicalNot,
				[]*exprpb.Type{decls.Bool}, decls.Bool)),

		decls.NewFunction(operators.NotStrictlyFalse,
			decls.NewOverload(overloads.NotStrictlyFalse,
				[]*exprpb.Type{decls.Bool}, decls.Bool)),

		decls.NewFunction(operators.Equals,
			decls.NewParameterizedOverload(overloads.Equals,
				[]*exprpb.Type{paramA, paramA}, decls.Bool,
				typeParamAList)),

		decls.NewFunction(operators.NotEquals,
			decls.NewParameterizedOverload(overloads.NotEquals,
				[]*exprpb.Type{paramA, paramA}, decls.Bool,
				typeParamAList)),

		// Algebra.

		decls.NewFunction(operators.Subtract,
			decls.NewOverload(overloads.SubtractInt64,
				[]*exprpb.Type{decls.Int, decls.Int}, decls.Int),
			decls.NewOverload(overloads.SubtractUint64,
				[]*exprpb.Type{decls.Uint, decls.Uint}, decls.Uint),
			decls.NewOverload(overloads.SubtractDouble,
				[]*exprpb.Type{decls.Double, decls.Double}, decls.Double),
			decls.NewOverload(overloads.SubtractTimestampTimestamp,
				[]*exprpb.Type{decls.Timestamp, decls.Timestamp}, decls.Duration),
			decls.NewOverload(overloads.SubtractTimestampDuration,
				[]*exprpb.Type{decls.Timestamp, decls.Duration}, decls.Timestamp),
			decls.NewOverload(overloads.SubtractDurationDuration,
				[]*exprpb.Type{decls.Duration, decls.Duration}, decls.Duration)),

		decls.NewFunction(operators.Multiply,
			decls.NewOverload(overloads.MultiplyInt64,
				[]*exprpb.Type{decls.Int, decls.Int}, decls.Int),
			decls.NewOverload(overloads.MultiplyUint64,
				[]*exprpb.Type{decls.Uint, decls.Uint}, decls.Uint),
			decls.NewOverload(overloads.MultiplyDouble,
				[]*exprpb.Type{decls.Double, decls.Double}, decls.Double)),

		decls.NewFunction(operators.Divide,
			decls.NewOverload(overloads.DivideInt64,
				[]*exprpb.Type{decls.Int, decls.Int}, decls.Int),
			decls.NewOverload(overloads.DivideUint64,
				[]*exprpb.Type{decls.Uint, decls.Uint}, decls.Uint),
			decls.NewOverload(overloads.DivideDouble,
				[]*exprpb.Type{decls.Double, decls.Double}, decls.Double)),

		decls.NewFunction(operators.Modulo,
			decls.NewOverload(overloads.ModuloInt64,
				[]*exprpb.Type{decls.Int, decls.Int}, decls.Int),
			decls.NewOverload(overloads.ModuloUint64,
				[]*exprpb.Type{decls.Uint, decls.Uint}, decls.Uint)),

		decls.NewFunction(operators.Add,
			decls.NewOverload(overloads.AddInt64,
				[]*exprpb.Type{decls.Int, decls.Int}, decls.Int),
			decls.NewOverload(overloads.AddUint64,
				[]*exprpb.Type{decls.Uint, decls.Uint}, decls.Uint),
			decls.NewOverload(overloads.AddDouble,
				[]*exprpb.Type{decls.Double, decls.Double}, decls.Double),
			decls.NewOverload(overloads.AddString,
				[]*exprpb.Type{decls.String, decls.String}, decls.String),
			decls.NewOverload(overloads.AddBytes,
				[]*exprpb.Type{decls.Bytes, decls.Bytes}, decls.Bytes),
			decls.NewParameterizedOverload(overloads.AddList,
				[]*exprpb.Type{listOfA, listOfA}, listOfA,
				typeParamAList),
			decls.NewOverload(overloads.AddTimestampDuration,
				[]*exprpb.Type{decls.Timestamp, decls.Duration}, decls.Timestamp),
			decls.NewOverload(overloads.AddDurationTimestamp,
				[]*exprpb.Type{decls.Duration, decls.Timestamp}, decls.Timestamp),
			decls.NewOverload(overloads.AddDurationDuration,
				[]*exprpb.Type{decls.Duration, decls.Duration}, decls.Duration)),

		decls.NewFunction(operators.Negate,
			decls.NewOverload(overloads.NegateInt64,
				[]*exprpb.Type{decls.Int}, decls.Int),
			decls.NewOverload(overloads.NegateDouble,
				[]*exprpb.Type{decls.Double}, decls.Double)),

		// Index.

		decls.NewFunction(operators.Index,
			decls.NewParameterizedOverload(overloads.IndexList,
				[]*exprpb.Type{listOfA, decls.Int}, paramA,
				typeParamAList),
			decls.NewParameterizedOverload(overloads.IndexMap,
				[]*exprpb.Type{mapOfAB, paramA}, paramB,
				typeParamABList)),

		// Collections.

		decls.NewFunction(overloads.Size,
			decls.NewInstanceOverload(overloads.SizeStringInst,
				[]*exprpb.Type{decls.String}, decls.Int),
			decls.NewInstanceOverload(overloads.SizeBytesInst,
				[]*exprpb.Type{decls.Bytes}, decls.Int),
			decls.NewParameterizedInstanceOverload(overloads.SizeListInst,
				[]*exprpb.Type{listOfA}, decls.Int, typeParamAList),
			decls.NewParameterizedInstanceOverload(overloads.SizeMapInst,
				[]*exprpb.Type{mapOfAB}, decls.Int, typeParamABList),
			decls.NewOverload(overloads.SizeString,
				[]*exprpb.Type{decls.String}, decls.Int),
			decls.NewOverload(overloads.SizeBytes,
				[]*exprpb.Type{decls.Bytes}, decls.Int),
			decls.NewParameterizedOverload(overloads.SizeList,
				[]*exprpb.Type{listOfA}, decls.Int, typeParamAList),
			decls.NewParameterizedOverload(overloads.SizeMap,
				[]*exprpb.Type{mapOfAB}, decls.Int, typeParamABList)),

		decls.NewFunction(operators.In,
			decls.NewParameterizedOverload(overloads.InList,
				[]*exprpb.Type{paramA, listOfA}, decls.Bool,
				typeParamAList),
			decls.NewParameterizedOverload(overloads.InMap,
				[]*exprpb.Type{paramA, mapOfAB}, decls.Bool,
				typeParamABList)),

		// Deprecated 'in()' function.

		decls.NewFunction(overloads.DeprecatedIn,
			decls.NewParameterizedOverload(overloads.InList,
				[]*exprpb.Type{paramA, listOfA}, decls.Bool,
				typeParamAList),
			decls.NewParameterizedOverload(overloads.InMap,
				[]*exprpb.Type{paramA, mapOfAB}, decls.Bool,
				typeParamABList)),

		// Conversions to type.

		decls.NewFunction(overloads.TypeConvertType,
			decls.NewParameterizedOverload(overloads.TypeConvertType,
				[]*exprpb.Type{paramA}, decls.NewTypeType(paramA), typeParamAList)),

		// Conversions to int.

		decls.NewFunction(overloads.TypeConvertInt,
			decls.NewOverload(overloads.IntToInt, []*exprpb.Type{decls.Int}, decls.Int),
			decls.NewOverload(overloads.UintToInt, []*exprpb.Type{decls.Uint}, decls.Int),
			decls.NewOverload(overloads.DoubleToInt, []*exprpb.Type{decls.Double}, decls.Int),
			decls.NewOverload(overloads.StringToInt, []*exprpb.Type{decls.String}, decls.Int),
			decls.NewOverload(overloads.TimestampToInt, []*exprpb.Type{decls.Timestamp}, decls.Int),
			decls.NewOverload(overloads.DurationToInt, []*exprpb.Type{decls.Duration}, decls.Int)),

		// Conversions to uint.

		decls.NewFunction(overloads.TypeConvertUint,
			decls.NewOverload(overloads.UintToUint, []*exprpb.Type{decls.Uint}, decls.Uint),
			decls.NewOverload(overloads.IntToUint, []*exprpb.Type{decls.Int}, decls.Uint),
			decls.NewOverload(overloads.DoubleToUint, []*exprpb.Type{decls.Double}, decls.Uint),
			decls.NewOverload(overloads.StringToUint, []*exprpb.Type{decls.String}, decls.Uint)),

		// Conversions to double.

		decls.NewFunction(overloads.TypeConvertDouble,
			decls.NewOverload(overloads.DoubleToDouble, []*exprpb.Type{decls.Double}, decls.Double),
			decls.NewOverload(overloads.IntToDouble, []*exprpb.Type{decls.Int}, decls.Double),
			decls.NewOverload(overloads.UintToDouble, []*exprpb.Type{decls.Uint}, decls.Double),
			decls.NewOverload(overloads.StringToDouble, []*exprpb.Type{decls.String}, decls.Double)),

		// Conversions to bool.

		decls.NewFunction(overloads.TypeConvertBool,
			decls.NewOverload(overloads.BoolToBool, []*exprpb.Type{decls.Bool}, decls.Bool),
			decls.NewOverload(overloads.StringToBool, []*exprpb.Type{decls.String}, decls.Bool)),

		// Conversions to string.

		decls.NewFunction(overloads.TypeConvertString,
			decls.NewOverload(overloads.StringToString, []*exprpb.Type{decls.String}, decls.String),
			decls.NewOverload(overloads.BoolToString, []*exprpb.Type{decls.Bool}, decls.String),
			decls.NewOverload(overloads.IntToString, []*exprpb.Type{decls.Int}, decls.String),
			decls.NewOverload(overloads.UintToString, []*exprpb.Type{decls.Uint}, decls.String),
			decls.NewOverload(overloads.DoubleToString, []*exprpb.Type{decls.Double}, decls.String),
			decls.NewOverload(overloads.BytesToString, []*exprpb.Type{decls.Bytes}, decls.String),
			decls.NewOverload(overloads.TimestampToString, []*exprpb.Type{decls.Timestamp}, decls.String),
			decls.NewOverload(overloads.DurationToString, []*exprpb.Type{decls.Duration}, decls.String)),

		// Conversions to bytes.

		decls.NewFunction(overloads.TypeConvertBytes,
			decls.NewOverload(overloads.BytesToBytes, []*exprpb.Type{decls.Bytes}, decls.Bytes),
			decls.NewOverload(overloads.StringToBytes, []*exprpb.Type{decls.String}, decls.Bytes)),

		// Conversions to timestamps.

		decls.NewFunction(overloads.TypeConvertTimestamp,
			decls.NewOverload(overloads.TimestampToTimestamp,
				[]*exprpb.Type{decls.Timestamp}, decls.Timestamp),
			decls.NewOverload(overloads.StringToTimestamp,
				[]*exprpb.Type{decls.String}, decls.Timestamp),
			decls.NewOverload(overloads.IntToTimestamp,
				[]*exprpb.Type{decls.Int}, decls.Timestamp)),

		// Conversions to durations.

		decls.NewFunction(overloads.TypeConvertDuration,
			decls.NewOverload(overloads.DurationToDuration,
				[]*exprpb.Type{decls.Duration}, decls.Duration),
			decls.NewOverload(overloads.StringToDuration,
				[]*exprpb.Type{decls.String}, decls.Duration),
			decls.NewOverload(overloads.IntToDuration,
				[]*exprpb.Type{decls.Int}, decls.Duration)),

		// Conversions to Dyn.

		decls.NewFunction(overloads.TypeConvertDyn,
			decls.NewParameterizedOverload(overloads.ToDyn,
				[]*exprpb.Type{paramA}, decls.Dyn,
				typeParamAList)),

		// String functions.

		decls.NewFunction(overloads.Contains,
			decls.NewInstanceOverload(overloads.ContainsString,
				[]*exprpb.Type{decls.String, decls.String}, decls.Bool)),
		decls.NewFunction(overloads.EndsWith,
			decls.NewInstanceOverload(overloads.EndsWithString,
				[]*exprpb.Type{decls.String, decls.String}, decls.Bool)),
		decls.NewFunction(overloads.Matches,
			decls.NewInstanceOverload(overloads.MatchesString,
				[]*exprpb.Type{decls.String, decls.String}, decls.Bool)),
		decls.NewFunction(overloads.StartsWith,
			decls.NewInstanceOverload(overloads.StartsWithString,
				[]*exprpb.Type{decls.String, decls.String}, decls.Bool)),

		// Date/time functions.

		decls.NewFunction(overloads.TimeGetFullYear,
			decls.NewInstanceOverload(overloads.TimestampToYear,
				[]*exprpb.Type{decls.Timestamp}, decls.Int),
			decls.NewInstanceOverload(overloads.TimestampToYearWithTz,
				[]*exprpb.Type{decls.Timestamp, decls.String}, decls.Int)),

		decls.NewFunction(overloads.TimeGetMonth,
			decls.NewInstanceOverload(overloads.TimestampToMonth,
				[]*exprpb.Type{decls.Timestamp}, decls.Int),
			decls.NewInstanceOverload(overloads.TimestampToMonthWithTz,
				[]*exprpb.Type{decls.Timestamp, decls.String}, decls.Int)),

		decls.NewFunction(overloads.TimeGetDayOfYear,
			decls.NewInstanceOverload(overloads.TimestampToDayOfYear,
				[]*exprpb.Type{decls.Timestamp}, decls.Int),
			decls.NewInstanceOverload(overloads.TimestampToDayOfYearWithTz,
				[]*exprpb.Type{decls.Timestamp, decls.String}, decls.Int)),

		decls.NewFunction(overloads.TimeGetDayOfMonth,
			decls.NewInstanceOverload(overloads.TimestampToDayOfMonthZeroBased,
				[]*exprpb.Type{decls.Timestamp}, decls.Int),
			decls.NewInstanceOverload(overloads.TimestampToDayOfMonthZeroBasedWithTz,
				[]*exprpb.Type{decls.Timestamp, decls.String}, decls.Int)),

		decls.NewFunction(overloads.TimeGetDate,
			decls.NewInstanceOverload(overloads.TimestampToDayOfMonthOneBased,
				[]*exprpb.Type{decls.Timestamp}, decls.Int),
			decls.NewInstanceOverload(overloads.TimestampToDayOfMonthOneBasedWithTz,
				[]*exprpb.Type{decls.Timestamp, decls.String}, decls.Int)),

		decls.NewFunction(overloads.TimeGetDayOfWeek,
			decls.NewInstanceOverload(overloads.TimestampToDayOfWeek,
				[]*exprpb.Type{decls.Timestamp}, decls.Int),
			decls.NewInstanceOverload(overloads.TimestampToDayOfWeekWithTz,
				[]*exprpb.Type{decls.Timestamp, decls.String}, decls.Int)),

		decls.NewFunction(overloads.TimeGetHours,
			decls.NewInstanceOverload(overloads.TimestampToHours,
				[]*exprpb.Type{decls.Timestamp}, decls.Int),
			decls.NewInstanceOverload(overloads.TimestampToHoursWithTz,
				[]*exprpb.Type{decls.Timestamp, decls.String}, decls.Int),
			decls.NewInstanceOverload(overloads.DurationToHours,
				[]*exprpb.Type{decls.Duration}, decls.Int)),

		decls.NewFunction(overloads.TimeGetMinutes,
			decls.NewInstanceOverload(overloads.TimestampToMinutes,
				[]*exprpb.Type{decls.Timestamp}, decls.Int),
			decls.NewInstanceOverload(overloads.TimestampToMinutesWithTz,
				[]*exprpb.Type{decls.Timestamp, decls.String}, decls.Int),
			decls.NewInstanceOverload(overloads.DurationToMinutes,
				[]*exprpb.Type{decls.Duration}, decls.Int)),

		decls.NewFunction(overloads.TimeGetSeconds,
			decls.NewInstanceOverload(overloads.TimestampToSeconds,
				[]*exprpb.Type{decls.Timestamp}, decls.Int),
			decls.NewInstanceOverload(overloads.TimestampToSecondsWithTz,
				[]*exprpb.Type{decls.Timestamp, decls.String}, decls.Int),
			decls.NewInstanceOverload(overloads.DurationToSeconds,
				[]*exprpb.Type{decls.Duration}, decls.Int)),

		decls.NewFunction(overloads.TimeGetMilliseconds,
			decls.NewInstanceOverload(overloads.TimestampToMilliseconds,
				[]*exprpb.Type{decls.Timestamp}, decls.Int),
			decls.NewInstanceOverload(overloads.TimestampToMillisecondsWithTz,
				[]*exprpb.Type{decls.Timestamp, decls.String}, decls.Int),
			decls.NewInstanceOverload(overloads.DurationToMilliseconds,
				[]*exprpb.Type{decls.Duration}, decls.Int)),

		// Relations.
		decls.NewFunction(operators.Less,
			decls.NewOverload(overloads.LessBool,
				[]*exprpb.Type{decls.Bool, decls.Bool}, decls.Bool),
			decls.NewOverload(overloads.LessInt64,
				[]*exprpb.Type{decls.Int, decls.Int}, decls.Bool),
			decls.NewOverload(overloads.LessInt64Double,
				[]*exprpb.Type{decls.Int, decls.Double}, decls.Bool),
			decls.NewOverload(overloads.LessInt64Uint64,
				[]*exprpb.Type{decls.Int, decls.Uint}, decls.Bool),
			decls.NewOverload(overloads.LessUint64,
				[]*exprpb.Type{decls.Uint, decls.Uint}, decls.Bool),
			decls.NewOverload(overloads.LessUint64Double,
				[]*exprpb.Type{decls.Uint, decls.Double}, decls.Bool),
			decls.NewOverload(overloads.LessUint64Int64,
				[]*exprpb.Type{decls.Uint, decls.Int}, decls.Bool),
			decls.NewOverload(overloads.LessDouble,
				[]*exprpb.Type{decls.Double, decls.Double}, decls.Bool),
			decls.NewOverload(overloads.LessDoubleInt64,
				[]*exprpb.Type{decls.Double, decls.Int}, decls.Bool),
			decls.NewOverload(overloads.LessDoubleUint64,
				[]*exprpb.Type{decls.Double, decls.Uint}, decls.Bool),
			decls.NewOverload(overloads.LessString,
				[]*exprpb.Type{decls.String, decls.String}, decls.Bool),
			decls.NewOverload(overloads.LessBytes,
				[]*exprpb.Type{decls.Bytes, decls.Bytes}, decls.Bool),
			decls.NewOverload(overloads.LessTimestamp,
				[]*exprpb.Type{decls.Timestamp, decls.Timestamp}, decls.Bool),
			decls.NewOverload(overloads.LessDuration,
				[]*exprpb.Type{decls.Duration, decls.Duration}, decls.Bool)),

		decls.NewFunction(operators.LessEquals,
			decls.NewOverload(overloads.LessEqualsBool,
				[]*exprpb.Type{decls.Bool, decls.Bool}, decls.Bool),
			decls.NewOverload(overloads.LessEqualsInt64,
				[]*exprpb.Type{decls.Int, decls.Int}, decls.Bool),
			decls.NewOverload(overloads.LessEqualsInt64Double,
				[]*exprpb.Type{decls.Int, decls.Double}, decls.Bool),
			decls.NewOverload(overloads.LessEqualsInt64Uint64,
				[]*exprpb.Type{decls.Int, decls.Uint}, decls.Bool),
			decls.NewOverload(overloads.LessEqualsUint64,
				[]*exprpb.Type{decls.Uint, decls.Uint}, decls.Bool),
			decls.NewOverload(overloads.LessEqualsUint64Double,
				[]*exprpb.Type{decls.Uint, decls.Double}, decls.Bool),
			decls.NewOverload(overloads.LessEqualsUint64Int64,
				[]*exprpb.Type{decls.Uint, decls.Int}, decls.Bool),
			decls.NewOverload(overloads.LessEqualsDouble,
				[]*exprpb.Type{decls.Double, decls.Double}, decls.Bool),
			decls.NewOverload(overloads.LessEqualsDoubleInt64,
				[]*exprpb.Type{decls.Double, decls.Int}, decls.Bool),
			decls.NewOverload(overloads.LessEqualsDoubleUint64,
				[]*exprpb.Type{decls.Double, decls.Uint}, decls.Bool),
			decls.NewOverload(overloads.LessEqualsString,
				[]*exprpb.Type{decls.String, decls.String}, decls.Bool),
			decls.NewOverload(overloads.LessEqualsBytes,
				[]*exprpb.Type{decls.Bytes, decls.Bytes}, decls.Bool),
			decls.NewOverload(overloads.LessEqualsTimestamp,
				[]*exprpb.Type{decls.Timestamp, decls.Timestamp}, decls.Bool),
			decls.NewOverload(overloads.LessEqualsDuration,
				[]*exprpb.Type{decls.Duration, decls.Duration}, decls.Bool)),

		decls.NewFunction(operators.Greater,
			decls.NewOverload(overloads.GreaterBool,
				[]*exprpb.Type{decls.Bool, decls.Bool}, decls.Bool),
			decls.NewOverload(overloads.GreaterInt64,
				[]*exprpb.Type{decls.Int, decls.Int}, decls.Bool),
			decls.NewOverload(overloads.GreaterInt64Double,
				[]*exprpb.Type{decls.Int, decls.Double}, decls.Bool),
			decls.NewOverload(overloads.GreaterInt64Uint64,
				[]*exprpb.Type{decls.Int, decls.Uint}, decls.Bool),
			decls.NewOverload(overloads.GreaterUint64,
				[]*exprpb.Type{decls.Uint, decls.Uint}, decls.Bool),
			decls.NewOverload(overloads.GreaterUint64Double,
				[]*exprpb.Type{decls.Uint, decls.Double}, decls.Bool),
			decls.NewOverload(overloads.GreaterUint64Int64,
				[]*exprpb.Type{decls.Uint, decls.Int}, decls.Bool),
			decls.NewOverload(overloads.GreaterDouble,
				[]*exprpb.Type{decls.Double, decls.Double}, decls.Bool),
			decls.NewOverload(overloads.GreaterDoubleInt64,
				[]*exprpb.Type{decls.Double, decls.Int}, decls.Bool),
			decls.NewOverload(overloads.GreaterDoubleUint64,
				[]*exprpb.Type{decls.Double, decls.Uint}, decls.Bool),
			decls.NewOverload(overloads.GreaterString,
				[]*exprpb.Type{decls.String, decls.String}, decls.Bool),
			decls.NewOverload(overloads.GreaterBytes,
				[]*exprpb.Type{decls.Bytes, decls.Bytes}, decls.Bool),
			decls.NewOverload(overloads.GreaterTimestamp,
				[]*exprpb.Type{decls.Timestamp, decls.Timestamp}, decls.Bool),
			decls.NewOverload(overloads.GreaterDuration,
				[]*exprpb.Type{decls.Duration, decls.Duration}, decls.Bool)),

		decls.NewFunction(operators.GreaterEquals,
			decls.NewOverload(overloads.GreaterEqualsBool,
				[]*exprpb.Type{decls.Bool, decls.Bool}, decls.Bool),
			decls.NewOverload(overloads.GreaterEqualsInt64,
				[]*exprpb.Type{decls.Int, decls.Int}, decls.Bool),
			decls.NewOverload(overloads.GreaterEqualsInt64Double,
				[]*exprpb.Type{decls.Int, decls.Double}, decls.Bool),
			decls.NewOverload(overloads.GreaterEqualsInt64Uint64,
				[]*exprpb.Type{decls.Int, decls.Uint}, decls.Bool),
			decls.NewOverload(overloads.GreaterEqualsUint64,
				[]*exprpb.Type{decls.Uint, decls.Uint}, decls.Bool),
			decls.NewOverload(overloads.GreaterEqualsUint64Double,
				[]*exprpb.Type{decls.Uint, decls.Double}, decls.Bool),
			decls.NewOverload(overloads.GreaterEqualsUint64Int64,
				[]*exprpb.Type{decls.Uint, decls.Int}, decls.Bool),
			decls.NewOverload(overloads.GreaterEqualsDouble,
				[]*exprpb.Type{decls.Double, decls.Double}, decls.Bool),
			decls.NewOverload(overloads.GreaterEqualsDoubleInt64,
				[]*exprpb.Type{decls.Double, decls.Int}, decls.Bool),
			decls.NewOverload(overloads.GreaterEqualsDoubleUint64,
				[]*exprpb.Type{decls.Double, decls.Uint}, decls.Bool),
			decls.NewOverload(overloads.GreaterEqualsString,
				[]*exprpb.Type{decls.String, decls.String}, decls.Bool),
			decls.NewOverload(overloads.GreaterEqualsBytes,
				[]*exprpb.Type{decls.Bytes, decls.Bytes}, decls.Bool),
			decls.NewOverload(overloads.GreaterEqualsTimestamp,
				[]*exprpb.Type{decls.Timestamp, decls.Timestamp}, decls.Bool),
			decls.NewOverload(overloads.GreaterEqualsDuration,
				[]*exprpb.Type{decls.Duration, decls.Duration}, decls.Bool)),
	}...)
}

// StandardDeclarations returns the Decls for all functions and constants in the evaluator.
func StandardDeclarations() []*exprpb.Decl {
	return standardDeclarations
}
