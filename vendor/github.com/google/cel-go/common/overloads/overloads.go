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

// Package overloads defines the internal overload identifiers for function and
// operator overloads.
package overloads

// Boolean logic overloads
const (
	Conditional               = "conditional"
	LogicalAnd                = "logical_and"
	LogicalOr                 = "logical_or"
	LogicalNot                = "logical_not"
	NotStrictlyFalse          = "not_strictly_false"
	Equals                    = "equals"
	NotEquals                 = "not_equals"
	LessBool                  = "less_bool"
	LessInt64                 = "less_int64"
	LessInt64Double           = "less_int64_double"
	LessInt64Uint64           = "less_int64_uint64"
	LessUint64                = "less_uint64"
	LessUint64Double          = "less_uint64_double"
	LessUint64Int64           = "less_uint64_int64"
	LessDouble                = "less_double"
	LessDoubleInt64           = "less_double_int64"
	LessDoubleUint64          = "less_double_uint64"
	LessString                = "less_string"
	LessBytes                 = "less_bytes"
	LessTimestamp             = "less_timestamp"
	LessDuration              = "less_duration"
	LessEqualsBool            = "less_equals_bool"
	LessEqualsInt64           = "less_equals_int64"
	LessEqualsInt64Double     = "less_equals_int64_double"
	LessEqualsInt64Uint64     = "less_equals_int64_uint64"
	LessEqualsUint64          = "less_equals_uint64"
	LessEqualsUint64Double    = "less_equals_uint64_double"
	LessEqualsUint64Int64     = "less_equals_uint64_int64"
	LessEqualsDouble          = "less_equals_double"
	LessEqualsDoubleInt64     = "less_equals_double_int64"
	LessEqualsDoubleUint64    = "less_equals_double_uint64"
	LessEqualsString          = "less_equals_string"
	LessEqualsBytes           = "less_equals_bytes"
	LessEqualsTimestamp       = "less_equals_timestamp"
	LessEqualsDuration        = "less_equals_duration"
	GreaterBool               = "greater_bool"
	GreaterInt64              = "greater_int64"
	GreaterInt64Double        = "greater_int64_double"
	GreaterInt64Uint64        = "greater_int64_uint64"
	GreaterUint64             = "greater_uint64"
	GreaterUint64Double       = "greater_uint64_double"
	GreaterUint64Int64        = "greater_uint64_int64"
	GreaterDouble             = "greater_double"
	GreaterDoubleInt64        = "greater_double_int64"
	GreaterDoubleUint64       = "greater_double_uint64"
	GreaterString             = "greater_string"
	GreaterBytes              = "greater_bytes"
	GreaterTimestamp          = "greater_timestamp"
	GreaterDuration           = "greater_duration"
	GreaterEqualsBool         = "greater_equals_bool"
	GreaterEqualsInt64        = "greater_equals_int64"
	GreaterEqualsInt64Double  = "greater_equals_int64_double"
	GreaterEqualsInt64Uint64  = "greater_equals_int64_uint64"
	GreaterEqualsUint64       = "greater_equals_uint64"
	GreaterEqualsUint64Double = "greater_equals_uint64_double"
	GreaterEqualsUint64Int64  = "greater_equals_uint64_int64"
	GreaterEqualsDouble       = "greater_equals_double"
	GreaterEqualsDoubleInt64  = "greater_equals_double_int64"
	GreaterEqualsDoubleUint64 = "greater_equals_double_uint64"
	GreaterEqualsString       = "greater_equals_string"
	GreaterEqualsBytes        = "greater_equals_bytes"
	GreaterEqualsTimestamp    = "greater_equals_timestamp"
	GreaterEqualsDuration     = "greater_equals_duration"
)

// Math overloads
const (
	AddInt64                   = "add_int64"
	AddUint64                  = "add_uint64"
	AddDouble                  = "add_double"
	AddString                  = "add_string"
	AddBytes                   = "add_bytes"
	AddList                    = "add_list"
	AddTimestampDuration       = "add_timestamp_duration"
	AddDurationTimestamp       = "add_duration_timestamp"
	AddDurationDuration        = "add_duration_duration"
	SubtractInt64              = "subtract_int64"
	SubtractUint64             = "subtract_uint64"
	SubtractDouble             = "subtract_double"
	SubtractTimestampTimestamp = "subtract_timestamp_timestamp"
	SubtractTimestampDuration  = "subtract_timestamp_duration"
	SubtractDurationDuration   = "subtract_duration_duration"
	MultiplyInt64              = "multiply_int64"
	MultiplyUint64             = "multiply_uint64"
	MultiplyDouble             = "multiply_double"
	DivideInt64                = "divide_int64"
	DivideUint64               = "divide_uint64"
	DivideDouble               = "divide_double"
	ModuloInt64                = "modulo_int64"
	ModuloUint64               = "modulo_uint64"
	NegateInt64                = "negate_int64"
	NegateDouble               = "negate_double"
)

// Index overloads
const (
	IndexList    = "index_list"
	IndexMap     = "index_map"
	IndexMessage = "index_message" // TODO: introduce concept of types.Message
)

// In operators
const (
	DeprecatedIn = "in"
	InList       = "in_list"
	InMap        = "in_map"
	InMessage    = "in_message" // TODO: introduce concept of types.Message
)

// Size overloads
const (
	Size           = "size"
	SizeString     = "size_string"
	SizeBytes      = "size_bytes"
	SizeList       = "size_list"
	SizeMap        = "size_map"
	SizeStringInst = "string_size"
	SizeBytesInst  = "bytes_size"
	SizeListInst   = "list_size"
	SizeMapInst    = "map_size"
)

// String function names.
const (
	Contains   = "contains"
	EndsWith   = "endsWith"
	Matches    = "matches"
	StartsWith = "startsWith"
)

// Extension function overloads with complex behaviors that need to be referenced in runtime and static analysis cost computations.
const (
	ExtQuoteString = "strings_quote"
)

// String function overload names.
const (
	ContainsString   = "contains_string"
	EndsWithString   = "ends_with_string"
	MatchesString    = "matches_string"
	StartsWithString = "starts_with_string"
)

// Extension function overloads with complex behaviors that need to be referenced in runtime and static analysis cost computations.
const (
	ExtFormatString = "string_format"
)

// Time-based functions.
const (
	TimeGetFullYear     = "getFullYear"
	TimeGetMonth        = "getMonth"
	TimeGetDayOfYear    = "getDayOfYear"
	TimeGetDate         = "getDate"
	TimeGetDayOfMonth   = "getDayOfMonth"
	TimeGetDayOfWeek    = "getDayOfWeek"
	TimeGetHours        = "getHours"
	TimeGetMinutes      = "getMinutes"
	TimeGetSeconds      = "getSeconds"
	TimeGetMilliseconds = "getMilliseconds"
)

// Timestamp overloads for time functions without timezones.
const (
	TimestampToYear                = "timestamp_to_year"
	TimestampToMonth               = "timestamp_to_month"
	TimestampToDayOfYear           = "timestamp_to_day_of_year"
	TimestampToDayOfMonthZeroBased = "timestamp_to_day_of_month"
	TimestampToDayOfMonthOneBased  = "timestamp_to_day_of_month_1_based"
	TimestampToDayOfWeek           = "timestamp_to_day_of_week"
	TimestampToHours               = "timestamp_to_hours"
	TimestampToMinutes             = "timestamp_to_minutes"
	TimestampToSeconds             = "timestamp_to_seconds"
	TimestampToMilliseconds        = "timestamp_to_milliseconds"
)

// Timestamp overloads for time functions with timezones.
const (
	TimestampToYearWithTz                = "timestamp_to_year_with_tz"
	TimestampToMonthWithTz               = "timestamp_to_month_with_tz"
	TimestampToDayOfYearWithTz           = "timestamp_to_day_of_year_with_tz"
	TimestampToDayOfMonthZeroBasedWithTz = "timestamp_to_day_of_month_with_tz"
	TimestampToDayOfMonthOneBasedWithTz  = "timestamp_to_day_of_month_1_based_with_tz"
	TimestampToDayOfWeekWithTz           = "timestamp_to_day_of_week_with_tz"
	TimestampToHoursWithTz               = "timestamp_to_hours_with_tz"
	TimestampToMinutesWithTz             = "timestamp_to_minutes_with_tz"
	TimestampToSecondsWithTz             = "timestamp_to_seconds_tz"
	TimestampToMillisecondsWithTz        = "timestamp_to_milliseconds_with_tz"
)

// Duration overloads for time functions.
const (
	DurationToHours        = "duration_to_hours"
	DurationToMinutes      = "duration_to_minutes"
	DurationToSeconds      = "duration_to_seconds"
	DurationToMilliseconds = "duration_to_milliseconds"
)

// Type conversion methods and overloads
const (
	TypeConvertInt       = "int"
	TypeConvertUint      = "uint"
	TypeConvertDouble    = "double"
	TypeConvertBool      = "bool"
	TypeConvertString    = "string"
	TypeConvertBytes     = "bytes"
	TypeConvertTimestamp = "timestamp"
	TypeConvertDuration  = "duration"
	TypeConvertType      = "type"
	TypeConvertDyn       = "dyn"
)

// Int conversion functions.
const (
	IntToInt       = "int64_to_int64"
	UintToInt      = "uint64_to_int64"
	DoubleToInt    = "double_to_int64"
	StringToInt    = "string_to_int64"
	TimestampToInt = "timestamp_to_int64"
	DurationToInt  = "duration_to_int64"
)

// Uint conversion functions.
const (
	UintToUint   = "uint64_to_uint64"
	IntToUint    = "int64_to_uint64"
	DoubleToUint = "double_to_uint64"
	StringToUint = "string_to_uint64"
)

// Double conversion functions.
const (
	DoubleToDouble = "double_to_double"
	IntToDouble    = "int64_to_double"
	UintToDouble   = "uint64_to_double"
	StringToDouble = "string_to_double"
)

// Bool conversion functions.
const (
	BoolToBool   = "bool_to_bool"
	StringToBool = "string_to_bool"
)

// Bytes conversion functions.
const (
	BytesToBytes  = "bytes_to_bytes"
	StringToBytes = "string_to_bytes"
)

// String conversion functions.
const (
	StringToString    = "string_to_string"
	BoolToString      = "bool_to_string"
	IntToString       = "int64_to_string"
	UintToString      = "uint64_to_string"
	DoubleToString    = "double_to_string"
	BytesToString     = "bytes_to_string"
	TimestampToString = "timestamp_to_string"
	DurationToString  = "duration_to_string"
)

// Timestamp conversion functions
const (
	TimestampToTimestamp = "timestamp_to_timestamp"
	StringToTimestamp    = "string_to_timestamp"
	IntToTimestamp       = "int64_to_timestamp"
)

// Convert duration from string
const (
	DurationToDuration = "duration_to_duration"
	StringToDuration   = "string_to_duration"
	IntToDuration      = "int64_to_duration"
)

// Convert to dyn
const (
	ToDyn = "to_dyn"
)

// Comprehensions helper methods, not directly accessible via a developer.
const (
	Iterator = "@iterator"
	HasNext  = "@hasNext"
	Next     = "@next"
)

// IsTypeConversionFunction returns whether the input function is a standard library type
// conversion function.
func IsTypeConversionFunction(function string) bool {
	switch function {
	case TypeConvertBool,
		TypeConvertBytes,
		TypeConvertDouble,
		TypeConvertDuration,
		TypeConvertDyn,
		TypeConvertInt,
		TypeConvertString,
		TypeConvertTimestamp,
		TypeConvertType,
		TypeConvertUint:
		return true
	default:
		return false
	}
}
