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
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter/functions"
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

// Lib creates an EnvOption out of a Library, allowing libraries to be provided as functional args,
// and to be linked to each other.
func Lib(l Library) EnvOption {
	return func(e *Env) (*Env, error) {
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
