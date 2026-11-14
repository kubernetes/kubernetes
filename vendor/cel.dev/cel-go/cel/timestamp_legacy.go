// Copyright 2026 Google LLC
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

	"cel.dev/cel-go/common/overloads"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
)

// timestampLegacyParsingLibrary re-declares the standard overloads affected by strict timestamp
// parsing with the bindings used before strict parsing was introduced. It is configured when
// StrictTimestampParsing(false) is set.
type timestampLegacyParsingLibrary struct{}

func (timestampLegacyParsingLibrary) CompileOptions() []EnvOption {
	return timestampLegacyParsingDeclarations
}

func (timestampLegacyParsingLibrary) ProgramOptions() []ProgramOption {
	return []ProgramOption{}
}

const (
	// Bounds of the CEL timestamp range: 0001-01-01T00:00:00Z to 9999-12-31T23:59:59.999999999Z.
	legacyMinUnixTime int64 = -62135596800
	legacyMaxUnixTime int64 = 253402300799
)

// legacyStringToTimestamp converts a string to a timestamp the way cel-go did before v0.30.0:
// any string accepted by time.Parse(time.RFC3339, ...) is accepted.
func legacyStringToTimestamp(val ref.Val) ref.Val {
	s, ok := val.(types.String)
	if !ok {
		return types.MaybeNoSuchOverloadErr(val)
	}
	t, err := time.Parse(time.RFC3339, string(s))
	if err != nil {
		return types.NewErr("type conversion error from '%s' to '%s'", types.StringType, types.TimestampType)
	}
	if t.Unix() < legacyMinUnixTime || t.Unix() > legacyMaxUnixTime {
		return types.NewErr("timestamp overflow")
	}
	return types.Timestamp{Time: t}
}

// legacyInTimeZone resolves a timezone argument the way cel-go did before v0.31.0: numeric
// offsets of the form [+-]HH:MM are not range checked.
func legacyInTimeZone(ts, tz ref.Val) (time.Time, error) {
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
	hr, err := strconv.Atoi(val[0:ind])
	if err != nil {
		return time.Time{}, err
	}
	min, err := strconv.Atoi(val[ind+1:])
	if err != nil {
		return time.Time{}, err
	}
	var offset int
	if val[0] == '-' {
		offset = hr*60 - min
	} else {
		offset = hr*60 + min
	}
	secondsEastOfUTC := int((time.Duration(offset) * time.Minute).Seconds())
	return t.In(time.FixedZone("", secondsEastOfUTC)), nil
}

// legacyTimestampAccessor builds a binding for a timestamp accessor with a timezone argument.
func legacyTimestampAccessor(get func(time.Time) int) func(ref.Val, ref.Val) ref.Val {
	return func(ts, tz ref.Val) ref.Val {
		t, err := legacyInTimeZone(ts, tz)
		if err != nil {
			return types.NewErrFromString(err.Error())
		}
		return types.Int(get(t))
	}
}

func legacyTimezoneOverload(function, overloadID string, get func(time.Time) int) EnvOption {
	return Function(function,
		MemberOverload(overloadID, []*Type{TimestampType, StringType}, IntType,
			BinaryBinding(legacyTimestampAccessor(get))))
}

var timestampLegacyParsingDeclarations = []EnvOption{
	Function(overloads.TypeConvertTimestamp,
		Overload(overloads.StringToTimestamp, []*Type{StringType}, TimestampType,
			UnaryBinding(legacyStringToTimestamp))),
	legacyTimezoneOverload(overloads.TimeGetFullYear, overloads.TimestampToYearWithTz,
		func(t time.Time) int { return t.Year() }),
	legacyTimezoneOverload(overloads.TimeGetMonth, overloads.TimestampToMonthWithTz,
		func(t time.Time) int { return int(t.Month()) - 1 }),
	legacyTimezoneOverload(overloads.TimeGetDayOfYear, overloads.TimestampToDayOfYearWithTz,
		func(t time.Time) int { return t.YearDay() - 1 }),
	legacyTimezoneOverload(overloads.TimeGetDayOfMonth, overloads.TimestampToDayOfMonthZeroBasedWithTz,
		func(t time.Time) int { return t.Day() - 1 }),
	legacyTimezoneOverload(overloads.TimeGetDate, overloads.TimestampToDayOfMonthOneBasedWithTz,
		func(t time.Time) int { return t.Day() }),
	legacyTimezoneOverload(overloads.TimeGetDayOfWeek, overloads.TimestampToDayOfWeekWithTz,
		func(t time.Time) int { return int(t.Weekday()) }),
	legacyTimezoneOverload(overloads.TimeGetHours, overloads.TimestampToHoursWithTz,
		func(t time.Time) int { return t.Hour() }),
	legacyTimezoneOverload(overloads.TimeGetMinutes, overloads.TimestampToMinutesWithTz,
		func(t time.Time) int { return t.Minute() }),
	legacyTimezoneOverload(overloads.TimeGetSeconds, overloads.TimestampToSecondsWithTz,
		func(t time.Time) int { return t.Second() }),
	legacyTimezoneOverload(overloads.TimeGetMilliseconds, overloads.TimestampToMillisecondsWithTz,
		func(t time.Time) int { return t.Nanosecond() / 1000000 }),
}
