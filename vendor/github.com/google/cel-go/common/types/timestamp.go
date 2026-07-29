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

package types

import (
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types/ref"

	anypb "google.golang.org/protobuf/types/known/anypb"
	structpb "google.golang.org/protobuf/types/known/structpb"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

// Timestamp type implementation which supports add, compare, and subtract
// operations. Timestamps are also capable of participating in dynamic
// function dispatch to instance methods.
type Timestamp struct {
	time.Time
}

func timestampOf(t time.Time) Timestamp {
	// Note that this function does not validate that time.Time is in our supported range.
	return Timestamp{Time: t}
}

const (
	// The number of seconds between year 1 and year 1970. This is borrowed from
	// https://golang.org/src/time/time.go.
	unixToInternal int64 = (1969*365 + 1969/4 - 1969/100 + 1969/400) * (60 * 60 * 24)

	// Number of seconds between `0001-01-01T00:00:00Z` and the Unix epoch.
	minUnixTime int64 = -62135596800
	// Number of seconds between `9999-12-31T23:59:59.999999999Z` and the Unix epoch.
	maxUnixTime int64 = 253402300799
)

// strictRFC3339Pattern gates the strings accepted by the `timestamp()` overload.
// time.Parse accepts inputs that RFC 3339 forbids: a ',' fractional-second
// separator, single-digit time fields, and numeric offsets whose hours exceed
// 23 or minutes exceed 59. Those slip past unnoticed and shift the parsed
// instant, so they are rejected before time.Parse runs. Month and day are held
// to the grammar ranges 01-12 and 01-31; the remaining calendar validation
// (day-of-month vs. month, leap years) is left to time.Parse.
//
// isStrictRFC3339 is the implementation used on the conversion path; the pattern
// is retained as the reference the scan is conformance tested against.
var strictRFC3339Pattern = regexp.MustCompile(
	`^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])[Tt]([01]\d|2[0-3]):[0-5]\d:([0-5]\d|60)(\.\d+)?([Zz]|[+-]([01]\d|2[0-3]):[0-5]\d)$`)

// isStrictRFC3339 reports whether s matches strictRFC3339Pattern, hand-rolled to
// keep the conversion path off the regexp engine and its per-call cost.
func isStrictRFC3339(s string) bool {
	// Shortest accepted form is "2006-01-02T15:04:05Z" (20 bytes): a 19-byte
	// fixed-width date-time followed by at least a 'Z'/'z' zone.
	if len(s) < 20 {
		return false
	}
	// full-date "T" partial-time
	if !isYear(s[0:4]) || !isChar(s[4], '-') || !isMonth(s[5:7]) || !isChar(s[7], '-') || !isDay(s[8:10]) ||
		!isChar(s[10], 't') ||
		!isHour(s[11:13]) || !isChar(s[13], ':') || !isMinute(s[14:16]) || !isChar(s[16], ':') || !isSecond(s[17:19]) {
		return false
	}
	rest := s[19:]
	// optional fractional seconds: "." 1*DIGIT
	if rest[0] == '.' {
		rest = rest[1:]
		n := 0
		for n < len(rest) && isDigit(rest[n]) {
			n++
		}
		if n == 0 {
			return false
		}
		rest = rest[n:]
	}
	// time-offset: "Z" or ("+" / "-") time-hour ":" time-minute
	if len(rest) == 1 {
		return isChar(rest[0], 'z')
	}
	if len(rest) == 6 && (rest[0] == '+' || rest[0] == '-') {
		return isHour(rest[1:3]) && isChar(rest[3], ':') && isMinute(rest[4:6])
	}
	return false
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

// isChar reports whether got is want, case-insensitively; want must be lower case.
func isChar(got, want byte) bool {
	g, w := rune(got), rune(want)
	return g == w || unicode.ToLower(g) == w
}

// inRange reports whether s is all decimal digits and its value lies in [lo, hi].
func inRange(s string, lo, hi uint64) bool {
	u, err := strconv.ParseUint(s, 10, 64)
	return err == nil && u >= lo && u <= hi
}

func isYear(s string) bool   { return inRange(s, 0, 9999) }
func isMonth(s string) bool  { return inRange(s, 1, 12) }
func isDay(s string) bool    { return inRange(s, 1, 31) }
func isHour(s string) bool   { return inRange(s, 0, 23) }
func isMinute(s string) bool { return inRange(s, 0, 59) }

// isSecond permits 60 for a leap second.
func isSecond(s string) bool { return inRange(s, 0, 60) }

// Add implements traits.Adder.Add.
func (t Timestamp) Add(other ref.Val) ref.Val {
	switch other.Type() {
	case DurationType:
		return other.(Duration).Add(t)
	}
	return MaybeNoSuchOverloadErr(other)
}

// Compare implements traits.Comparer.Compare.
func (t Timestamp) Compare(other ref.Val) ref.Val {
	if TimestampType != other.Type() {
		return MaybeNoSuchOverloadErr(other)
	}
	ts1 := t.Time
	ts2 := other.(Timestamp).Time
	switch {
	case ts1.Before(ts2):
		return IntNegOne
	case ts1.After(ts2):
		return IntOne
	default:
		return IntZero
	}
}

// ConvertToNative implements ref.Val.ConvertToNative.
func (t Timestamp) ConvertToNative(typeDesc reflect.Type) (any, error) {
	// If the timestamp is already assignable to the desired type return it.
	if reflect.TypeOf(t.Time).AssignableTo(typeDesc) {
		return t.Time, nil
	}
	if reflect.TypeOf(t).AssignableTo(typeDesc) {
		return t, nil
	}
	switch typeDesc {
	case anyValueType:
		// Pack the underlying time as a tpb.Timestamp into an Any value.
		return anypb.New(tpb.New(t.Time))
	case JSONValueType:
		// CEL follows the proto3 to JSON conversion which formats as an RFC 3339 encoded JSON
		// string.
		v := t.ConvertToType(StringType)
		if IsError(v) {
			return nil, v.(*Err)
		}
		return structpb.NewStringValue(string(v.(String))), nil
	case timestampValueType:
		// Unwrap the underlying tpb.Timestamp.
		return tpb.New(t.Time), nil
	}
	return nil, fmt.Errorf("type conversion error from 'Timestamp' to '%v'", typeDesc)
}

// ConvertToType implements ref.Val.ConvertToType.
func (t Timestamp) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case StringType:
		return String(t.Format(time.RFC3339Nano))
	case IntType:
		// Return the Unix time in seconds since 1970
		return Int(t.Unix())
	case TimestampType:
		return t
	case TypeType:
		return TimestampType
	}
	return NewErr("type conversion error from '%s' to '%s'", TimestampType, typeVal)
}

// Equal implements ref.Val.Equal.
func (t Timestamp) Equal(other ref.Val) ref.Val {
	otherTime, ok := other.(Timestamp)
	return Bool(ok && t.Time.Equal(otherTime.Time))
}

// IsZeroValue returns true if the timestamp is epoch 0.
func (t Timestamp) IsZeroValue() bool {
	return t.IsZero()
}

// Receive implements traits.Receiver.Receive.
func (t Timestamp) Receive(function string, overload string, args []ref.Val) ref.Val {
	switch len(args) {
	case 0:
		if f, found := timestampZeroArgOverloads[function]; found {
			return f(t.Time)
		}
	case 1:
		if f, found := timestampOneArgOverloads[function]; found {
			return f(t.Time, args[0])
		}
	}
	return NoSuchOverloadErr()
}

// Subtract implements traits.Subtractor.Subtract.
func (t Timestamp) Subtract(subtrahend ref.Val) ref.Val {
	switch subtrahend.Type() {
	case DurationType:
		dur := subtrahend.(Duration)
		val, err := subtractTimeDurationChecked(t.Time, dur.Duration)
		if err != nil {
			return WrapErr(err)
		}
		return timestampOf(val)
	case TimestampType:
		t2 := subtrahend.(Timestamp).Time
		val, err := subtractTimeChecked(t.Time, t2)
		if err != nil {
			return WrapErr(err)
		}
		return durationOf(val)
	}
	return MaybeNoSuchOverloadErr(subtrahend)
}

// Type implements ref.Val.Type.
func (t Timestamp) Type() ref.Type {
	return TimestampType
}

// Value implements ref.Val.Value.
func (t Timestamp) Value() any {
	return t.Time
}

func (t Timestamp) format(sb *strings.Builder) {
	fmt.Fprintf(sb, `timestamp("%s")`, t.Time.UTC().Format(time.RFC3339Nano))
}

var (
	timestampValueType = reflect.TypeOf(&tpb.Timestamp{})

	timestampZeroArgOverloads = map[string]func(time.Time) ref.Val{
		overloads.TimeGetFullYear:     timestampGetFullYear,
		overloads.TimeGetMonth:        timestampGetMonth,
		overloads.TimeGetDayOfYear:    timestampGetDayOfYear,
		overloads.TimeGetDate:         timestampGetDayOfMonthOneBased,
		overloads.TimeGetDayOfMonth:   timestampGetDayOfMonthZeroBased,
		overloads.TimeGetDayOfWeek:    timestampGetDayOfWeek,
		overloads.TimeGetHours:        timestampGetHours,
		overloads.TimeGetMinutes:      timestampGetMinutes,
		overloads.TimeGetSeconds:      timestampGetSeconds,
		overloads.TimeGetMilliseconds: timestampGetMilliseconds}

	timestampOneArgOverloads = map[string]func(time.Time, ref.Val) ref.Val{
		overloads.TimeGetFullYear:     timestampGetFullYearWithTz,
		overloads.TimeGetMonth:        timestampGetMonthWithTz,
		overloads.TimeGetDayOfYear:    timestampGetDayOfYearWithTz,
		overloads.TimeGetDate:         timestampGetDayOfMonthOneBasedWithTz,
		overloads.TimeGetDayOfMonth:   timestampGetDayOfMonthZeroBasedWithTz,
		overloads.TimeGetDayOfWeek:    timestampGetDayOfWeekWithTz,
		overloads.TimeGetHours:        timestampGetHoursWithTz,
		overloads.TimeGetMinutes:      timestampGetMinutesWithTz,
		overloads.TimeGetSeconds:      timestampGetSecondsWithTz,
		overloads.TimeGetMilliseconds: timestampGetMillisecondsWithTz}
)

type timestampVisitor func(time.Time) ref.Val

func timestampGetFullYear(t time.Time) ref.Val {
	return Int(t.Year())
}
func timestampGetMonth(t time.Time) ref.Val {
	// CEL spec indicates that the month should be 0-based, but the Time value
	// for Month() is 1-based.
	return Int(t.Month() - 1)
}
func timestampGetDayOfYear(t time.Time) ref.Val {
	return Int(t.YearDay() - 1)
}
func timestampGetDayOfMonthZeroBased(t time.Time) ref.Val {
	return Int(t.Day() - 1)
}
func timestampGetDayOfMonthOneBased(t time.Time) ref.Val {
	return Int(t.Day())
}
func timestampGetDayOfWeek(t time.Time) ref.Val {
	return Int(t.Weekday())
}
func timestampGetHours(t time.Time) ref.Val {
	return Int(t.Hour())
}
func timestampGetMinutes(t time.Time) ref.Val {
	return Int(t.Minute())
}
func timestampGetSeconds(t time.Time) ref.Val {
	return Int(t.Second())
}
func timestampGetMilliseconds(t time.Time) ref.Val {
	return Int(t.Nanosecond() / 1000000)
}

func timestampGetFullYearWithTz(t time.Time, tz ref.Val) ref.Val {
	return timeZone(tz, timestampGetFullYear)(t)
}
func timestampGetMonthWithTz(t time.Time, tz ref.Val) ref.Val {
	return timeZone(tz, timestampGetMonth)(t)
}
func timestampGetDayOfYearWithTz(t time.Time, tz ref.Val) ref.Val {
	return timeZone(tz, timestampGetDayOfYear)(t)
}
func timestampGetDayOfMonthZeroBasedWithTz(t time.Time, tz ref.Val) ref.Val {
	return timeZone(tz, timestampGetDayOfMonthZeroBased)(t)
}
func timestampGetDayOfMonthOneBasedWithTz(t time.Time, tz ref.Val) ref.Val {
	return timeZone(tz, timestampGetDayOfMonthOneBased)(t)
}
func timestampGetDayOfWeekWithTz(t time.Time, tz ref.Val) ref.Val {
	return timeZone(tz, timestampGetDayOfWeek)(t)
}
func timestampGetHoursWithTz(t time.Time, tz ref.Val) ref.Val {
	return timeZone(tz, timestampGetHours)(t)
}
func timestampGetMinutesWithTz(t time.Time, tz ref.Val) ref.Val {
	return timeZone(tz, timestampGetMinutes)(t)
}
func timestampGetSecondsWithTz(t time.Time, tz ref.Val) ref.Val {
	return timeZone(tz, timestampGetSeconds)(t)
}
func timestampGetMillisecondsWithTz(t time.Time, tz ref.Val) ref.Val {
	return timeZone(tz, timestampGetMilliseconds)(t)
}

func timeZone(tz ref.Val, visitor timestampVisitor) timestampVisitor {
	return func(t time.Time) ref.Val {
		if StringType != tz.Type() {
			return MaybeNoSuchOverloadErr(tz)
		}
		val := string(tz.(String))
		ind := strings.Index(val, ":")
		if ind == -1 {
			loc, err := time.LoadLocation(val)
			if err != nil {
				return WrapErr(err)
			}
			return visitor(t.In(loc))
		}

		// If the input is not the name of a timezone (for example, 'US/Central'), it should be a numerical offset from UTC
		// in the format ^(+|-)(0[0-9]|1[0-4]):[0-5][0-9]$. The numerical input is parsed in terms of hours and minutes.
		hr, err := strconv.Atoi(string(val[0:ind]))
		if err != nil {
			return WrapErr(err)
		}
		min, err := strconv.Atoi(string(val[ind+1:]))
		if err != nil {
			return WrapErr(err)
		}
		if min < 0 || min > 59 {
			return WrapErr(fmt.Errorf("timezone offset minutes out of range [0, 59]: %s", val))
		}
		var offset int
		if string(val[0]) == "-" {
			offset = hr*60 - min
		} else {
			offset = hr*60 + min
		}
		secondsEastOfUTC := int((time.Duration(offset) * time.Minute).Seconds())
		timezone := time.FixedZone("", secondsEastOfUTC)
		return visitor(t.In(timezone))
	}
}
