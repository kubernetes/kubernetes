// Implementation of TOML's local date/time.
//
// Copied over from Google's civil to avoid pulling all the Google dependencies.
// Originals:
//   https://raw.githubusercontent.com/googleapis/google-cloud-go/ed46f5086358513cf8c25f8e3f022cb838a49d66/civil/civil.go
// Changes:
//   * Renamed files from civil* to localtime*.
//   * Package changed from civil to toml.
//   * 'Local' prefix added to all structs.
//
// Copyright 2016 Google LLC
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

// Package civil implements types for civil time, a time-zone-independent
// representation of time that follows the rules of the proleptic
// Gregorian calendar with exactly 24-hour days, 60-minute hours, and 60-second
// minutes.
//
// Because they lack location information, these types do not represent unique
// moments or intervals of time. Use time.Time for that purpose.
package toml

import (
	"fmt"
	"time"
)

// A LocalDate represents a date (year, month, day).
//
// This type does not include location information, and therefore does not
// describe a unique 24-hour timespan.
type LocalDate struct {
	Year  int        // Year (e.g., 2014).
	Month time.Month // Month of the year (January = 1, ...).
	Day   int        // Day of the month, starting at 1.
}

// LocalDateOf returns the LocalDate in which a time occurs in that time's location.
func LocalDateOf(t time.Time) LocalDate {
	var d LocalDate
	d.Year, d.Month, d.Day = t.Date()
	return d
}

// ParseLocalDate parses a string in RFC3339 full-date format and returns the date value it represents.
func ParseLocalDate(s string) (LocalDate, error) {
	t, err := time.Parse("2006-01-02", s)
	if err != nil {
		return LocalDate{}, err
	}
	return LocalDateOf(t), nil
}

// String returns the date in RFC3339 full-date format.
func (d LocalDate) String() string {
	return fmt.Sprintf("%04d-%02d-%02d", d.Year, d.Month, d.Day)
}

// IsValid reports whether the date is valid.
func (d LocalDate) IsValid() bool {
	return LocalDateOf(d.In(time.UTC)) == d
}

// In returns the time corresponding to time 00:00:00 of the date in the location.
//
// In is always consistent with time.LocalDate, even when time.LocalDate returns a time
// on a different day. For example, if loc is America/Indiana/Vincennes, then both
//     time.LocalDate(1955, time.May, 1, 0, 0, 0, 0, loc)
// and
//     civil.LocalDate{Year: 1955, Month: time.May, Day: 1}.In(loc)
// return 23:00:00 on April 30, 1955.
//
// In panics if loc is nil.
func (d LocalDate) In(loc *time.Location) time.Time {
	return time.Date(d.Year, d.Month, d.Day, 0, 0, 0, 0, loc)
}

// AddDays returns the date that is n days in the future.
// n can also be negative to go into the past.
func (d LocalDate) AddDays(n int) LocalDate {
	return LocalDateOf(d.In(time.UTC).AddDate(0, 0, n))
}

// DaysSince returns the signed number of days between the date and s, not including the end day.
// This is the inverse operation to AddDays.
func (d LocalDate) DaysSince(s LocalDate) (days int) {
	// We convert to Unix time so we do not have to worry about leap seconds:
	// Unix time increases by exactly 86400 seconds per day.
	deltaUnix := d.In(time.UTC).Unix() - s.In(time.UTC).Unix()
	return int(deltaUnix / 86400)
}

// Before reports whether d1 occurs before d2.
func (d1 LocalDate) Before(d2 LocalDate) bool {
	if d1.Year != d2.Year {
		return d1.Year < d2.Year
	}
	if d1.Month != d2.Month {
		return d1.Month < d2.Month
	}
	return d1.Day < d2.Day
}

// After reports whether d1 occurs after d2.
func (d1 LocalDate) After(d2 LocalDate) bool {
	return d2.Before(d1)
}

// MarshalText implements the encoding.TextMarshaler interface.
// The output is the result of d.String().
func (d LocalDate) MarshalText() ([]byte, error) {
	return []byte(d.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface.
// The date is expected to be a string in a format accepted by ParseLocalDate.
func (d *LocalDate) UnmarshalText(data []byte) error {
	var err error
	*d, err = ParseLocalDate(string(data))
	return err
}

// A LocalTime represents a time with nanosecond precision.
//
// This type does not include location information, and therefore does not
// describe a unique moment in time.
//
// This type exists to represent the TIME type in storage-based APIs like BigQuery.
// Most operations on Times are unlikely to be meaningful. Prefer the LocalDateTime type.
type LocalTime struct {
	Hour       int // The hour of the day in 24-hour format; range [0-23]
	Minute     int // The minute of the hour; range [0-59]
	Second     int // The second of the minute; range [0-59]
	Nanosecond int // The nanosecond of the second; range [0-999999999]
}

// LocalTimeOf returns the LocalTime representing the time of day in which a time occurs
// in that time's location. It ignores the date.
func LocalTimeOf(t time.Time) LocalTime {
	var tm LocalTime
	tm.Hour, tm.Minute, tm.Second = t.Clock()
	tm.Nanosecond = t.Nanosecond()
	return tm
}

// ParseLocalTime parses a string and returns the time value it represents.
// ParseLocalTime accepts an extended form of the RFC3339 partial-time format. After
// the HH:MM:SS part of the string, an optional fractional part may appear,
// consisting of a decimal point followed by one to nine decimal digits.
// (RFC3339 admits only one digit after the decimal point).
func ParseLocalTime(s string) (LocalTime, error) {
	t, err := time.Parse("15:04:05.999999999", s)
	if err != nil {
		return LocalTime{}, err
	}
	return LocalTimeOf(t), nil
}

// String returns the date in the format described in ParseLocalTime. If Nanoseconds
// is zero, no fractional part will be generated. Otherwise, the result will
// end with a fractional part consisting of a decimal point and nine digits.
func (t LocalTime) String() string {
	s := fmt.Sprintf("%02d:%02d:%02d", t.Hour, t.Minute, t.Second)
	if t.Nanosecond == 0 {
		return s
	}
	return s + fmt.Sprintf(".%09d", t.Nanosecond)
}

// IsValid reports whether the time is valid.
func (t LocalTime) IsValid() bool {
	// Construct a non-zero time.
	tm := time.Date(2, 2, 2, t.Hour, t.Minute, t.Second, t.Nanosecond, time.UTC)
	return LocalTimeOf(tm) == t
}

// MarshalText implements the encoding.TextMarshaler interface.
// The output is the result of t.String().
func (t LocalTime) MarshalText() ([]byte, error) {
	return []byte(t.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface.
// The time is expected to be a string in a format accepted by ParseLocalTime.
func (t *LocalTime) UnmarshalText(data []byte) error {
	var err error
	*t, err = ParseLocalTime(string(data))
	return err
}

// A LocalDateTime represents a date and time.
//
// This type does not include location information, and therefore does not
// describe a unique moment in time.
type LocalDateTime struct {
	Date LocalDate
	Time LocalTime
}

// Note: We deliberately do not embed LocalDate into LocalDateTime, to avoid promoting AddDays and Sub.

// LocalDateTimeOf returns the LocalDateTime in which a time occurs in that time's location.
func LocalDateTimeOf(t time.Time) LocalDateTime {
	return LocalDateTime{
		Date: LocalDateOf(t),
		Time: LocalTimeOf(t),
	}
}

// ParseLocalDateTime parses a string and returns the LocalDateTime it represents.
// ParseLocalDateTime accepts a variant of the RFC3339 date-time format that omits
// the time offset but includes an optional fractional time, as described in
// ParseLocalTime. Informally, the accepted format is
//     YYYY-MM-DDTHH:MM:SS[.FFFFFFFFF]
// where the 'T' may be a lower-case 't'.
func ParseLocalDateTime(s string) (LocalDateTime, error) {
	t, err := time.Parse("2006-01-02T15:04:05.999999999", s)
	if err != nil {
		t, err = time.Parse("2006-01-02t15:04:05.999999999", s)
		if err != nil {
			return LocalDateTime{}, err
		}
	}
	return LocalDateTimeOf(t), nil
}

// String returns the date in the format described in ParseLocalDate.
func (dt LocalDateTime) String() string {
	return dt.Date.String() + "T" + dt.Time.String()
}

// IsValid reports whether the datetime is valid.
func (dt LocalDateTime) IsValid() bool {
	return dt.Date.IsValid() && dt.Time.IsValid()
}

// In returns the time corresponding to the LocalDateTime in the given location.
//
// If the time is missing or ambigous at the location, In returns the same
// result as time.LocalDate. For example, if loc is America/Indiana/Vincennes, then
// both
//     time.LocalDate(1955, time.May, 1, 0, 30, 0, 0, loc)
// and
//     civil.LocalDateTime{
//         civil.LocalDate{Year: 1955, Month: time.May, Day: 1}},
//         civil.LocalTime{Minute: 30}}.In(loc)
// return 23:30:00 on April 30, 1955.
//
// In panics if loc is nil.
func (dt LocalDateTime) In(loc *time.Location) time.Time {
	return time.Date(dt.Date.Year, dt.Date.Month, dt.Date.Day, dt.Time.Hour, dt.Time.Minute, dt.Time.Second, dt.Time.Nanosecond, loc)
}

// Before reports whether dt1 occurs before dt2.
func (dt1 LocalDateTime) Before(dt2 LocalDateTime) bool {
	return dt1.In(time.UTC).Before(dt2.In(time.UTC))
}

// After reports whether dt1 occurs after dt2.
func (dt1 LocalDateTime) After(dt2 LocalDateTime) bool {
	return dt2.Before(dt1)
}

// MarshalText implements the encoding.TextMarshaler interface.
// The output is the result of dt.String().
func (dt LocalDateTime) MarshalText() ([]byte, error) {
	return []byte(dt.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface.
// The datetime is expected to be a string in a format accepted by ParseLocalDateTime
func (dt *LocalDateTime) UnmarshalText(data []byte) error {
	var err error
	*dt, err = ParseLocalDateTime(string(data))
	return err
}
