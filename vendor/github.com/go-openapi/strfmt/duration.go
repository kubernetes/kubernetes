// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package strfmt

import (
	"database/sql/driver"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/globalsign/mgo/bson"
	"github.com/mailru/easyjson/jlexer"
	"github.com/mailru/easyjson/jwriter"
)

func init() {
	d := Duration(0)
	// register this format in the default registry
	Default.Add("duration", &d, IsDuration)
}

var (
	timeUnits = [][]string{
		{"ns", "nano"},
		{"us", "µs", "micro"},
		{"ms", "milli"},
		{"s", "sec"},
		{"m", "min"},
		{"h", "hr", "hour"},
		{"d", "day"},
		{"w", "wk", "week"},
	}

	timeMultiplier = map[string]time.Duration{
		"ns": time.Nanosecond,
		"us": time.Microsecond,
		"ms": time.Millisecond,
		"s":  time.Second,
		"m":  time.Minute,
		"h":  time.Hour,
		"d":  24 * time.Hour,
		"w":  7 * 24 * time.Hour,
	}

	durationMatcher = regexp.MustCompile(`((\d+)\s*([A-Za-zµ]+))`)
)

// IsDuration returns true if the provided string is a valid duration
func IsDuration(str string) bool {
	_, err := ParseDuration(str)
	return err == nil
}

// Duration represents a duration
//
// Duration stores a period of time as a nanosecond count, with the largest
// repesentable duration being approximately 290 years.
//
// swagger:strfmt duration
type Duration time.Duration

// MarshalText turns this instance into text
func (d Duration) MarshalText() ([]byte, error) {
	return []byte(time.Duration(d).String()), nil
}

// UnmarshalText hydrates this instance from text
func (d *Duration) UnmarshalText(data []byte) error { // validation is performed later on
	dd, err := ParseDuration(string(data))
	if err != nil {
		return err
	}
	*d = Duration(dd)
	return nil
}

// ParseDuration parses a duration from a string, compatible with scala duration syntax
func ParseDuration(cand string) (time.Duration, error) {
	if dur, err := time.ParseDuration(cand); err == nil {
		return dur, nil
	}

	var dur time.Duration
	ok := false
	for _, match := range durationMatcher.FindAllStringSubmatch(cand, -1) {

		factor, err := strconv.Atoi(match[2]) // converts string to int
		if err != nil {
			return 0, err
		}
		unit := strings.ToLower(strings.TrimSpace(match[3]))

		for _, variants := range timeUnits {
			last := len(variants) - 1
			multiplier := timeMultiplier[variants[0]]

			for i, variant := range variants {
				if (last == i && strings.HasPrefix(unit, variant)) || strings.EqualFold(variant, unit) {
					ok = true
					dur += (time.Duration(factor) * multiplier)
				}
			}
		}
	}

	if ok {
		return dur, nil
	}
	return 0, fmt.Errorf("Unable to parse %s as duration", cand)
}

// Scan reads a Duration value from database driver type.
func (d *Duration) Scan(raw interface{}) error {
	switch v := raw.(type) {
	// TODO: case []byte: // ?
	case int64:
		*d = Duration(v)
	case float64:
		*d = Duration(int64(v))
	case nil:
		*d = Duration(0)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.Duration from: %#v", v)
	}

	return nil
}

// Value converts Duration to a primitive value ready to be written to a database.
func (d Duration) Value() (driver.Value, error) {
	return driver.Value(int64(d)), nil
}

// String converts this duration to a string
func (d Duration) String() string {
	return time.Duration(d).String()
}

// MarshalJSON returns the Duration as JSON
func (d Duration) MarshalJSON() ([]byte, error) {
	var w jwriter.Writer
	d.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

// MarshalEasyJSON writes the Duration to a easyjson.Writer
func (d Duration) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(time.Duration(d).String())
}

// UnmarshalJSON sets the Duration from JSON
func (d *Duration) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	d.UnmarshalEasyJSON(&l)
	return l.Error()
}

// UnmarshalEasyJSON sets the Duration from a easyjson.Lexer
func (d *Duration) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		tt, err := ParseDuration(data)
		if err != nil {
			in.AddError(err)
			return
		}
		*d = Duration(tt)
	}
}

// GetBSON returns the Duration a bson.M{} map.
func (d *Duration) GetBSON() (interface{}, error) {
	return bson.M{"data": int64(*d)}, nil
}

// SetBSON sets the Duration from raw bson data
func (d *Duration) SetBSON(raw bson.Raw) error {
	var m bson.M
	if err := raw.Unmarshal(&m); err != nil {
		return err
	}

	if data, ok := m["data"].(int64); ok {
		*d = Duration(data)
		return nil
	}

	return errors.New("couldn't unmarshal bson raw value as Duration")
}
