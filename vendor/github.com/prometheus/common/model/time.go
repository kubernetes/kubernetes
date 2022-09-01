// Copyright 2013 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	// MinimumTick is the minimum supported time resolution. This has to be
	// at least time.Second in order for the code below to work.
	minimumTick = time.Millisecond
	// second is the Time duration equivalent to one second.
	second = int64(time.Second / minimumTick)
	// The number of nanoseconds per minimum tick.
	nanosPerTick = int64(minimumTick / time.Nanosecond)

	// Earliest is the earliest Time representable. Handy for
	// initializing a high watermark.
	Earliest = Time(math.MinInt64)
	// Latest is the latest Time representable. Handy for initializing
	// a low watermark.
	Latest = Time(math.MaxInt64)
)

// Time is the number of milliseconds since the epoch
// (1970-01-01 00:00 UTC) excluding leap seconds.
type Time int64

// Interval describes an interval between two timestamps.
type Interval struct {
	Start, End Time
}

// Now returns the current time as a Time.
func Now() Time {
	return TimeFromUnixNano(time.Now().UnixNano())
}

// TimeFromUnix returns the Time equivalent to the Unix Time t
// provided in seconds.
func TimeFromUnix(t int64) Time {
	return Time(t * second)
}

// TimeFromUnixNano returns the Time equivalent to the Unix Time
// t provided in nanoseconds.
func TimeFromUnixNano(t int64) Time {
	return Time(t / nanosPerTick)
}

// Equal reports whether two Times represent the same instant.
func (t Time) Equal(o Time) bool {
	return t == o
}

// Before reports whether the Time t is before o.
func (t Time) Before(o Time) bool {
	return t < o
}

// After reports whether the Time t is after o.
func (t Time) After(o Time) bool {
	return t > o
}

// Add returns the Time t + d.
func (t Time) Add(d time.Duration) Time {
	return t + Time(d/minimumTick)
}

// Sub returns the Duration t - o.
func (t Time) Sub(o Time) time.Duration {
	return time.Duration(t-o) * minimumTick
}

// Time returns the time.Time representation of t.
func (t Time) Time() time.Time {
	return time.Unix(int64(t)/second, (int64(t)%second)*nanosPerTick)
}

// Unix returns t as a Unix time, the number of seconds elapsed
// since January 1, 1970 UTC.
func (t Time) Unix() int64 {
	return int64(t) / second
}

// UnixNano returns t as a Unix time, the number of nanoseconds elapsed
// since January 1, 1970 UTC.
func (t Time) UnixNano() int64 {
	return int64(t) * nanosPerTick
}

// The number of digits after the dot.
var dotPrecision = int(math.Log10(float64(second)))

// String returns a string representation of the Time.
func (t Time) String() string {
	return strconv.FormatFloat(float64(t)/float64(second), 'f', -1, 64)
}

// MarshalJSON implements the json.Marshaler interface.
func (t Time) MarshalJSON() ([]byte, error) {
	return []byte(t.String()), nil
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (t *Time) UnmarshalJSON(b []byte) error {
	p := strings.Split(string(b), ".")
	switch len(p) {
	case 1:
		v, err := strconv.ParseInt(string(p[0]), 10, 64)
		if err != nil {
			return err
		}
		*t = Time(v * second)

	case 2:
		v, err := strconv.ParseInt(string(p[0]), 10, 64)
		if err != nil {
			return err
		}
		v *= second

		prec := dotPrecision - len(p[1])
		if prec < 0 {
			p[1] = p[1][:dotPrecision]
		} else if prec > 0 {
			p[1] = p[1] + strings.Repeat("0", prec)
		}

		va, err := strconv.ParseInt(p[1], 10, 32)
		if err != nil {
			return err
		}

		// If the value was something like -0.1 the negative is lost in the
		// parsing because of the leading zero, this ensures that we capture it.
		if len(p[0]) > 0 && p[0][0] == '-' && v+va > 0 {
			*t = Time(v+va) * -1
		} else {
			*t = Time(v + va)
		}

	default:
		return fmt.Errorf("invalid time %q", string(b))
	}
	return nil
}

// Duration wraps time.Duration. It is used to parse the custom duration format
// from YAML.
// This type should not propagate beyond the scope of input/output processing.
type Duration time.Duration

// Set implements pflag/flag.Value
func (d *Duration) Set(s string) error {
	var err error
	*d, err = ParseDuration(s)
	return err
}

// Type implements pflag.Value
func (d *Duration) Type() string {
	return "duration"
}

var durationRE = regexp.MustCompile("^(([0-9]+)y)?(([0-9]+)w)?(([0-9]+)d)?(([0-9]+)h)?(([0-9]+)m)?(([0-9]+)s)?(([0-9]+)ms)?$")

// ParseDuration parses a string into a time.Duration, assuming that a year
// always has 365d, a week always has 7d, and a day always has 24h.
func ParseDuration(durationStr string) (Duration, error) {
	switch durationStr {
	case "0":
		// Allow 0 without a unit.
		return 0, nil
	case "":
		return 0, errors.New("empty duration string")
	}
	matches := durationRE.FindStringSubmatch(durationStr)
	if matches == nil {
		return 0, fmt.Errorf("not a valid duration string: %q", durationStr)
	}
	var dur time.Duration

	// Parse the match at pos `pos` in the regex and use `mult` to turn that
	// into ms, then add that value to the total parsed duration.
	var overflowErr error
	m := func(pos int, mult time.Duration) {
		if matches[pos] == "" {
			return
		}
		n, _ := strconv.Atoi(matches[pos])

		// Check if the provided duration overflows time.Duration (> ~ 290years).
		if n > int((1<<63-1)/mult/time.Millisecond) {
			overflowErr = errors.New("duration out of range")
		}
		d := time.Duration(n) * time.Millisecond
		dur += d * mult

		if dur < 0 {
			overflowErr = errors.New("duration out of range")
		}
	}

	m(2, 1000*60*60*24*365) // y
	m(4, 1000*60*60*24*7)   // w
	m(6, 1000*60*60*24)     // d
	m(8, 1000*60*60)        // h
	m(10, 1000*60)          // m
	m(12, 1000)             // s
	m(14, 1)                // ms

	return Duration(dur), overflowErr
}

func (d Duration) String() string {
	var (
		ms = int64(time.Duration(d) / time.Millisecond)
		r  = ""
	)
	if ms == 0 {
		return "0s"
	}

	f := func(unit string, mult int64, exact bool) {
		if exact && ms%mult != 0 {
			return
		}
		if v := ms / mult; v > 0 {
			r += fmt.Sprintf("%d%s", v, unit)
			ms -= v * mult
		}
	}

	// Only format years and weeks if the remainder is zero, as it is often
	// easier to read 90d than 12w6d.
	f("y", 1000*60*60*24*365, true)
	f("w", 1000*60*60*24*7, true)

	f("d", 1000*60*60*24, false)
	f("h", 1000*60*60, false)
	f("m", 1000*60, false)
	f("s", 1000, false)
	f("ms", 1, false)

	return r
}

// MarshalJSON implements the json.Marshaler interface.
func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(d.String())
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (d *Duration) UnmarshalJSON(bytes []byte) error {
	var s string
	if err := json.Unmarshal(bytes, &s); err != nil {
		return err
	}
	dur, err := ParseDuration(s)
	if err != nil {
		return err
	}
	*d = dur
	return nil
}

// MarshalText implements the encoding.TextMarshaler interface.
func (d *Duration) MarshalText() ([]byte, error) {
	return []byte(d.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface.
func (d *Duration) UnmarshalText(text []byte) error {
	var err error
	*d, err = ParseDuration(string(text))
	return err
}

// MarshalYAML implements the yaml.Marshaler interface.
func (d Duration) MarshalYAML() (interface{}, error) {
	return d.String(), nil
}

// UnmarshalYAML implements the yaml.Unmarshaler interface.
func (d *Duration) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	if err := unmarshal(&s); err != nil {
		return err
	}
	dur, err := ParseDuration(s)
	if err != nil {
		return err
	}
	*d = dur
	return nil
}
