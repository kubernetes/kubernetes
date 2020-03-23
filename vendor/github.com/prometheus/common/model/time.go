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

var durationRE = regexp.MustCompile("^([0-9]+)(y|w|d|h|m|s|ms)$")

// ParseDuration parses a string into a time.Duration, assuming that a year
// always has 365d, a week always has 7d, and a day always has 24h.
func ParseDuration(durationStr string) (Duration, error) {
	matches := durationRE.FindStringSubmatch(durationStr)
	if len(matches) != 3 {
		return 0, fmt.Errorf("not a valid duration string: %q", durationStr)
	}
	var (
		n, _ = strconv.Atoi(matches[1])
		dur  = time.Duration(n) * time.Millisecond
	)
	switch unit := matches[2]; unit {
	case "y":
		dur *= 1000 * 60 * 60 * 24 * 365
	case "w":
		dur *= 1000 * 60 * 60 * 24 * 7
	case "d":
		dur *= 1000 * 60 * 60 * 24
	case "h":
		dur *= 1000 * 60 * 60
	case "m":
		dur *= 1000 * 60
	case "s":
		dur *= 1000
	case "ms":
		// Value already correct
	default:
		return 0, fmt.Errorf("invalid time unit in duration string: %q", unit)
	}
	return Duration(dur), nil
}

func (d Duration) String() string {
	var (
		ms   = int64(time.Duration(d) / time.Millisecond)
		unit = "ms"
	)
	if ms == 0 {
		return "0s"
	}
	factors := map[string]int64{
		"y":  1000 * 60 * 60 * 24 * 365,
		"w":  1000 * 60 * 60 * 24 * 7,
		"d":  1000 * 60 * 60 * 24,
		"h":  1000 * 60 * 60,
		"m":  1000 * 60,
		"s":  1000,
		"ms": 1,
	}

	switch int64(0) {
	case ms % factors["y"]:
		unit = "y"
	case ms % factors["w"]:
		unit = "w"
	case ms % factors["d"]:
		unit = "d"
	case ms % factors["h"]:
		unit = "h"
	case ms % factors["m"]:
		unit = "m"
	case ms % factors["s"]:
		unit = "s"
	}
	return fmt.Sprintf("%v%v", ms/factors[unit], unit)
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
