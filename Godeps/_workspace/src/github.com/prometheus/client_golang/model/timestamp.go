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
	"math"
	"strconv"

	native_time "time"
)

// Timestamp is the number of milliseconds since the epoch
// (1970-01-01 00:00 UTC) excluding leap seconds.
type Timestamp int64

const (
	// MinimumTick is the minimum supported time resolution. This has to be
	// at least native_time.Second in order for the code below to work.
	MinimumTick = native_time.Millisecond
	// second is the timestamp duration equivalent to one second.
	second = int64(native_time.Second / MinimumTick)
	// The number of nanoseconds per minimum tick.
	nanosPerTick = int64(MinimumTick / native_time.Nanosecond)

	// Earliest is the earliest timestamp representable. Handy for
	// initializing a high watermark.
	Earliest = Timestamp(math.MinInt64)
	// Latest is the latest timestamp representable. Handy for initializing
	// a low watermark.
	Latest = Timestamp(math.MaxInt64)
)

// Equal reports whether two timestamps represent the same instant.
func (t Timestamp) Equal(o Timestamp) bool {
	return t == o
}

// Before reports whether the timestamp t is before o.
func (t Timestamp) Before(o Timestamp) bool {
	return t < o
}

// After reports whether the timestamp t is after o.
func (t Timestamp) After(o Timestamp) bool {
	return t > o
}

// Add returns the Timestamp t + d.
func (t Timestamp) Add(d native_time.Duration) Timestamp {
	return t + Timestamp(d/MinimumTick)
}

// Sub returns the Duration t - o.
func (t Timestamp) Sub(o Timestamp) native_time.Duration {
	return native_time.Duration(t-o) * MinimumTick
}

// Time returns the time.Time representation of t.
func (t Timestamp) Time() native_time.Time {
	return native_time.Unix(int64(t)/second, (int64(t)%second)*nanosPerTick)
}

// Unix returns t as a Unix time, the number of seconds elapsed
// since January 1, 1970 UTC.
func (t Timestamp) Unix() int64 {
	return int64(t) / second
}

// UnixNano returns t as a Unix time, the number of nanoseconds elapsed
// since January 1, 1970 UTC.
func (t Timestamp) UnixNano() int64 {
	return int64(t) * nanosPerTick
}

// String returns a string representation of the timestamp.
func (t Timestamp) String() string {
	return strconv.FormatFloat(float64(t)/float64(second), 'f', -1, 64)
}

// MarshalJSON implements the json.Marshaler interface.
func (t Timestamp) MarshalJSON() ([]byte, error) {
	return []byte(t.String()), nil
}

// Now returns the current time as a Timestamp.
func Now() Timestamp {
	return TimestampFromTime(native_time.Now())
}

// TimestampFromTime returns the Timestamp equivalent to the time.Time t.
func TimestampFromTime(t native_time.Time) Timestamp {
	return TimestampFromUnixNano(t.UnixNano())
}

// TimestampFromUnix returns the Timestamp equivalent to the Unix timestamp t
// provided in seconds.
func TimestampFromUnix(t int64) Timestamp {
	return Timestamp(t * second)
}

// TimestampFromUnixNano returns the Timestamp equivalent to the Unix timestamp
// t provided in nanoseconds.
func TimestampFromUnixNano(t int64) Timestamp {
	return Timestamp(t / nanosPerTick)
}
