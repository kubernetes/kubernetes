// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2016 The Go Authors.  All rights reserved.
// https://github.com/golang/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package types

import (
	"math"
	"testing"
	"time"

	"github.com/gogo/protobuf/proto"
)

var tests = []struct {
	ts    *Timestamp
	valid bool
	t     time.Time
}{
	// The timestamp representing the Unix epoch date.
	{&Timestamp{0, 0}, true, utcDate(1970, 1, 1)},
	// The smallest representable timestamp.
	{&Timestamp{math.MinInt64, math.MinInt32}, false,
		time.Unix(math.MinInt64, math.MinInt32).UTC()},
	// The smallest representable timestamp with non-negative nanos.
	{&Timestamp{math.MinInt64, 0}, false, time.Unix(math.MinInt64, 0).UTC()},
	// The earliest valid timestamp.
	{&Timestamp{minValidSeconds, 0}, true, utcDate(1, 1, 1)},
	//"0001-01-01T00:00:00Z"},
	// The largest representable timestamp.
	{&Timestamp{math.MaxInt64, math.MaxInt32}, false,
		time.Unix(math.MaxInt64, math.MaxInt32).UTC()},
	// The largest representable timestamp with nanos in range.
	{&Timestamp{math.MaxInt64, 1e9 - 1}, false,
		time.Unix(math.MaxInt64, 1e9-1).UTC()},
	// The largest valid timestamp.
	{&Timestamp{maxValidSeconds - 1, 1e9 - 1}, true,
		time.Date(9999, 12, 31, 23, 59, 59, 1e9-1, time.UTC)},
	// The smallest invalid timestamp that is larger than the valid range.
	{&Timestamp{maxValidSeconds, 0}, false, time.Unix(maxValidSeconds, 0).UTC()},
	// A date before the epoch.
	{&Timestamp{-281836800, 0}, true, utcDate(1961, 1, 26)},
	// A date after the epoch.
	{&Timestamp{1296000000, 0}, true, utcDate(2011, 1, 26)},
	// A date after the epoch, in the middle of the day.
	{&Timestamp{1296012345, 940483}, true,
		time.Date(2011, 1, 26, 3, 25, 45, 940483, time.UTC)},
}

func TestValidateTimestamp(t *testing.T) {
	for _, s := range tests {
		got := validateTimestamp(s.ts)
		if (got == nil) != s.valid {
			t.Errorf("validateTimestamp(%v) = %v, want %v", s.ts, got, s.valid)
		}
	}
}

func TestTimestampFromProto(t *testing.T) {
	for _, s := range tests {
		got, err := TimestampFromProto(s.ts)
		if (err == nil) != s.valid {
			t.Errorf("TimestampFromProto(%v) error = %v, but valid = %t", s.ts, err, s.valid)
		} else if s.valid && got != s.t {
			t.Errorf("TimestampFromProto(%v) = %v, want %v", s.ts, got, s.t)
		}
	}
	// Special case: a nil TimestampFromProto is an error, but returns the 0 Unix time.
	got, err := TimestampFromProto(nil)
	want := time.Unix(0, 0).UTC()
	if got != want {
		t.Errorf("TimestampFromProto(nil) = %v, want %v", got, want)
	}
	if err == nil {
		t.Errorf("TimestampFromProto(nil) error = nil, expected error")
	}
}

func TestTimestampProto(t *testing.T) {
	for _, s := range tests {
		got, err := TimestampProto(s.t)
		if (err == nil) != s.valid {
			t.Errorf("TimestampProto(%v) error = %v, but valid = %t", s.t, err, s.valid)
		} else if s.valid && !proto.Equal(got, s.ts) {
			t.Errorf("TimestampProto(%v) = %v, want %v", s.t, got, s.ts)
		}
	}
	// No corresponding special case here: no time.Time results in a nil Timestamp.
}

func TestTimestampString(t *testing.T) {
	for _, test := range []struct {
		ts   *Timestamp
		want string
	}{
		// Not much testing needed because presumably time.Format is
		// well-tested.
		{&Timestamp{0, 0}, "1970-01-01T00:00:00Z"},
		{&Timestamp{minValidSeconds - 1, 0}, "(timestamp: &types.Timestamp{Seconds: -62135596801,\nNanos: 0,\n} before 0001-01-01)"},
	} {
		got := TimestampString(test.ts)
		if got != test.want {
			t.Errorf("TimestampString(%v) = %q, want %q", test.ts, got, test.want)
		}
	}
}

func utcDate(year, month, day int) time.Time {
	return time.Date(year, time.Month(month), day, 0, 0, 0, 0, time.UTC)
}
