// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ptypes

import (
	"math"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"

	tspb "github.com/golang/protobuf/ptypes/timestamp"
)

var tests = []struct {
	ts    *tspb.Timestamp
	valid bool
	t     time.Time
}{
	// The timestamp representing the Unix epoch date.
	{&tspb.Timestamp{Seconds: 0, Nanos: 0}, true, utcDate(1970, 1, 1)},
	// The smallest representable timestamp.
	{&tspb.Timestamp{Seconds: math.MinInt64, Nanos: math.MinInt32}, false,
		time.Unix(math.MinInt64, math.MinInt32).UTC()},
	// The smallest representable timestamp with non-negative nanos.
	{&tspb.Timestamp{Seconds: math.MinInt64, Nanos: 0}, false, time.Unix(math.MinInt64, 0).UTC()},
	// The earliest valid timestamp.
	{&tspb.Timestamp{Seconds: minValidSeconds, Nanos: 0}, true, utcDate(1, 1, 1)},
	//"0001-01-01T00:00:00Z"},
	// The largest representable timestamp.
	{&tspb.Timestamp{Seconds: math.MaxInt64, Nanos: math.MaxInt32}, false,
		time.Unix(math.MaxInt64, math.MaxInt32).UTC()},
	// The largest representable timestamp with nanos in range.
	{&tspb.Timestamp{Seconds: math.MaxInt64, Nanos: 1e9 - 1}, false,
		time.Unix(math.MaxInt64, 1e9-1).UTC()},
	// The largest valid timestamp.
	{&tspb.Timestamp{Seconds: maxValidSeconds - 1, Nanos: 1e9 - 1}, true,
		time.Date(9999, 12, 31, 23, 59, 59, 1e9-1, time.UTC)},
	// The smallest invalid timestamp that is larger than the valid range.
	{&tspb.Timestamp{Seconds: maxValidSeconds, Nanos: 0}, false, time.Unix(maxValidSeconds, 0).UTC()},
	// A date before the epoch.
	{&tspb.Timestamp{Seconds: -281836800, Nanos: 0}, true, utcDate(1961, 1, 26)},
	// A date after the epoch.
	{&tspb.Timestamp{Seconds: 1296000000, Nanos: 0}, true, utcDate(2011, 1, 26)},
	// A date after the epoch, in the middle of the day.
	{&tspb.Timestamp{Seconds: 1296012345, Nanos: 940483}, true,
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

func TestTimestamp(t *testing.T) {
	for _, s := range tests {
		got, err := Timestamp(s.ts)
		if (err == nil) != s.valid {
			t.Errorf("Timestamp(%v) error = %v, but valid = %t", s.ts, err, s.valid)
		} else if s.valid && got != s.t {
			t.Errorf("Timestamp(%v) = %v, want %v", s.ts, got, s.t)
		}
	}
	// Special case: a nil Timestamp is an error, but returns the 0 Unix time.
	got, err := Timestamp(nil)
	want := time.Unix(0, 0).UTC()
	if got != want {
		t.Errorf("Timestamp(nil) = %v, want %v", got, want)
	}
	if err == nil {
		t.Errorf("Timestamp(nil) error = nil, expected error")
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
		ts   *tspb.Timestamp
		want string
	}{
		// Not much testing needed because presumably time.Format is
		// well-tested.
		{&tspb.Timestamp{Seconds: 0, Nanos: 0}, "1970-01-01T00:00:00Z"},
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

func TestTimestampNow(t *testing.T) {
	// Bracket the expected time.
	before := time.Now()
	ts := TimestampNow()
	after := time.Now()

	tm, err := Timestamp(ts)
	if err != nil {
		t.Errorf("between %v and %v\nTimestampNow() = %v\nwhich is invalid (%v)", before, after, ts, err)
	}
	if tm.Before(before) || tm.After(after) {
		t.Errorf("between %v and %v\nTimestamp(TimestampNow()) = %v", before, after, tm)
	}
}
