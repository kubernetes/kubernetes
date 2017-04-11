// Copyright 2012 Google Inc. All Rights Reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package datastore

import (
	"testing"
	"time"
)

func TestUnixMicro(t *testing.T) {
	// Test that all these time.Time values survive a round trip to unix micros.
	testCases := []time.Time{
		{},
		time.Date(2, 1, 1, 0, 0, 0, 0, time.UTC),
		time.Date(23, 1, 1, 0, 0, 0, 0, time.UTC),
		time.Date(234, 1, 1, 0, 0, 0, 0, time.UTC),
		time.Date(1000, 1, 1, 0, 0, 0, 0, time.UTC),
		time.Date(1600, 1, 1, 0, 0, 0, 0, time.UTC),
		time.Date(1700, 1, 1, 0, 0, 0, 0, time.UTC),
		time.Date(1800, 1, 1, 0, 0, 0, 0, time.UTC),
		time.Date(1900, 1, 1, 0, 0, 0, 0, time.UTC),
		time.Unix(-1e6, -1000),
		time.Unix(-1e6, 0),
		time.Unix(-1e6, +1000),
		time.Unix(-60, -1000),
		time.Unix(-60, 0),
		time.Unix(-60, +1000),
		time.Unix(-1, -1000),
		time.Unix(-1, 0),
		time.Unix(-1, +1000),
		time.Unix(0, -3000),
		time.Unix(0, -2000),
		time.Unix(0, -1000),
		time.Unix(0, 0),
		time.Unix(0, +1000),
		time.Unix(0, +2000),
		time.Unix(+60, -1000),
		time.Unix(+60, 0),
		time.Unix(+60, +1000),
		time.Unix(+1e6, -1000),
		time.Unix(+1e6, 0),
		time.Unix(+1e6, +1000),
		time.Date(1999, 12, 31, 23, 59, 59, 999000, time.UTC),
		time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC),
		time.Date(2006, 1, 2, 15, 4, 5, 678000, time.UTC),
		time.Date(2009, 11, 10, 23, 0, 0, 0, time.UTC),
		time.Date(3456, 1, 1, 0, 0, 0, 0, time.UTC),
	}
	for _, tc := range testCases {
		got := fromUnixMicro(toUnixMicro(tc))
		if !got.Equal(tc) {
			t.Errorf("got %q, want %q", got, tc)
		}
	}

	// Test that a time.Time that isn't an integral number of microseconds
	// is not perfectly reconstructed after a round trip.
	t0 := time.Unix(0, 123)
	t1 := fromUnixMicro(toUnixMicro(t0))
	if t1.Nanosecond()%1000 != 0 || t0.Nanosecond()%1000 == 0 {
		t.Errorf("quantization to µs: got %q with %d ns, started with %d ns", t1, t1.Nanosecond(), t0.Nanosecond())
	}
}
