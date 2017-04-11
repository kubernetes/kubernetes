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
	"testing"
	"time"
)

func TestComparators(t *testing.T) {
	t1a := TimeFromUnix(0)
	t1b := TimeFromUnix(0)
	t2 := TimeFromUnix(2*second - 1)

	if !t1a.Equal(t1b) {
		t.Fatalf("Expected %s to be equal to %s", t1a, t1b)
	}
	if t1a.Equal(t2) {
		t.Fatalf("Expected %s to not be equal to %s", t1a, t2)
	}

	if !t1a.Before(t2) {
		t.Fatalf("Expected %s to be before %s", t1a, t2)
	}
	if t1a.Before(t1b) {
		t.Fatalf("Expected %s to not be before %s", t1a, t1b)
	}

	if !t2.After(t1a) {
		t.Fatalf("Expected %s to be after %s", t2, t1a)
	}
	if t1b.After(t1a) {
		t.Fatalf("Expected %s to not be after %s", t1b, t1a)
	}
}

func TestTimeConversions(t *testing.T) {
	unixSecs := int64(1136239445)
	unixNsecs := int64(123456789)
	unixNano := unixSecs*1e9 + unixNsecs

	t1 := time.Unix(unixSecs, unixNsecs-unixNsecs%nanosPerTick)
	t2 := time.Unix(unixSecs, unixNsecs)

	ts := TimeFromUnixNano(unixNano)
	if !ts.Time().Equal(t1) {
		t.Fatalf("Expected %s, got %s", t1, ts.Time())
	}

	// Test available precision.
	ts = TimeFromUnixNano(t2.UnixNano())
	if !ts.Time().Equal(t1) {
		t.Fatalf("Expected %s, got %s", t1, ts.Time())
	}

	if ts.UnixNano() != unixNano-unixNano%nanosPerTick {
		t.Fatalf("Expected %d, got %d", unixNano, ts.UnixNano())
	}
}

func TestDuration(t *testing.T) {
	duration := time.Second + time.Minute + time.Hour
	goTime := time.Unix(1136239445, 0)

	ts := TimeFromUnix(goTime.Unix())
	if !goTime.Add(duration).Equal(ts.Add(duration).Time()) {
		t.Fatalf("Expected %s to be equal to %s", goTime.Add(duration), ts.Add(duration))
	}

	earlier := ts.Add(-duration)
	delta := ts.Sub(earlier)
	if delta != duration {
		t.Fatalf("Expected %s to be equal to %s", delta, duration)
	}
}
