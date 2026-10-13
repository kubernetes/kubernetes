/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package request

import (
	"math"
	"testing"
	"time"
)

// TestSeatSecondsString exercises the SeatSeconds constructor and de-constructors (String, ToFloat).
func TestSeatSecondsString(t *testing.T) {
	testCases := []struct {
		ss          SeatSeconds
		expectFloat float64
		expectStr   string
	}{
		{ss: SeatSeconds(1), expectFloat: 1.0 / ssScale, expectStr: "0.00000001ss"},
		{ss: SeatSeconds(ssScale - 1), expectFloat: (ssScale - 1) / ssScale, expectStr: "0.99999999ss"},
		{ss: 0, expectFloat: 0, expectStr: "0.00000000ss"},
		{ss: SeatsTimesDuration(1, time.Second), expectFloat: 1, expectStr: "1.00000000ss"},
		{ss: SeatsTimesDuration(123, 100*time.Millisecond), expectFloat: 12.3, expectStr: "12.30000000ss"},
		{ss: SeatsTimesDuration(1203, 10*time.Millisecond), expectFloat: 12.03, expectStr: "12.03000000ss"},
	}
	for _, testCase := range testCases {
		actualStr := testCase.ss.String()
		if actualStr != testCase.expectStr {
			t.Errorf("SeatSeconds(%d).String() is %q but expected %q", uint64(testCase.ss), actualStr, testCase.expectStr)
		}
		actualFloat := testCase.ss.ToFloat()
		if math.Round(actualFloat*ssScale) != math.Round(testCase.expectFloat*ssScale) {
			t.Errorf("SeatSeconds(%d).ToFloat() is %v but expected %v", uint64(testCase.ss), actualFloat, testCase.expectFloat)
		}
	}
}

func TestSeatSecondsPerSeat(t *testing.T) {
	testCases := []struct {
		ss     SeatSeconds
		seats  float64
		expect time.Duration
	}{
		{ss: SeatsTimesDuration(10, time.Second), seats: 1, expect: 10 * time.Second},
		{ss: SeatsTimesDuration(1, time.Second), seats: 10, expect: 100 * time.Millisecond},
		{ss: SeatsTimesDuration(13, 5*time.Millisecond), seats: 5, expect: 13 * time.Millisecond},
		{ss: SeatsTimesDuration(12, 0), seats: 10, expect: 0},
	}
	for _, testCase := range testCases {
		actualDuration := testCase.ss.DurationPerSeat(testCase.seats)
		if actualDuration != testCase.expect {
			t.Errorf("DurationPerSeats returned %v rather than expected %q", actualDuration, testCase.expect)
		}
	}
}
