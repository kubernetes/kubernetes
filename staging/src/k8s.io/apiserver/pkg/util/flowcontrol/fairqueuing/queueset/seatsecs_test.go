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

package queueset

import (
	"fmt"
	"math"
	"testing"
	"time"
)

func TestSeatSecondsString(t *testing.T) {
	digits := math.Log10(ssScale)
	expectFmt := fmt.Sprintf("%%%d.%dfss", int(digits+2), int(digits))
	testCases := []struct {
		ss     SeatSeconds
		expect string
	}{
		{ss: SeatSeconds(1), expect: fmt.Sprintf(expectFmt, 1.0/ssScale)},
		{ss: 0, expect: "0.00000000ss"},
		{ss: SeatsTimesDuration(1, time.Second), expect: "1.00000000ss"},
		{ss: SeatsTimesDuration(123, 100*time.Millisecond), expect: "12.30000000ss"},
		{ss: SeatsTimesDuration(1203, 10*time.Millisecond), expect: "12.03000000ss"},
	}
	for _, testCase := range testCases {
		actualStr := testCase.ss.String()
		if actualStr != testCase.expect {
			t.Errorf("SeatSeonds(%d) formatted as %q rather than expected %q", uint64(testCase.ss), actualStr, testCase.expect)
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
