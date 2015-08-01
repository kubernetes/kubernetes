/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package main

import (
	"testing"
	"time"
)

func TestParseTimeISO8601(t *testing.T) {
	cases := []struct {
		input    string
		expected time.Time
		err      bool
	}{
		{"00", timeMustParse("15", "00"), false},
		{"49", time.Time{}, true},
		{"-2", time.Time{}, true},
		{"12:34:56", timeMustParse("15:04:05", "12:34:56"), false},
		{"123456", timeMustParse("15:04:05", "12:34:56"), false},
		{"12:34", timeMustParse("15:04:05", "12:34:00"), false},
		{"1234", timeMustParse("15:04:05", "12:34:00"), false},
		{"1234:56", time.Time{}, true},
		{"12:3456", time.Time{}, true},
		{"12:34:96", time.Time{}, true},
		{"12:34:-00", time.Time{}, true},
		{"123476", time.Time{}, true},
		{"12:-34", time.Time{}, true},
		{"12104", time.Time{}, true},

		{"00Z", timeMustParse("15 MST", "00 UTC"), false},
		{"-2Z", time.Time{}, true},
		{"12:34:56Z", timeMustParse("15:04:05 MST", "12:34:56 UTC"), false},
		{"12:34Z", timeMustParse("15:04:05 MST", "12:34:00 UTC"), false},
		{"12:34:-00Z", time.Time{}, true},
		{"12104Z", time.Time{}, true},

		{"00+00", timeMustParse("15 MST", "00 UTC"), false},
		{"-2+03", time.Time{}, true},
		{"11:34:56+12", timeMustParse("15:04:05 MST", "23:34:56 UTC"), false},
		{"12:34:14+10:30", timeMustParse("15:04:05 MST", "23:04:00 UTC"), false},
		{"12:34:-00+10", time.Time{}, true},
		{"1210+00:00", time.Time{}, true},
		{"12:10+0000", time.Time{}, true},
		{"1210Z+00", time.Time{}, true},

		{"00-00", time.Time{}, true},
		{"-2-03", time.Time{}, true},
		{"11:34:56-11", timeMustParse("15:04:05 MST", "00:34:56 UTC"), false},
		{"12:34:14-10:30", timeMustParse("15:04:05 MST", "02:04:00 UTC"), false},
		{"12:34:-00-10", time.Time{}, true},
		{"1210-00:00", time.Time{}, true},
		{"12:10-0000", time.Time{}, true},
		{"1210Z-00", time.Time{}, true},

		// boundary cases
		{"-01", time.Time{}, true},
		{"00", timeMustParse("15", "00"), false},
		{"23", timeMustParse("15", "23"), false},
		{"24", time.Time{}, true},
		{"00:-01", time.Time{}, true},
		{"00:00", timeMustParse("15:04", "00:00"), false},
		{"00:59", timeMustParse("15:04", "00:59"), false},
		{"00:60", time.Time{}, true},
		{"01:02:-01", time.Time{}, true},
		{"01:02:00", timeMustParse("15:04:05", "01:02:00"), false},
		{"01:02:59", timeMustParse("15:04:05", "01:02:59"), false},
		{"01:02:60", time.Time{}, true},
		{"01:02:03-13", time.Time{}, true},
		{"01:02:03-12", timeMustParse("15:04:05 MST", "01:02:03 UTC").Add(-12 * time.Hour), false},
		{"01:02:03+14", timeMustParse("15:04:05 MST", "15:02:03 UTC"), false},
		{"01:02:03+15", time.Time{}, true},
	}
	for i, test := range cases {
		curTime, err := parseTimeISO8601(test.input)
		if test.err {
			if err == nil {
				t.Errorf("case %d [%s]: expected error, got: %v", i, test.input, curTime)
			}
			continue
		}
		if err != nil {
			t.Errorf("case %d [%s]: unexpected error: %v", i, test.input, err)
			continue
		}
		if test.expected.Equal(curTime) {
			t.Errorf("case %d [%s]: expected: %v got: %v", i, test.input, test.expected, curTime)
		}
	}
}
