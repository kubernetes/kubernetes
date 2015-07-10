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

func equalsTimeCounts(a, b []timeCount) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].time != b[i].time || a[i].count != b[i].count {
			return false
		}
	}
	return true
}

func TestParseTimeCounts(t *testing.T) {
	cases := []struct {
		times  string
		counts string
		out    []timeCount
		err    bool
	}{
		{
			"00:00:01Z,00:02Z,03:00Z,04:00Z", "1,4,1,8", []timeCount{
				{time.Second, 1},
				{2 * time.Minute, 4},
				{3 * time.Hour, 1},
				{4 * time.Hour, 8},
			}, false,
		},
		{
			"00:01Z,00:02Z,00:05Z,00:03Z", "1,2,3,4", []timeCount{
				{1 * time.Minute, 1},
				{2 * time.Minute, 2},
				{3 * time.Minute, 4},
				{5 * time.Minute, 3},
			}, false,
		},
		{"00:00Z,00:01Z", "1,0", []timeCount{{0, 1}, {1 * time.Minute, 0}}, false},
		{"00:00+00,00:01+00:00,01:00Z", "0,-1,0", nil, true},
		{"-00:01Z,01:00Z", "0,1", nil, true},
		{"00:00Z", "1,2,3", nil, true},
	}
	for i, test := range cases {
		out, err := parseTimeCounts(test.times, test.counts)
		if test.err && err == nil {
			t.Errorf("case %d: expected error", i)
		} else if !test.err && err != nil {
			t.Errorf("case %d: unexpected error: %v", i, err)
		}
		if !test.err {
			if !equalsTimeCounts(test.out, out) {
				t.Errorf("case %d: expected timeCounts: %v got %v", i, test.out, out)
			}
		}
	}
}

func TestFindPos(t *testing.T) {
	cases := []struct {
		tc       []timeCount
		cur      int
		offset   time.Duration
		expected int
	}{
		{[]timeCount{{0, 1}, {4, 0}}, 1, 1, 1},
		{[]timeCount{{0, 1}, {4, 0}}, 0, 1, 1},
		{[]timeCount{{0, 1}, {4, 0}}, 1, 70, 0},
		{[]timeCount{{5, 1}, {100, 9000}, {4000, 2}, {10000, 4}}, 0, 0, 0},
		{[]timeCount{{5, 1}, {100, 9000}, {4000, 2}, {10000, 4}}, 1, 5000, 3},
		{[]timeCount{{5, 1}, {100, 9000}, {4000, 2}, {10000, 4}}, 2, 10000000, 0},
		{[]timeCount{{5, 1}, {100, 9000}, {4000, 2}, {10000, 4}}, 0, 50, 1},
	}
	for i, test := range cases {
		pos := findPos(test.tc, test.cur, test.offset)
		if pos != test.expected {
			t.Errorf("case %d: expected %d got %d", i, test.expected, pos)
		}
	}
}
