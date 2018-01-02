// Copyright (c) 2012-2016 The go-diff authors. All rights reserved.
// https://github.com/sergi/go-diff
// See the included LICENSE file for license details.
//
// go-diff is a Go implementation of Google's Diff, Match, and Patch library
// Original library is Copyright (c) 2006 Google Inc.
// http://code.google.com/p/google-diff-match-patch/

package diffmatchpatch

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMatchAlphabet(t *testing.T) {
	type TestCase struct {
		Pattern string

		Expected map[byte]int
	}

	dmp := New()

	for i, tc := range []TestCase{
		{
			Pattern: "abc",

			Expected: map[byte]int{
				'a': 4,
				'b': 2,
				'c': 1,
			},
		},
		{
			Pattern: "abcaba",

			Expected: map[byte]int{
				'a': 37,
				'b': 18,
				'c': 8,
			},
		},
	} {
		actual := dmp.MatchAlphabet(tc.Pattern)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}
}

func TestMatchBitap(t *testing.T) {
	type TestCase struct {
		Name string

		Text     string
		Pattern  string
		Location int

		Expected int
	}

	dmp := New()
	dmp.MatchDistance = 100
	dmp.MatchThreshold = 0.5

	for i, tc := range []TestCase{
		{"Exact match #1", "abcdefghijk", "fgh", 5, 5},
		{"Exact match #2", "abcdefghijk", "fgh", 0, 5},
		{"Fuzzy match #1", "abcdefghijk", "efxhi", 0, 4},
		{"Fuzzy match #2", "abcdefghijk", "cdefxyhijk", 5, 2},
		{"Fuzzy match #3", "abcdefghijk", "bxy", 1, -1},
		{"Overflow", "123456789xx0", "3456789x0", 2, 2},
		{"Before start match", "abcdef", "xxabc", 4, 0},
		{"Beyond end match", "abcdef", "defyy", 4, 3},
		{"Oversized pattern", "abcdef", "xabcdefy", 0, 0},
	} {
		actual := dmp.MatchBitap(tc.Text, tc.Pattern, tc.Location)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}

	dmp.MatchThreshold = 0.4

	for i, tc := range []TestCase{
		{"Threshold #1", "abcdefghijk", "efxyhi", 1, 4},
	} {
		actual := dmp.MatchBitap(tc.Text, tc.Pattern, tc.Location)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}

	dmp.MatchThreshold = 0.3

	for i, tc := range []TestCase{
		{"Threshold #2", "abcdefghijk", "efxyhi", 1, -1},
	} {
		actual := dmp.MatchBitap(tc.Text, tc.Pattern, tc.Location)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}

	dmp.MatchThreshold = 0.0

	for i, tc := range []TestCase{
		{"Threshold #3", "abcdefghijk", "bcdef", 1, 1},
	} {
		actual := dmp.MatchBitap(tc.Text, tc.Pattern, tc.Location)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}

	dmp.MatchThreshold = 0.5

	for i, tc := range []TestCase{
		{"Multiple select #1", "abcdexyzabcde", "abccde", 3, 0},
		{"Multiple select #2", "abcdexyzabcde", "abccde", 5, 8},
	} {
		actual := dmp.MatchBitap(tc.Text, tc.Pattern, tc.Location)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}

	// Strict location.
	dmp.MatchDistance = 10

	for i, tc := range []TestCase{
		{"Distance test #1", "abcdefghijklmnopqrstuvwxyz", "abcdefg", 24, -1},
		{"Distance test #2", "abcdefghijklmnopqrstuvwxyz", "abcdxxefg", 1, 0},
	} {
		actual := dmp.MatchBitap(tc.Text, tc.Pattern, tc.Location)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}

	// Loose location.
	dmp.MatchDistance = 1000

	for i, tc := range []TestCase{
		{"Distance test #3", "abcdefghijklmnopqrstuvwxyz", "abcdefg", 24, 0},
	} {
		actual := dmp.MatchBitap(tc.Text, tc.Pattern, tc.Location)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}
}

func TestMatchMain(t *testing.T) {
	type TestCase struct {
		Name string

		Text1    string
		Text2    string
		Location int

		Expected int
	}

	dmp := New()

	for i, tc := range []TestCase{
		{"Equality", "abcdef", "abcdef", 1000, 0},
		{"Null text", "", "abcdef", 1, -1},
		{"Null pattern", "abcdef", "", 3, 3},
		{"Exact match", "abcdef", "de", 3, 3},
		{"Beyond end match", "abcdef", "defy", 4, 3},
		{"Oversized pattern", "abcdef", "abcdefy", 0, 0},
	} {
		actual := dmp.MatchMain(tc.Text1, tc.Text2, tc.Location)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}

	dmp.MatchThreshold = 0.7

	for i, tc := range []TestCase{
		{"Complex match", "I am the very model of a modern major general.", " that berry ", 5, 4},
	} {
		actual := dmp.MatchMain(tc.Text1, tc.Text2, tc.Location)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}
}
