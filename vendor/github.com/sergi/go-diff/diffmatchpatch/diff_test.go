// Copyright (c) 2012-2016 The go-diff authors. All rights reserved.
// https://github.com/sergi/go-diff
// See the included LICENSE file for license details.
//
// go-diff is a Go implementation of Google's Diff, Match, and Patch library
// Original library is Copyright (c) 2006 Google Inc.
// http://code.google.com/p/google-diff-match-patch/

package diffmatchpatch

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
)

func pretty(diffs []Diff) string {
	var w bytes.Buffer

	for i, diff := range diffs {
		_, _ = w.WriteString(fmt.Sprintf("%v. ", i))

		switch diff.Type {
		case DiffInsert:
			_, _ = w.WriteString("DiffIns")
		case DiffDelete:
			_, _ = w.WriteString("DiffDel")
		case DiffEqual:
			_, _ = w.WriteString("DiffEql")
		default:
			_, _ = w.WriteString("Unknown")
		}

		_, _ = w.WriteString(fmt.Sprintf(": %v\n", diff.Text))
	}

	return w.String()
}

func diffRebuildTexts(diffs []Diff) []string {
	texts := []string{"", ""}

	for _, d := range diffs {
		if d.Type != DiffInsert {
			texts[0] += d.Text
		}
		if d.Type != DiffDelete {
			texts[1] += d.Text
		}
	}

	return texts
}

func TestDiffCommonPrefix(t *testing.T) {
	type TestCase struct {
		Name string

		Text1 string
		Text2 string

		Expected int
	}

	dmp := New()

	for i, tc := range []TestCase{
		{"Null", "abc", "xyz", 0},
		{"Non-null", "1234abcdef", "1234xyz", 4},
		{"Whole", "1234", "1234xyz", 4},
	} {
		actual := dmp.DiffCommonPrefix(tc.Text1, tc.Text2)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}
}

func BenchmarkDiffCommonPrefix(b *testing.B) {
	s := "ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ"

	dmp := New()

	for i := 0; i < b.N; i++ {
		dmp.DiffCommonPrefix(s, s)
	}
}

func TestCommonPrefixLength(t *testing.T) {
	type TestCase struct {
		Text1 string
		Text2 string

		Expected int
	}

	for i, tc := range []TestCase{
		{"abc", "xyz", 0},
		{"1234abcdef", "1234xyz", 4},
		{"1234", "1234xyz", 4},
	} {
		actual := commonPrefixLength([]rune(tc.Text1), []rune(tc.Text2))
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}
}

func TestDiffCommonSuffix(t *testing.T) {
	type TestCase struct {
		Name string

		Text1 string
		Text2 string

		Expected int
	}

	dmp := New()

	for i, tc := range []TestCase{
		{"Null", "abc", "xyz", 0},
		{"Non-null", "abcdef1234", "xyz1234", 4},
		{"Whole", "1234", "xyz1234", 4},
	} {
		actual := dmp.DiffCommonSuffix(tc.Text1, tc.Text2)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}
}

func BenchmarkDiffCommonSuffix(b *testing.B) {
	s := "ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ"

	dmp := New()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		dmp.DiffCommonSuffix(s, s)
	}
}

func TestCommonSuffixLength(t *testing.T) {
	type TestCase struct {
		Text1 string
		Text2 string

		Expected int
	}

	for i, tc := range []TestCase{
		{"abc", "xyz", 0},
		{"abcdef1234", "xyz1234", 4},
		{"1234", "xyz1234", 4},
		{"123", "a3", 1},
	} {
		actual := commonSuffixLength([]rune(tc.Text1), []rune(tc.Text2))
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}
}

func TestDiffCommonOverlap(t *testing.T) {
	type TestCase struct {
		Name string

		Text1 string
		Text2 string

		Expected int
	}

	dmp := New()

	for i, tc := range []TestCase{
		{"Null", "", "abcd", 0},
		{"Whole", "abc", "abcd", 3},
		{"Null", "123456", "abcd", 0},
		{"Null", "123456xxx", "xxxabcd", 3},
		// Some overly clever languages (C#) may treat ligatures as equal to their component letters, e.g. U+FB01 == 'fi'
		{"Unicode", "fi", "\ufb01i", 0},
	} {
		actual := dmp.DiffCommonOverlap(tc.Text1, tc.Text2)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}
}

func TestDiffHalfMatch(t *testing.T) {
	type TestCase struct {
		Text1 string
		Text2 string

		Expected []string
	}

	dmp := New()
	dmp.DiffTimeout = 1

	for i, tc := range []TestCase{
		// No match
		{"1234567890", "abcdef", nil},
		{"12345", "23", nil},

		// Single Match
		{"1234567890", "a345678z", []string{"12", "90", "a", "z", "345678"}},
		{"a345678z", "1234567890", []string{"a", "z", "12", "90", "345678"}},
		{"abc56789z", "1234567890", []string{"abc", "z", "1234", "0", "56789"}},
		{"a23456xyz", "1234567890", []string{"a", "xyz", "1", "7890", "23456"}},

		// Multiple Matches
		{"121231234123451234123121", "a1234123451234z", []string{"12123", "123121", "a", "z", "1234123451234"}},
		{"x-=-=-=-=-=-=-=-=-=-=-=-=", "xx-=-=-=-=-=-=-=", []string{"", "-=-=-=-=-=", "x", "", "x-=-=-=-=-=-=-="}},
		{"-=-=-=-=-=-=-=-=-=-=-=-=y", "-=-=-=-=-=-=-=yy", []string{"-=-=-=-=-=", "", "", "y", "-=-=-=-=-=-=-=y"}},

		// Non-optimal halfmatch, ptimal diff would be -q+x=H-i+e=lloHe+Hu=llo-Hew+y not -qHillo+x=HelloHe-w+Hulloy
		{"qHilloHelloHew", "xHelloHeHulloy", []string{"qHillo", "w", "x", "Hulloy", "HelloHe"}},
	} {
		actual := dmp.DiffHalfMatch(tc.Text1, tc.Text2)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}

	dmp.DiffTimeout = 0

	for i, tc := range []TestCase{
		// Optimal no halfmatch
		{"qHilloHelloHew", "xHelloHeHulloy", nil},
	} {
		actual := dmp.DiffHalfMatch(tc.Text1, tc.Text2)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}
}

func BenchmarkDiffHalfMatch(b *testing.B) {
	s1, s2 := speedtestTexts()

	dmp := New()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		dmp.DiffHalfMatch(s1, s2)
	}
}

func TestDiffBisectSplit(t *testing.T) {
	type TestCase struct {
		Text1 string
		Text2 string
	}

	dmp := New()

	for _, tc := range []TestCase{
		{"STUV\x05WX\x05YZ\x05[", "WĺĻļ\x05YZ\x05ĽľĿŀZ"},
	} {
		diffs := dmp.diffBisectSplit([]rune(tc.Text1),
			[]rune(tc.Text2), 7, 6, time.Now().Add(time.Hour))

		for _, d := range diffs {
			assert.True(t, utf8.ValidString(d.Text))
		}

		// TODO define the expected outcome
	}
}

func TestDiffLinesToChars(t *testing.T) {
	type TestCase struct {
		Text1 string
		Text2 string

		ExpectedChars1 string
		ExpectedChars2 string
		ExpectedLines  []string
	}

	dmp := New()

	for i, tc := range []TestCase{
		{"", "alpha\r\nbeta\r\n\r\n\r\n", "", "\u0001\u0002\u0003\u0003", []string{"", "alpha\r\n", "beta\r\n", "\r\n"}},
		{"a", "b", "\u0001", "\u0002", []string{"", "a", "b"}},
		// Omit final newline.
		{"alpha\nbeta\nalpha", "", "\u0001\u0002\u0003", "", []string{"", "alpha\n", "beta\n", "alpha"}},
	} {
		actualChars1, actualChars2, actualLines := dmp.DiffLinesToChars(tc.Text1, tc.Text2)
		assert.Equal(t, tc.ExpectedChars1, actualChars1, fmt.Sprintf("Test case #%d, %#v", i, tc))
		assert.Equal(t, tc.ExpectedChars2, actualChars2, fmt.Sprintf("Test case #%d, %#v", i, tc))
		assert.Equal(t, tc.ExpectedLines, actualLines, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}

	// More than 256 to reveal any 8-bit limitations.
	n := 300
	lineList := []string{
		"", // Account for the initial empty element of the lines array.
	}
	var charList []rune
	for x := 1; x < n+1; x++ {
		lineList = append(lineList, strconv.Itoa(x)+"\n")
		charList = append(charList, rune(x))
	}
	lines := strings.Join(lineList, "")
	chars := string(charList)
	assert.Equal(t, n, utf8.RuneCountInString(chars))

	actualChars1, actualChars2, actualLines := dmp.DiffLinesToChars(lines, "")
	assert.Equal(t, chars, actualChars1)
	assert.Equal(t, "", actualChars2)
	assert.Equal(t, lineList, actualLines)
}

func TestDiffCharsToLines(t *testing.T) {
	type TestCase struct {
		Diffs []Diff
		Lines []string

		Expected []Diff
	}

	dmp := New()

	for i, tc := range []TestCase{
		{
			Diffs: []Diff{
				{DiffEqual, "\u0001\u0002\u0001"},
				{DiffInsert, "\u0002\u0001\u0002"},
			},
			Lines: []string{"", "alpha\n", "beta\n"},

			Expected: []Diff{
				{DiffEqual, "alpha\nbeta\nalpha\n"},
				{DiffInsert, "beta\nalpha\nbeta\n"},
			},
		},
	} {
		actual := dmp.DiffCharsToLines(tc.Diffs, tc.Lines)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}

	// More than 256 to reveal any 8-bit limitations.
	n := 300
	lineList := []string{
		"", // Account for the initial empty element of the lines array.
	}
	charList := []rune{}
	for x := 1; x <= n; x++ {
		lineList = append(lineList, strconv.Itoa(x)+"\n")
		charList = append(charList, rune(x))
	}
	assert.Equal(t, n, len(charList))

	actual := dmp.DiffCharsToLines([]Diff{Diff{DiffDelete, string(charList)}}, lineList)
	assert.Equal(t, []Diff{Diff{DiffDelete, strings.Join(lineList, "")}}, actual)
}

func TestDiffCleanupMerge(t *testing.T) {
	type TestCase struct {
		Name string

		Diffs []Diff

		Expected []Diff
	}

	dmp := New()

	for i, tc := range []TestCase{
		{
			"Null case",
			[]Diff{},
			[]Diff{},
		},
		{
			"No Diff case",
			[]Diff{Diff{DiffEqual, "a"}, Diff{DiffDelete, "b"}, Diff{DiffInsert, "c"}},
			[]Diff{Diff{DiffEqual, "a"}, Diff{DiffDelete, "b"}, Diff{DiffInsert, "c"}},
		},
		{
			"Merge equalities",
			[]Diff{Diff{DiffEqual, "a"}, Diff{DiffEqual, "b"}, Diff{DiffEqual, "c"}},
			[]Diff{Diff{DiffEqual, "abc"}},
		},
		{
			"Merge deletions",
			[]Diff{Diff{DiffDelete, "a"}, Diff{DiffDelete, "b"}, Diff{DiffDelete, "c"}},
			[]Diff{Diff{DiffDelete, "abc"}},
		},
		{
			"Merge insertions",
			[]Diff{Diff{DiffInsert, "a"}, Diff{DiffInsert, "b"}, Diff{DiffInsert, "c"}},
			[]Diff{Diff{DiffInsert, "abc"}},
		},
		{
			"Merge interweave",
			[]Diff{Diff{DiffDelete, "a"}, Diff{DiffInsert, "b"}, Diff{DiffDelete, "c"}, Diff{DiffInsert, "d"}, Diff{DiffEqual, "e"}, Diff{DiffEqual, "f"}},
			[]Diff{Diff{DiffDelete, "ac"}, Diff{DiffInsert, "bd"}, Diff{DiffEqual, "ef"}},
		},
		{
			"Prefix and suffix detection",
			[]Diff{Diff{DiffDelete, "a"}, Diff{DiffInsert, "abc"}, Diff{DiffDelete, "dc"}},
			[]Diff{Diff{DiffEqual, "a"}, Diff{DiffDelete, "d"}, Diff{DiffInsert, "b"}, Diff{DiffEqual, "c"}},
		},
		{
			"Prefix and suffix detection with equalities",
			[]Diff{Diff{DiffEqual, "x"}, Diff{DiffDelete, "a"}, Diff{DiffInsert, "abc"}, Diff{DiffDelete, "dc"}, Diff{DiffEqual, "y"}},
			[]Diff{Diff{DiffEqual, "xa"}, Diff{DiffDelete, "d"}, Diff{DiffInsert, "b"}, Diff{DiffEqual, "cy"}},
		},
		{
			"Same test as above but with unicode (\u0101 will appear in diffs with at least 257 unique lines)",
			[]Diff{Diff{DiffEqual, "x"}, Diff{DiffDelete, "\u0101"}, Diff{DiffInsert, "\u0101bc"}, Diff{DiffDelete, "dc"}, Diff{DiffEqual, "y"}},
			[]Diff{Diff{DiffEqual, "x\u0101"}, Diff{DiffDelete, "d"}, Diff{DiffInsert, "b"}, Diff{DiffEqual, "cy"}},
		},
		{
			"Slide edit left",
			[]Diff{Diff{DiffEqual, "a"}, Diff{DiffInsert, "ba"}, Diff{DiffEqual, "c"}},
			[]Diff{Diff{DiffInsert, "ab"}, Diff{DiffEqual, "ac"}},
		},
		{
			"Slide edit right",
			[]Diff{Diff{DiffEqual, "c"}, Diff{DiffInsert, "ab"}, Diff{DiffEqual, "a"}},
			[]Diff{Diff{DiffEqual, "ca"}, Diff{DiffInsert, "ba"}},
		},
		{
			"Slide edit left recursive",
			[]Diff{Diff{DiffEqual, "a"}, Diff{DiffDelete, "b"}, Diff{DiffEqual, "c"}, Diff{DiffDelete, "ac"}, Diff{DiffEqual, "x"}},
			[]Diff{Diff{DiffDelete, "abc"}, Diff{DiffEqual, "acx"}},
		},
		{
			"Slide edit right recursive",
			[]Diff{Diff{DiffEqual, "x"}, Diff{DiffDelete, "ca"}, Diff{DiffEqual, "c"}, Diff{DiffDelete, "b"}, Diff{DiffEqual, "a"}},
			[]Diff{Diff{DiffEqual, "xca"}, Diff{DiffDelete, "cba"}},
		},
	} {
		actual := dmp.DiffCleanupMerge(tc.Diffs)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}
}

func TestDiffCleanupSemanticLossless(t *testing.T) {
	type TestCase struct {
		Name string

		Diffs []Diff

		Expected []Diff
	}

	dmp := New()

	for i, tc := range []TestCase{
		{
			"Null case",
			[]Diff{},
			[]Diff{},
		},
		{
			"Blank lines",
			[]Diff{
				Diff{DiffEqual, "AAA\r\n\r\nBBB"},
				Diff{DiffInsert, "\r\nDDD\r\n\r\nBBB"},
				Diff{DiffEqual, "\r\nEEE"},
			},
			[]Diff{
				Diff{DiffEqual, "AAA\r\n\r\n"},
				Diff{DiffInsert, "BBB\r\nDDD\r\n\r\n"},
				Diff{DiffEqual, "BBB\r\nEEE"},
			},
		},
		{
			"Line boundaries",
			[]Diff{
				Diff{DiffEqual, "AAA\r\nBBB"},
				Diff{DiffInsert, " DDD\r\nBBB"},
				Diff{DiffEqual, " EEE"},
			},
			[]Diff{
				Diff{DiffEqual, "AAA\r\n"},
				Diff{DiffInsert, "BBB DDD\r\n"},
				Diff{DiffEqual, "BBB EEE"},
			},
		},
		{
			"Word boundaries",
			[]Diff{
				Diff{DiffEqual, "The c"},
				Diff{DiffInsert, "ow and the c"},
				Diff{DiffEqual, "at."},
			},
			[]Diff{
				Diff{DiffEqual, "The "},
				Diff{DiffInsert, "cow and the "},
				Diff{DiffEqual, "cat."},
			},
		},
		{
			"Alphanumeric boundaries",
			[]Diff{
				Diff{DiffEqual, "The-c"},
				Diff{DiffInsert, "ow-and-the-c"},
				Diff{DiffEqual, "at."},
			},
			[]Diff{
				Diff{DiffEqual, "The-"},
				Diff{DiffInsert, "cow-and-the-"},
				Diff{DiffEqual, "cat."},
			},
		},
		{
			"Hitting the start",
			[]Diff{
				Diff{DiffEqual, "a"},
				Diff{DiffDelete, "a"},
				Diff{DiffEqual, "ax"},
			},
			[]Diff{
				Diff{DiffDelete, "a"},
				Diff{DiffEqual, "aax"},
			},
		},
		{
			"Hitting the end",
			[]Diff{
				Diff{DiffEqual, "xa"},
				Diff{DiffDelete, "a"},
				Diff{DiffEqual, "a"},
			},
			[]Diff{
				Diff{DiffEqual, "xaa"},
				Diff{DiffDelete, "a"},
			},
		},
		{
			"Sentence boundaries",
			[]Diff{
				Diff{DiffEqual, "The xxx. The "},
				Diff{DiffInsert, "zzz. The "},
				Diff{DiffEqual, "yyy."},
			},
			[]Diff{
				Diff{DiffEqual, "The xxx."},
				Diff{DiffInsert, " The zzz."},
				Diff{DiffEqual, " The yyy."},
			},
		},
		{
			"UTF-8 strings",
			[]Diff{
				Diff{DiffEqual, "The ♕. The "},
				Diff{DiffInsert, "♔. The "},
				Diff{DiffEqual, "♖."},
			},
			[]Diff{
				Diff{DiffEqual, "The ♕."},
				Diff{DiffInsert, " The ♔."},
				Diff{DiffEqual, " The ♖."},
			},
		},
		{
			"Rune boundaries",
			[]Diff{
				Diff{DiffEqual, "♕♕"},
				Diff{DiffInsert, "♔♔"},
				Diff{DiffEqual, "♖♖"},
			},
			[]Diff{
				Diff{DiffEqual, "♕♕"},
				Diff{DiffInsert, "♔♔"},
				Diff{DiffEqual, "♖♖"},
			},
		},
	} {
		actual := dmp.DiffCleanupSemanticLossless(tc.Diffs)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}
}

func TestDiffCleanupSemantic(t *testing.T) {
	type TestCase struct {
		Name string

		Diffs []Diff

		Expected []Diff
	}

	dmp := New()

	for i, tc := range []TestCase{
		{
			"Null case",
			[]Diff{},
			[]Diff{},
		},
		{
			"No elimination #1",
			[]Diff{
				{DiffDelete, "ab"},
				{DiffInsert, "cd"},
				{DiffEqual, "12"},
				{DiffDelete, "e"},
			},
			[]Diff{
				{DiffDelete, "ab"},
				{DiffInsert, "cd"},
				{DiffEqual, "12"},
				{DiffDelete, "e"},
			},
		},
		{
			"No elimination #2",
			[]Diff{
				{DiffDelete, "abc"},
				{DiffInsert, "ABC"},
				{DiffEqual, "1234"},
				{DiffDelete, "wxyz"},
			},
			[]Diff{
				{DiffDelete, "abc"},
				{DiffInsert, "ABC"},
				{DiffEqual, "1234"},
				{DiffDelete, "wxyz"},
			},
		},
		{
			"No elimination #3",
			[]Diff{
				{DiffEqual, "2016-09-01T03:07:1"},
				{DiffInsert, "5.15"},
				{DiffEqual, "4"},
				{DiffDelete, "."},
				{DiffEqual, "80"},
				{DiffInsert, "0"},
				{DiffEqual, "78"},
				{DiffDelete, "3074"},
				{DiffEqual, "1Z"},
			},
			[]Diff{
				{DiffEqual, "2016-09-01T03:07:1"},
				{DiffInsert, "5.15"},
				{DiffEqual, "4"},
				{DiffDelete, "."},
				{DiffEqual, "80"},
				{DiffInsert, "0"},
				{DiffEqual, "78"},
				{DiffDelete, "3074"},
				{DiffEqual, "1Z"},
			},
		},
		{
			"Simple elimination",
			[]Diff{
				{DiffDelete, "a"},
				{DiffEqual, "b"},
				{DiffDelete, "c"},
			},
			[]Diff{
				{DiffDelete, "abc"},
				{DiffInsert, "b"},
			},
		},
		{
			"Backpass elimination",
			[]Diff{
				{DiffDelete, "ab"},
				{DiffEqual, "cd"},
				{DiffDelete, "e"},
				{DiffEqual, "f"},
				{DiffInsert, "g"},
			},
			[]Diff{
				{DiffDelete, "abcdef"},
				{DiffInsert, "cdfg"},
			},
		},
		{
			"Multiple eliminations",
			[]Diff{
				{DiffInsert, "1"},
				{DiffEqual, "A"},
				{DiffDelete, "B"},
				{DiffInsert, "2"},
				{DiffEqual, "_"},
				{DiffInsert, "1"},
				{DiffEqual, "A"},
				{DiffDelete, "B"},
				{DiffInsert, "2"},
			},
			[]Diff{
				{DiffDelete, "AB_AB"},
				{DiffInsert, "1A2_1A2"},
			},
		},
		{
			"Word boundaries",
			[]Diff{
				{DiffEqual, "The c"},
				{DiffDelete, "ow and the c"},
				{DiffEqual, "at."},
			},
			[]Diff{
				{DiffEqual, "The "},
				{DiffDelete, "cow and the "},
				{DiffEqual, "cat."},
			},
		},
		{
			"No overlap elimination",
			[]Diff{
				{DiffDelete, "abcxx"},
				{DiffInsert, "xxdef"},
			},
			[]Diff{
				{DiffDelete, "abcxx"},
				{DiffInsert, "xxdef"},
			},
		},
		{
			"Overlap elimination",
			[]Diff{
				{DiffDelete, "abcxxx"},
				{DiffInsert, "xxxdef"},
			},
			[]Diff{
				{DiffDelete, "abc"},
				{DiffEqual, "xxx"},
				{DiffInsert, "def"},
			},
		},
		{
			"Reverse overlap elimination",
			[]Diff{
				{DiffDelete, "xxxabc"},
				{DiffInsert, "defxxx"},
			},
			[]Diff{
				{DiffInsert, "def"},
				{DiffEqual, "xxx"},
				{DiffDelete, "abc"},
			},
		},
		{
			"Two overlap eliminations",
			[]Diff{
				{DiffDelete, "abcd1212"},
				{DiffInsert, "1212efghi"},
				{DiffEqual, "----"},
				{DiffDelete, "A3"},
				{DiffInsert, "3BC"},
			},
			[]Diff{
				{DiffDelete, "abcd"},
				{DiffEqual, "1212"},
				{DiffInsert, "efghi"},
				{DiffEqual, "----"},
				{DiffDelete, "A"},
				{DiffEqual, "3"},
				{DiffInsert, "BC"},
			},
		},
		{
			"Test case for adapting DiffCleanupSemantic to be equal to the Python version #19",
			[]Diff{
				{DiffEqual, "James McCarthy "},
				{DiffDelete, "close to "},
				{DiffEqual, "sign"},
				{DiffDelete, "ing"},
				{DiffInsert, "s"},
				{DiffEqual, " new "},
				{DiffDelete, "E"},
				{DiffInsert, "fi"},
				{DiffEqual, "ve"},
				{DiffInsert, "-yea"},
				{DiffEqual, "r"},
				{DiffDelete, "ton"},
				{DiffEqual, " deal"},
				{DiffInsert, " at Everton"},
			},
			[]Diff{
				{DiffEqual, "James McCarthy "},
				{DiffDelete, "close to "},
				{DiffEqual, "sign"},
				{DiffDelete, "ing"},
				{DiffInsert, "s"},
				{DiffEqual, " new "},
				{DiffInsert, "five-year deal at "},
				{DiffEqual, "Everton"},
				{DiffDelete, " deal"},
			},
		},
	} {
		actual := dmp.DiffCleanupSemantic(tc.Diffs)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}
}

func BenchmarkDiffCleanupSemantic(b *testing.B) {
	s1, s2 := speedtestTexts()

	dmp := New()

	diffs := dmp.DiffMain(s1, s2, false)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		dmp.DiffCleanupSemantic(diffs)
	}
}

func TestDiffCleanupEfficiency(t *testing.T) {
	type TestCase struct {
		Name string

		Diffs []Diff

		Expected []Diff
	}

	dmp := New()
	dmp.DiffEditCost = 4

	for i, tc := range []TestCase{
		{
			"Null case",
			[]Diff{},
			[]Diff{},
		},
		{
			"No elimination",
			[]Diff{
				Diff{DiffDelete, "ab"},
				Diff{DiffInsert, "12"},
				Diff{DiffEqual, "wxyz"},
				Diff{DiffDelete, "cd"},
				Diff{DiffInsert, "34"},
			},
			[]Diff{
				Diff{DiffDelete, "ab"},
				Diff{DiffInsert, "12"},
				Diff{DiffEqual, "wxyz"},
				Diff{DiffDelete, "cd"},
				Diff{DiffInsert, "34"},
			},
		},
		{
			"Four-edit elimination",
			[]Diff{
				Diff{DiffDelete, "ab"},
				Diff{DiffInsert, "12"},
				Diff{DiffEqual, "xyz"},
				Diff{DiffDelete, "cd"},
				Diff{DiffInsert, "34"},
			},
			[]Diff{
				Diff{DiffDelete, "abxyzcd"},
				Diff{DiffInsert, "12xyz34"},
			},
		},
		{
			"Three-edit elimination",
			[]Diff{
				Diff{DiffInsert, "12"},
				Diff{DiffEqual, "x"},
				Diff{DiffDelete, "cd"},
				Diff{DiffInsert, "34"},
			},
			[]Diff{
				Diff{DiffDelete, "xcd"},
				Diff{DiffInsert, "12x34"},
			},
		},
		{
			"Backpass elimination",
			[]Diff{
				Diff{DiffDelete, "ab"},
				Diff{DiffInsert, "12"},
				Diff{DiffEqual, "xy"},
				Diff{DiffInsert, "34"},
				Diff{DiffEqual, "z"},
				Diff{DiffDelete, "cd"},
				Diff{DiffInsert, "56"},
			},
			[]Diff{
				Diff{DiffDelete, "abxyzcd"},
				Diff{DiffInsert, "12xy34z56"},
			},
		},
	} {
		actual := dmp.DiffCleanupEfficiency(tc.Diffs)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}

	dmp.DiffEditCost = 5

	for i, tc := range []TestCase{
		{
			"High cost elimination",
			[]Diff{
				Diff{DiffDelete, "ab"},
				Diff{DiffInsert, "12"},
				Diff{DiffEqual, "wxyz"},
				Diff{DiffDelete, "cd"},
				Diff{DiffInsert, "34"},
			},
			[]Diff{
				Diff{DiffDelete, "abwxyzcd"},
				Diff{DiffInsert, "12wxyz34"},
			},
		},
	} {
		actual := dmp.DiffCleanupEfficiency(tc.Diffs)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}
}

func TestDiffPrettyHtml(t *testing.T) {
	type TestCase struct {
		Diffs []Diff

		Expected string
	}

	dmp := New()

	for i, tc := range []TestCase{
		{
			Diffs: []Diff{
				{DiffEqual, "a\n"},
				{DiffDelete, "<B>b</B>"},
				{DiffInsert, "c&d"},
			},

			Expected: "<span>a&para;<br></span><del style=\"background:#ffe6e6;\">&lt;B&gt;b&lt;/B&gt;</del><ins style=\"background:#e6ffe6;\">c&amp;d</ins>",
		},
	} {
		actual := dmp.DiffPrettyHtml(tc.Diffs)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}
}

func TestDiffPrettyText(t *testing.T) {
	type TestCase struct {
		Diffs []Diff

		Expected string
	}

	dmp := New()

	for i, tc := range []TestCase{
		{
			Diffs: []Diff{
				{DiffEqual, "a\n"},
				{DiffDelete, "<B>b</B>"},
				{DiffInsert, "c&d"},
			},

			Expected: "a\n\x1b[31m<B>b</B>\x1b[0m\x1b[32mc&d\x1b[0m",
		},
	} {
		actual := dmp.DiffPrettyText(tc.Diffs)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}
}

func TestDiffText(t *testing.T) {
	type TestCase struct {
		Diffs []Diff

		ExpectedText1 string
		ExpectedText2 string
	}

	dmp := New()

	for i, tc := range []TestCase{
		{
			Diffs: []Diff{
				{DiffEqual, "jump"},
				{DiffDelete, "s"},
				{DiffInsert, "ed"},
				{DiffEqual, " over "},
				{DiffDelete, "the"},
				{DiffInsert, "a"},
				{DiffEqual, " lazy"},
			},

			ExpectedText1: "jumps over the lazy",
			ExpectedText2: "jumped over a lazy",
		},
	} {
		actualText1 := dmp.DiffText1(tc.Diffs)
		assert.Equal(t, tc.ExpectedText1, actualText1, fmt.Sprintf("Test case #%d, %#v", i, tc))

		actualText2 := dmp.DiffText2(tc.Diffs)
		assert.Equal(t, tc.ExpectedText2, actualText2, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}
}

func TestDiffDelta(t *testing.T) {
	type TestCase struct {
		Name string

		Text  string
		Delta string

		ErrorMessagePrefix string
	}

	dmp := New()

	for i, tc := range []TestCase{
		{"Delta shorter than text", "jumps over the lazyx", "=4\t-1\t+ed\t=6\t-3\t+a\t=5\t+old dog", "Delta length (19) is different from source text length (20)"},
		{"Delta longer than text", "umps over the lazy", "=4\t-1\t+ed\t=6\t-3\t+a\t=5\t+old dog", "Delta length (19) is different from source text length (18)"},
		{"Invalid URL escaping", "", "+%c3%xy", "invalid URL escape \"%xy\""},
		{"Invalid UTF-8 sequence", "", "+%c3xy", "invalid UTF-8 token: \"\\xc3xy\""},
		{"Invalid diff operation", "", "a", "Invalid diff operation in DiffFromDelta: a"},
		{"Invalid diff syntax", "", "-", "strconv.ParseInt: parsing \"\": invalid syntax"},
		{"Negative number in delta", "", "--1", "Negative number in DiffFromDelta: -1"},
		{"Empty case", "", "", ""},
	} {
		diffs, err := dmp.DiffFromDelta(tc.Text, tc.Delta)
		msg := fmt.Sprintf("Test case #%d, %s", i, tc.Name)
		if tc.ErrorMessagePrefix == "" {
			assert.Nil(t, err, msg)
			assert.Nil(t, diffs, msg)
		} else {
			e := err.Error()
			if strings.HasPrefix(e, tc.ErrorMessagePrefix) {
				e = tc.ErrorMessagePrefix
			}
			assert.Nil(t, diffs, msg)
			assert.Equal(t, tc.ErrorMessagePrefix, e, msg)
		}
	}

	// Convert a diff into delta string.
	diffs := []Diff{
		Diff{DiffEqual, "jump"},
		Diff{DiffDelete, "s"},
		Diff{DiffInsert, "ed"},
		Diff{DiffEqual, " over "},
		Diff{DiffDelete, "the"},
		Diff{DiffInsert, "a"},
		Diff{DiffEqual, " lazy"},
		Diff{DiffInsert, "old dog"},
	}
	text1 := dmp.DiffText1(diffs)
	assert.Equal(t, "jumps over the lazy", text1)

	delta := dmp.DiffToDelta(diffs)
	assert.Equal(t, "=4\t-1\t+ed\t=6\t-3\t+a\t=5\t+old dog", delta)

	// Convert delta string into a diff.
	deltaDiffs, err := dmp.DiffFromDelta(text1, delta)
	assert.Equal(t, diffs, deltaDiffs)

	// Test deltas with special characters.
	diffs = []Diff{
		Diff{DiffEqual, "\u0680 \x00 \t %"},
		Diff{DiffDelete, "\u0681 \x01 \n ^"},
		Diff{DiffInsert, "\u0682 \x02 \\ |"},
	}
	text1 = dmp.DiffText1(diffs)
	assert.Equal(t, "\u0680 \x00 \t %\u0681 \x01 \n ^", text1)

	// Lowercase, due to UrlEncode uses lower.
	delta = dmp.DiffToDelta(diffs)
	assert.Equal(t, "=7\t-7\t+%DA%82 %02 %5C %7C", delta)

	deltaDiffs, err = dmp.DiffFromDelta(text1, delta)
	assert.Equal(t, diffs, deltaDiffs)
	assert.Nil(t, err)

	// Verify pool of unchanged characters.
	diffs = []Diff{
		Diff{DiffInsert, "A-Z a-z 0-9 - _ . ! ~ * ' ( ) ; / ? : @ & = + $ , # "},
	}

	delta = dmp.DiffToDelta(diffs)
	assert.Equal(t, "+A-Z a-z 0-9 - _ . ! ~ * ' ( ) ; / ? : @ & = + $ , # ", delta, "Unchanged characters.")

	// Convert delta string into a diff.
	deltaDiffs, err = dmp.DiffFromDelta("", delta)
	assert.Equal(t, diffs, deltaDiffs)
	assert.Nil(t, err)
}

func TestDiffXIndex(t *testing.T) {
	type TestCase struct {
		Name string

		Diffs    []Diff
		Location int

		Expected int
	}

	dmp := New()

	for i, tc := range []TestCase{
		{"Translation on equality", []Diff{{DiffDelete, "a"}, {DiffInsert, "1234"}, {DiffEqual, "xyz"}}, 2, 5},
		{"Translation on deletion", []Diff{{DiffEqual, "a"}, {DiffDelete, "1234"}, {DiffEqual, "xyz"}}, 3, 1},
	} {
		actual := dmp.DiffXIndex(tc.Diffs, tc.Location)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}
}

func TestDiffLevenshtein(t *testing.T) {
	type TestCase struct {
		Name string

		Diffs []Diff

		Expected int
	}

	dmp := New()

	for i, tc := range []TestCase{
		{"Levenshtein with trailing equality", []Diff{{DiffDelete, "abc"}, {DiffInsert, "1234"}, {DiffEqual, "xyz"}}, 4},
		{"Levenshtein with leading equality", []Diff{{DiffEqual, "xyz"}, {DiffDelete, "abc"}, {DiffInsert, "1234"}}, 4},
		{"Levenshtein with middle equality", []Diff{{DiffDelete, "abc"}, {DiffEqual, "xyz"}, {DiffInsert, "1234"}}, 7},
	} {
		actual := dmp.DiffLevenshtein(tc.Diffs)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}
}

func TestDiffBisect(t *testing.T) {
	type TestCase struct {
		Name string

		Time time.Time

		Expected []Diff
	}

	dmp := New()

	for i, tc := range []TestCase{
		{
			Name: "normal",
			Time: time.Date(9999, time.December, 31, 23, 59, 59, 59, time.UTC),

			Expected: []Diff{
				{DiffDelete, "c"},
				{DiffInsert, "m"},
				{DiffEqual, "a"},
				{DiffDelete, "t"},
				{DiffInsert, "p"},
			},
		},
		{
			Name: "Negative deadlines count as having infinite time",
			Time: time.Date(0001, time.January, 01, 00, 00, 00, 00, time.UTC),

			Expected: []Diff{
				{DiffDelete, "c"},
				{DiffInsert, "m"},
				{DiffEqual, "a"},
				{DiffDelete, "t"},
				{DiffInsert, "p"},
			},
		},
		{
			Name: "Timeout",
			Time: time.Now().Add(time.Nanosecond),

			Expected: []Diff{
				{DiffDelete, "cat"},
				{DiffInsert, "map"},
			},
		},
	} {
		actual := dmp.DiffBisect("cat", "map", tc.Time)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}

	// Test for invalid UTF-8 sequences
	assert.Equal(t, []Diff{
		Diff{DiffEqual, "��"},
	}, dmp.DiffBisect("\xe0\xe5", "\xe0\xe5", time.Now().Add(time.Minute)))
}

func TestDiffMain(t *testing.T) {
	type TestCase struct {
		Text1 string
		Text2 string

		Expected []Diff
	}

	dmp := New()

	// Perform a trivial diff.
	for i, tc := range []TestCase{
		{
			"",
			"",
			nil,
		},
		{
			"abc",
			"abc",
			[]Diff{Diff{DiffEqual, "abc"}},
		},
		{
			"abc",
			"ab123c",
			[]Diff{Diff{DiffEqual, "ab"}, Diff{DiffInsert, "123"}, Diff{DiffEqual, "c"}},
		},
		{
			"a123bc",
			"abc",
			[]Diff{Diff{DiffEqual, "a"}, Diff{DiffDelete, "123"}, Diff{DiffEqual, "bc"}},
		},
		{
			"abc",
			"a123b456c",
			[]Diff{Diff{DiffEqual, "a"}, Diff{DiffInsert, "123"}, Diff{DiffEqual, "b"}, Diff{DiffInsert, "456"}, Diff{DiffEqual, "c"}},
		},
		{
			"a123b456c",
			"abc",
			[]Diff{Diff{DiffEqual, "a"}, Diff{DiffDelete, "123"}, Diff{DiffEqual, "b"}, Diff{DiffDelete, "456"}, Diff{DiffEqual, "c"}},
		},
	} {
		actual := dmp.DiffMain(tc.Text1, tc.Text2, false)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}

	// Perform a real diff and switch off the timeout.
	dmp.DiffTimeout = 0

	for i, tc := range []TestCase{
		{
			"a",
			"b",
			[]Diff{Diff{DiffDelete, "a"}, Diff{DiffInsert, "b"}},
		},
		{
			"Apples are a fruit.",
			"Bananas are also fruit.",
			[]Diff{
				Diff{DiffDelete, "Apple"},
				Diff{DiffInsert, "Banana"},
				Diff{DiffEqual, "s are a"},
				Diff{DiffInsert, "lso"},
				Diff{DiffEqual, " fruit."},
			},
		},
		{
			"ax\t",
			"\u0680x\u0000",
			[]Diff{
				Diff{DiffDelete, "a"},
				Diff{DiffInsert, "\u0680"},
				Diff{DiffEqual, "x"},
				Diff{DiffDelete, "\t"},
				Diff{DiffInsert, "\u0000"},
			},
		},
		{
			"1ayb2",
			"abxab",
			[]Diff{
				Diff{DiffDelete, "1"},
				Diff{DiffEqual, "a"},
				Diff{DiffDelete, "y"},
				Diff{DiffEqual, "b"},
				Diff{DiffDelete, "2"},
				Diff{DiffInsert, "xab"},
			},
		},
		{
			"abcy",
			"xaxcxabc",
			[]Diff{
				Diff{DiffInsert, "xaxcx"},
				Diff{DiffEqual, "abc"}, Diff{DiffDelete, "y"},
			},
		},
		{
			"ABCDa=bcd=efghijklmnopqrsEFGHIJKLMNOefg",
			"a-bcd-efghijklmnopqrs",
			[]Diff{
				Diff{DiffDelete, "ABCD"},
				Diff{DiffEqual, "a"},
				Diff{DiffDelete, "="},
				Diff{DiffInsert, "-"},
				Diff{DiffEqual, "bcd"},
				Diff{DiffDelete, "="},
				Diff{DiffInsert, "-"},
				Diff{DiffEqual, "efghijklmnopqrs"},
				Diff{DiffDelete, "EFGHIJKLMNOefg"},
			},
		},
		{
			"a [[Pennsylvania]] and [[New",
			" and [[Pennsylvania]]",
			[]Diff{
				Diff{DiffInsert, " "},
				Diff{DiffEqual, "a"},
				Diff{DiffInsert, "nd"},
				Diff{DiffEqual, " [[Pennsylvania]]"},
				Diff{DiffDelete, " and [[New"},
			},
		},
	} {
		actual := dmp.DiffMain(tc.Text1, tc.Text2, false)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}

	// Test for invalid UTF-8 sequences
	assert.Equal(t, []Diff{
		Diff{DiffDelete, "��"},
	}, dmp.DiffMain("\xe0\xe5", "", false))
}

func TestDiffMainWithTimeout(t *testing.T) {
	dmp := New()
	dmp.DiffTimeout = 200 * time.Millisecond

	a := "`Twas brillig, and the slithy toves\nDid gyre and gimble in the wabe:\nAll mimsy were the borogoves,\nAnd the mome raths outgrabe.\n"
	b := "I am the very model of a modern major general,\nI've information vegetable, animal, and mineral,\nI know the kings of England, and I quote the fights historical,\nFrom Marathon to Waterloo, in order categorical.\n"
	// Increase the text lengths by 1024 times to ensure a timeout.
	for x := 0; x < 13; x++ {
		a = a + a
		b = b + b
	}

	startTime := time.Now()
	dmp.DiffMain(a, b, true)
	endTime := time.Now()

	delta := endTime.Sub(startTime)

	// Test that we took at least the timeout period.
	assert.True(t, delta >= dmp.DiffTimeout, fmt.Sprintf("%v !>= %v", delta, dmp.DiffTimeout))

	// Test that we didn't take forever (be very forgiving). Theoretically this test could fail very occasionally if the OS task swaps or locks up for a second at the wrong moment.
	assert.True(t, delta < (dmp.DiffTimeout*100), fmt.Sprintf("%v !< %v", delta, dmp.DiffTimeout*100))
}

func TestDiffMainWithCheckLines(t *testing.T) {
	type TestCase struct {
		Text1 string
		Text2 string
	}

	dmp := New()
	dmp.DiffTimeout = 0

	// Test cases must be at least 100 chars long to pass the cutoff.
	for i, tc := range []TestCase{
		{
			"1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n",
			"abcdefghij\nabcdefghij\nabcdefghij\nabcdefghij\nabcdefghij\nabcdefghij\nabcdefghij\nabcdefghij\nabcdefghij\nabcdefghij\nabcdefghij\nabcdefghij\nabcdefghij\n",
		},
		{
			"1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890",
			"abcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghij",
		},
		{
			"1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n1234567890\n",
			"abcdefghij\n1234567890\n1234567890\n1234567890\nabcdefghij\n1234567890\n1234567890\n1234567890\nabcdefghij\n1234567890\n1234567890\n1234567890\nabcdefghij\n",
		},
	} {
		resultWithoutCheckLines := dmp.DiffMain(tc.Text1, tc.Text2, false)
		resultWithCheckLines := dmp.DiffMain(tc.Text1, tc.Text2, true)

		// TODO this fails for the third test case, why?
		if i != 2 {
			assert.Equal(t, resultWithoutCheckLines, resultWithCheckLines, fmt.Sprintf("Test case #%d, %#v", i, tc))
		}
		assert.Equal(t, diffRebuildTexts(resultWithoutCheckLines), diffRebuildTexts(resultWithCheckLines), fmt.Sprintf("Test case #%d, %#v", i, tc))
	}
}

func BenchmarkDiffMain(bench *testing.B) {
	s1 := "`Twas brillig, and the slithy toves\nDid gyre and gimble in the wabe:\nAll mimsy were the borogoves,\nAnd the mome raths outgrabe.\n"
	s2 := "I am the very model of a modern major general,\nI've information vegetable, animal, and mineral,\nI know the kings of England, and I quote the fights historical,\nFrom Marathon to Waterloo, in order categorical.\n"

	// Increase the text lengths by 1024 times to ensure a timeout.
	for x := 0; x < 10; x++ {
		s1 = s1 + s1
		s2 = s2 + s2
	}

	dmp := New()
	dmp.DiffTimeout = time.Second

	bench.ResetTimer()

	for i := 0; i < bench.N; i++ {
		dmp.DiffMain(s1, s2, true)
	}
}

func BenchmarkDiffMainLarge(b *testing.B) {
	s1, s2 := speedtestTexts()

	dmp := New()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		dmp.DiffMain(s1, s2, true)
	}
}

func BenchmarkDiffMainRunesLargeLines(b *testing.B) {
	s1, s2 := speedtestTexts()

	dmp := New()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		text1, text2, linearray := dmp.DiffLinesToRunes(s1, s2)

		diffs := dmp.DiffMainRunes(text1, text2, false)
		diffs = dmp.DiffCharsToLines(diffs, linearray)
	}
}
