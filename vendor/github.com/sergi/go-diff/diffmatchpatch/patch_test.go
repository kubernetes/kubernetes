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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPatchString(t *testing.T) {
	type TestCase struct {
		Patch Patch

		Expected string
	}

	for i, tc := range []TestCase{
		{
			Patch: Patch{
				Start1:  20,
				Start2:  21,
				Length1: 18,
				Length2: 17,

				diffs: []Diff{
					{DiffEqual, "jump"},
					{DiffDelete, "s"},
					{DiffInsert, "ed"},
					{DiffEqual, " over "},
					{DiffDelete, "the"},
					{DiffInsert, "a"},
					{DiffEqual, "\nlaz"},
				},
			},

			Expected: "@@ -21,18 +22,17 @@\n jump\n-s\n+ed\n  over \n-the\n+a\n %0Alaz\n",
		},
	} {
		actual := tc.Patch.String()
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}
}

func TestPatchFromText(t *testing.T) {
	type TestCase struct {
		Patch string

		ErrorMessagePrefix string
	}

	dmp := New()

	for i, tc := range []TestCase{
		{"", ""},
		{"@@ -21,18 +22,17 @@\n jump\n-s\n+ed\n  over \n-the\n+a\n %0Alaz\n", ""},
		{"@@ -1 +1 @@\n-a\n+b\n", ""},
		{"@@ -1,3 +0,0 @@\n-abc\n", ""},
		{"@@ -0,0 +1,3 @@\n+abc\n", ""},
		{"@@ _0,0 +0,0 @@\n+abc\n", "Invalid patch string: @@ _0,0 +0,0 @@"},
		{"Bad\nPatch\n", "Invalid patch string"},
	} {
		patches, err := dmp.PatchFromText(tc.Patch)
		if tc.ErrorMessagePrefix == "" {
			assert.Nil(t, err)

			if tc.Patch == "" {
				assert.Equal(t, []Patch{}, patches, fmt.Sprintf("Test case #%d, %#v", i, tc))
			} else {
				assert.Equal(t, tc.Patch, patches[0].String(), fmt.Sprintf("Test case #%d, %#v", i, tc))
			}
		} else {
			e := err.Error()
			if strings.HasPrefix(e, tc.ErrorMessagePrefix) {
				e = tc.ErrorMessagePrefix
			}
			assert.Equal(t, tc.ErrorMessagePrefix, e)
		}
	}

	diffs := []Diff{
		{DiffDelete, "`1234567890-=[]\\;',./"},
		{DiffInsert, "~!@#$%^&*()_+{}|:\"<>?"},
	}

	patches, err := dmp.PatchFromText("@@ -1,21 +1,21 @@\n-%601234567890-=%5B%5D%5C;',./\n+~!@#$%25%5E&*()_+%7B%7D%7C:%22%3C%3E?\n")
	assert.Len(t, patches, 1)
	assert.Equal(t, diffs,
		patches[0].diffs,
	)
	assert.Nil(t, err)
}

func TestPatchToText(t *testing.T) {
	type TestCase struct {
		Patch string
	}

	dmp := New()

	for i, tc := range []TestCase{
		{"@@ -21,18 +22,17 @@\n jump\n-s\n+ed\n  over \n-the\n+a\n  laz\n"},
		{"@@ -1,9 +1,9 @@\n-f\n+F\n oo+fooba\n@@ -7,9 +7,9 @@\n obar\n-,\n+.\n  tes\n"},
	} {
		patches, err := dmp.PatchFromText(tc.Patch)
		assert.Nil(t, err)

		actual := dmp.PatchToText(patches)
		assert.Equal(t, tc.Patch, actual, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}
}

func TestPatchAddContext(t *testing.T) {
	type TestCase struct {
		Name string

		Patch string
		Text  string

		Expected string
	}

	dmp := New()
	dmp.PatchMargin = 4

	for i, tc := range []TestCase{
		{"Simple case", "@@ -21,4 +21,10 @@\n-jump\n+somersault\n", "The quick brown fox jumps over the lazy dog.", "@@ -17,12 +17,18 @@\n fox \n-jump\n+somersault\n s ov\n"},
		{"Not enough trailing context", "@@ -21,4 +21,10 @@\n-jump\n+somersault\n", "The quick brown fox jumps.", "@@ -17,10 +17,16 @@\n fox \n-jump\n+somersault\n s.\n"},
		{"Not enough leading context", "@@ -3 +3,2 @@\n-e\n+at\n", "The quick brown fox jumps.", "@@ -1,7 +1,8 @@\n Th\n-e\n+at\n  qui\n"},
		{"Ambiguity", "@@ -3 +3,2 @@\n-e\n+at\n", "The quick brown fox jumps.  The quick brown fox crashes.", "@@ -1,27 +1,28 @@\n Th\n-e\n+at\n  quick brown fox jumps. \n"},
	} {
		patches, err := dmp.PatchFromText(tc.Patch)
		assert.Nil(t, err)

		actual := dmp.PatchAddContext(patches[0], tc.Text)
		assert.Equal(t, tc.Expected, actual.String(), fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}
}

func TestPatchMakeAndPatchToText(t *testing.T) {
	type TestCase struct {
		Name string

		Input1 interface{}
		Input2 interface{}
		Input3 interface{}

		Expected string
	}

	dmp := New()

	text1 := "The quick brown fox jumps over the lazy dog."
	text2 := "That quick brown fox jumped over a lazy dog."

	for i, tc := range []TestCase{
		{"Null case", "", "", nil, ""},
		{"Text2+Text1 inputs", text2, text1, nil, "@@ -1,8 +1,7 @@\n Th\n-at\n+e\n  qui\n@@ -21,17 +21,18 @@\n jump\n-ed\n+s\n  over \n-a\n+the\n  laz\n"},
		{"Text1+Text2 inputs", text1, text2, nil, "@@ -1,11 +1,12 @@\n Th\n-e\n+at\n  quick b\n@@ -22,18 +22,17 @@\n jump\n-s\n+ed\n  over \n-the\n+a\n  laz\n"},
		{"Diff input", dmp.DiffMain(text1, text2, false), nil, nil, "@@ -1,11 +1,12 @@\n Th\n-e\n+at\n  quick b\n@@ -22,18 +22,17 @@\n jump\n-s\n+ed\n  over \n-the\n+a\n  laz\n"},
		{"Text1+Diff inputs", text1, dmp.DiffMain(text1, text2, false), nil, "@@ -1,11 +1,12 @@\n Th\n-e\n+at\n  quick b\n@@ -22,18 +22,17 @@\n jump\n-s\n+ed\n  over \n-the\n+a\n  laz\n"},
		{"Text1+Text2+Diff inputs (deprecated)", text1, text2, dmp.DiffMain(text1, text2, false), "@@ -1,11 +1,12 @@\n Th\n-e\n+at\n  quick b\n@@ -22,18 +22,17 @@\n jump\n-s\n+ed\n  over \n-the\n+a\n  laz\n"},
		{"Character encoding", "`1234567890-=[]\\;',./", "~!@#$%^&*()_+{}|:\"<>?", nil, "@@ -1,21 +1,21 @@\n-%601234567890-=%5B%5D%5C;',./\n+~!@#$%25%5E&*()_+%7B%7D%7C:%22%3C%3E?\n"},
		{"Long string with repeats", strings.Repeat("abcdef", 100), strings.Repeat("abcdef", 100) + "123", nil, "@@ -573,28 +573,31 @@\n cdefabcdefabcdefabcdefabcdef\n+123\n"},
		{"Corner case of #31 fixed by #32", "2016-09-01T03:07:14.807830741Z", "2016-09-01T03:07:15.154800781Z", nil, "@@ -15,16 +15,16 @@\n 07:1\n+5.15\n 4\n-.\n 80\n+0\n 78\n-3074\n 1Z\n"},
	} {
		var patches []Patch
		if tc.Input3 != nil {
			patches = dmp.PatchMake(tc.Input1, tc.Input2, tc.Input3)
		} else if tc.Input2 != nil {
			patches = dmp.PatchMake(tc.Input1, tc.Input2)
		} else if ps, ok := tc.Input1.([]Patch); ok {
			patches = ps
		} else {
			patches = dmp.PatchMake(tc.Input1)
		}

		actual := dmp.PatchToText(patches)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}

	// Corner case of #28 wrong patch with timeout of 0
	dmp.DiffTimeout = 0

	text1 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ut risus et enim consectetur convallis a non ipsum. Sed nec nibh cursus, interdum libero vel."
	text2 = "Lorem a ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ut risus et enim consectetur convallis a non ipsum. Sed nec nibh cursus, interdum liberovel."

	diffs := dmp.DiffMain(text1, text2, true)
	// Additional check that the diff texts are equal to the originals even if we are using DiffMain with checklines=true #29
	assert.Equal(t, text1, dmp.DiffText1(diffs))
	assert.Equal(t, text2, dmp.DiffText2(diffs))

	patches := dmp.PatchMake(text1, diffs)

	actual := dmp.PatchToText(patches)
	assert.Equal(t, "@@ -1,14 +1,16 @@\n Lorem \n+a \n ipsum do\n@@ -148,13 +148,12 @@\n m libero\n- \n vel.\n", actual)

	// Check that empty Patch array is returned for no parameter call
	patches = dmp.PatchMake()
	assert.Equal(t, []Patch{}, patches)
}

func TestPatchSplitMax(t *testing.T) {
	type TestCase struct {
		Text1 string
		Text2 string

		Expected string
	}

	dmp := New()

	for i, tc := range []TestCase{
		{"abcdefghijklmnopqrstuvwxyz01234567890", "XabXcdXefXghXijXklXmnXopXqrXstXuvXwxXyzX01X23X45X67X89X0", "@@ -1,32 +1,46 @@\n+X\n ab\n+X\n cd\n+X\n ef\n+X\n gh\n+X\n ij\n+X\n kl\n+X\n mn\n+X\n op\n+X\n qr\n+X\n st\n+X\n uv\n+X\n wx\n+X\n yz\n+X\n 012345\n@@ -25,13 +39,18 @@\n zX01\n+X\n 23\n+X\n 45\n+X\n 67\n+X\n 89\n+X\n 0\n"},
		{"abcdef1234567890123456789012345678901234567890123456789012345678901234567890uvwxyz", "abcdefuvwxyz", "@@ -3,78 +3,8 @@\n cdef\n-1234567890123456789012345678901234567890123456789012345678901234567890\n uvwx\n"},
		{"1234567890123456789012345678901234567890123456789012345678901234567890", "abc", "@@ -1,32 +1,4 @@\n-1234567890123456789012345678\n 9012\n@@ -29,32 +1,4 @@\n-9012345678901234567890123456\n 7890\n@@ -57,14 +1,3 @@\n-78901234567890\n+abc\n"},
		{"abcdefghij , h : 0 , t : 1 abcdefghij , h : 0 , t : 1 abcdefghij , h : 0 , t : 1", "abcdefghij , h : 1 , t : 1 abcdefghij , h : 1 , t : 1 abcdefghij , h : 0 , t : 1", "@@ -2,32 +2,32 @@\n bcdefghij , h : \n-0\n+1\n  , t : 1 abcdef\n@@ -29,32 +29,32 @@\n bcdefghij , h : \n-0\n+1\n  , t : 1 abcdef\n"},
	} {
		patches := dmp.PatchMake(tc.Text1, tc.Text2)
		patches = dmp.PatchSplitMax(patches)

		actual := dmp.PatchToText(patches)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %#v", i, tc))
	}
}

func TestPatchAddPadding(t *testing.T) {
	type TestCase struct {
		Name string

		Text1 string
		Text2 string

		Expected            string
		ExpectedWithPadding string
	}

	dmp := New()

	for i, tc := range []TestCase{
		{"Both edges full", "", "test", "@@ -0,0 +1,4 @@\n+test\n", "@@ -1,8 +1,12 @@\n %01%02%03%04\n+test\n %01%02%03%04\n"},
		{"Both edges partial", "XY", "XtestY", "@@ -1,2 +1,6 @@\n X\n+test\n Y\n", "@@ -2,8 +2,12 @@\n %02%03%04X\n+test\n Y%01%02%03\n"},
		{"Both edges none", "XXXXYYYY", "XXXXtestYYYY", "@@ -1,8 +1,12 @@\n XXXX\n+test\n YYYY\n", "@@ -5,8 +5,12 @@\n XXXX\n+test\n YYYY\n"},
	} {
		patches := dmp.PatchMake(tc.Text1, tc.Text2)

		actual := dmp.PatchToText(patches)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))

		dmp.PatchAddPadding(patches)

		actualWithPadding := dmp.PatchToText(patches)
		assert.Equal(t, tc.ExpectedWithPadding, actualWithPadding, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}
}

func TestPatchApply(t *testing.T) {
	type TestCase struct {
		Name string

		Text1    string
		Text2    string
		TextBase string

		Expected        string
		ExpectedApplies []bool
	}

	dmp := New()
	dmp.MatchDistance = 1000
	dmp.MatchThreshold = 0.5
	dmp.PatchDeleteThreshold = 0.5

	for i, tc := range []TestCase{
		{"Null case", "", "", "Hello world.", "Hello world.", []bool{}},
		{"Exact match", "The quick brown fox jumps over the lazy dog.", "That quick brown fox jumped over a lazy dog.", "The quick brown fox jumps over the lazy dog.", "That quick brown fox jumped over a lazy dog.", []bool{true, true}},
		{"Partial match", "The quick brown fox jumps over the lazy dog.", "That quick brown fox jumped over a lazy dog.", "The quick red rabbit jumps over the tired tiger.", "That quick red rabbit jumped over a tired tiger.", []bool{true, true}},
		{"Failed match", "The quick brown fox jumps over the lazy dog.", "That quick brown fox jumped over a lazy dog.", "I am the very model of a modern major general.", "I am the very model of a modern major general.", []bool{false, false}},
		{"Big delete, small Diff", "x1234567890123456789012345678901234567890123456789012345678901234567890y", "xabcy", "x123456789012345678901234567890-----++++++++++-----123456789012345678901234567890y", "xabcy", []bool{true, true}},
		{"Big delete, big Diff 1", "x1234567890123456789012345678901234567890123456789012345678901234567890y", "xabcy", "x12345678901234567890---------------++++++++++---------------12345678901234567890y", "xabc12345678901234567890---------------++++++++++---------------12345678901234567890y", []bool{false, true}},
	} {
		patches := dmp.PatchMake(tc.Text1, tc.Text2)

		actual, actualApplies := dmp.PatchApply(patches, tc.TextBase)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
		assert.Equal(t, tc.ExpectedApplies, actualApplies, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}

	dmp.PatchDeleteThreshold = 0.6

	for i, tc := range []TestCase{
		{"Big delete, big Diff 2", "x1234567890123456789012345678901234567890123456789012345678901234567890y", "xabcy", "x12345678901234567890---------------++++++++++---------------12345678901234567890y", "xabcy", []bool{true, true}},
	} {
		patches := dmp.PatchMake(tc.Text1, tc.Text2)

		actual, actualApplies := dmp.PatchApply(patches, tc.TextBase)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
		assert.Equal(t, tc.ExpectedApplies, actualApplies, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}

	dmp.MatchDistance = 0
	dmp.MatchThreshold = 0.0
	dmp.PatchDeleteThreshold = 0.5

	for i, tc := range []TestCase{
		{"Compensate for failed patch", "abcdefghijklmnopqrstuvwxyz--------------------1234567890", "abcXXXXXXXXXXdefghijklmnopqrstuvwxyz--------------------1234567YYYYYYYYYY890", "ABCDEFGHIJKLMNOPQRSTUVWXYZ--------------------1234567890", "ABCDEFGHIJKLMNOPQRSTUVWXYZ--------------------1234567YYYYYYYYYY890", []bool{false, true}},
	} {
		patches := dmp.PatchMake(tc.Text1, tc.Text2)

		actual, actualApplies := dmp.PatchApply(patches, tc.TextBase)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
		assert.Equal(t, tc.ExpectedApplies, actualApplies, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}

	dmp.MatchThreshold = 0.5
	dmp.MatchDistance = 1000

	for i, tc := range []TestCase{
		{"No side effects", "", "test", "", "test", []bool{true}},
		{"No side effects with major delete", "The quick brown fox jumps over the lazy dog.", "Woof", "The quick brown fox jumps over the lazy dog.", "Woof", []bool{true, true}},
		{"Edge exact match", "", "test", "", "test", []bool{true}},
		{"Near edge exact match", "XY", "XtestY", "XY", "XtestY", []bool{true}},
		{"Edge partial match", "y", "y123", "x", "x123", []bool{true}},
	} {
		patches := dmp.PatchMake(tc.Text1, tc.Text2)

		actual, actualApplies := dmp.PatchApply(patches, tc.TextBase)
		assert.Equal(t, tc.Expected, actual, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
		assert.Equal(t, tc.ExpectedApplies, actualApplies, fmt.Sprintf("Test case #%d, %s", i, tc.Name))
	}
}
