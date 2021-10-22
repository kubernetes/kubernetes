// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import (
	"strings"
	"testing"
)

func doIterNormString(f Form, s string) []byte {
	acc := []byte{}
	i := Iter{}
	i.InitString(f, s)
	for !i.Done() {
		acc = append(acc, i.Next()...)
	}
	return acc
}

func doIterNorm(f Form, s string) []byte {
	acc := []byte{}
	i := Iter{}
	i.Init(f, []byte(s))
	for !i.Done() {
		acc = append(acc, i.Next()...)
	}
	return acc
}

func TestIterNext(t *testing.T) {
	runNormTests(t, "IterNext", func(f Form, out []byte, s string) []byte {
		return doIterNormString(f, string(append(out, s...)))
	})
	runNormTests(t, "IterNext", func(f Form, out []byte, s string) []byte {
		return doIterNorm(f, string(append(out, s...)))
	})
}

type SegmentTest struct {
	in  string
	out []string
}

var segmentTests = []SegmentTest{
	{"\u1E0A\u0323a", []string{"\x44\u0323\u0307", "a", ""}},
	{rep('a', segSize), append(strings.Split(rep('a', segSize), ""), "")},
	{rep('a', segSize+2), append(strings.Split(rep('a', segSize+2), ""), "")},
	{rep('a', segSize) + "\u0300aa",
		append(strings.Split(rep('a', segSize-1), ""), "a\u0300", "a", "a", "")},

	// U+0f73 is NOT treated as a starter as it is a modifier
	{"a" + grave(29) + "\u0f73", []string{"a" + grave(29), cgj + "\u0f73"}},
	{"a\u0f73", []string{"a\u0f73"}},

	// U+ff9e is treated as a non-starter.
	// TODO: should we? Note that this will only affect iteration, as whether
	// or not we do so does not affect the normalization output and will either
	// way result in consistent iteration output.
	{"a" + grave(30) + "\uff9e", []string{"a" + grave(30), cgj + "\uff9e"}},
	{"a\uff9e", []string{"a\uff9e"}},
}

var segmentTestsK = []SegmentTest{
	{"\u3332", []string{"\u30D5", "\u30A1", "\u30E9", "\u30C3", "\u30C8\u3099", ""}},
	// last segment of multi-segment decomposition needs normalization
	{"\u3332\u093C", []string{"\u30D5", "\u30A1", "\u30E9", "\u30C3", "\u30C8\u093C\u3099", ""}},
	{"\u320E", []string{"\x28", "\uAC00", "\x29"}},

	// last segment should be copied to start of buffer.
	{"\ufdfa", []string{"\u0635", "\u0644", "\u0649", " ", "\u0627", "\u0644", "\u0644", "\u0647", " ", "\u0639", "\u0644", "\u064a", "\u0647", " ", "\u0648", "\u0633", "\u0644", "\u0645", ""}},
	{"\ufdfa" + grave(30), []string{"\u0635", "\u0644", "\u0649", " ", "\u0627", "\u0644", "\u0644", "\u0647", " ", "\u0639", "\u0644", "\u064a", "\u0647", " ", "\u0648", "\u0633", "\u0644", "\u0645" + grave(30), ""}},
	{"\uFDFA" + grave(64), []string{"\u0635", "\u0644", "\u0649", " ", "\u0627", "\u0644", "\u0644", "\u0647", " ", "\u0639", "\u0644", "\u064a", "\u0647", " ", "\u0648", "\u0633", "\u0644", "\u0645" + grave(30), cgj + grave(30), cgj + grave(4), ""}},

	// Hangul and Jamo are grouped together.
	{"\uAC00", []string{"\u1100\u1161", ""}},
	{"\uAC01", []string{"\u1100\u1161\u11A8", ""}},
	{"\u1100\u1161", []string{"\u1100\u1161", ""}},
}

// Note that, by design, segmentation is equal for composing and decomposing forms.
func TestIterSegmentation(t *testing.T) {
	segmentTest(t, "SegmentTestD", NFD, segmentTests)
	segmentTest(t, "SegmentTestC", NFC, segmentTests)
	segmentTest(t, "SegmentTestKD", NFKD, segmentTestsK)
	segmentTest(t, "SegmentTestKC", NFKC, segmentTestsK)
}

func segmentTest(t *testing.T, name string, f Form, tests []SegmentTest) {
	iter := Iter{}
	for i, tt := range tests {
		iter.InitString(f, tt.in)
		for j, seg := range tt.out {
			if seg == "" {
				if !iter.Done() {
					res := string(iter.Next())
					t.Errorf(`%s:%d:%d: expected Done()==true, found segment %+q`, name, i, j, res)
				}
				continue
			}
			if iter.Done() {
				t.Errorf("%s:%d:%d: Done()==true, want false", name, i, j)
			}
			seg = f.String(seg)
			if res := string(iter.Next()); res != seg {
				t.Errorf(`%s:%d:%d" segment was %+q (%d); want %+q (%d)`, name, i, j, pc(res), len(res), pc(seg), len(seg))
			}
		}
	}
}
