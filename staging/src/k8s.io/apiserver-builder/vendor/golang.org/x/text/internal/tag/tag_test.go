// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tag

import (
	"strings"
	"testing"
)

var strdata = []string{
	"aa  ",
	"aaa ",
	"aaaa",
	"aaab",
	"aab ",
	"ab  ",
	"ba  ",
	"xxxx",
	"\xff\xff\xff\xff",
}

var testCases = map[string]int{
	"a":    0,
	"aa":   0,
	"aaa":  1,
	"aa ":  0,
	"aaaa": 2,
	"aaab": 3,
	"b":    6,
	"ba":   6,
	"    ": -1,
	"aaax": -1,
	"bbbb": -1,
	"zzzz": -1,
}

func TestIndex(t *testing.T) {
	index := Index(strings.Join(strdata, ""))
	for k, v := range testCases {
		if i := index.Index([]byte(k)); i != v {
			t.Errorf("%s: got %d; want %d", k, i, v)
		}
	}
}

func TestFixCase(t *testing.T) {
	tests := []string{
		"aaaa", "AbCD", "abcd",
		"Zzzz", "AbCD", "Abcd",
		"Zzzz", "AbC", "",
		"XXX", "ab ", "",
		"XXX", "usd", "USD",
		"cmn", "AB ", "",
		"gsw", "CMN", "cmn",
	}
	for tc := tests; len(tc) > 0; tc = tc[3:] {
		b := []byte(tc[1])
		if !FixCase(tc[0], b) {
			b = nil
		}
		if string(b) != tc[2] {
			t.Errorf("FixCase(%q, %q) = %q; want %q", tc[0], tc[1], b, tc[2])
		}
	}
}
