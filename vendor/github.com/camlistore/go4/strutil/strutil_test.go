/*
Copyright 2013 The Camlistore Authors

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

package strutil

import (
	"reflect"
	"strings"
	"testing"
)

func TestAppendSplitN(t *testing.T) {
	var got []string
	tests := []struct {
		s, sep string
		n      int
	}{
		{"foo", "|", 1},
		{"foo", "|", -1},
		{"foo|bar", "|", 1},
		{"foo|bar", "|", -1},
		{"foo|bar|", "|", 2},
		{"foo|bar|", "|", -1},
		{"foo|bar|baz", "|", 1},
		{"foo|bar|baz", "|", 2},
		{"foo|bar|baz", "|", 3},
		{"foo|bar|baz", "|", -1},
	}
	for _, tt := range tests {
		want := strings.SplitN(tt.s, tt.sep, tt.n)
		got = AppendSplitN(got[:0], tt.s, tt.sep, tt.n)
		if !reflect.DeepEqual(want, got) {
			t.Errorf("AppendSplitN(%q, %q, %d) = %q; want %q",
				tt.s, tt.sep, tt.n, got, want)
		}
	}
}

func TestStringFromBytes(t *testing.T) {
	for _, s := range []string{"foo", "permanode", "file", "zzzz"} {
		got := StringFromBytes([]byte(s))
		if got != s {
			t.Errorf("StringFromBytes(%q) didn't round-trip; got %q instead", s, got)
		}
	}
}

func TestHasPrefixFold(t *testing.T) {
	tests := []struct {
		s, prefix string
		result    bool
	}{
		{"camli", "CAML", true},
		{"CAMLI", "caml", true},
		{"cam", "Cam", true},
		{"camli", "car", false},
		{"caml", "camli", false},
		{"Hello, 世界 dasdsa", "HeLlO, 世界", true},
		{"Hello, 世界", "HeLlO, 世界-", false},

		{"kelvin", "\u212A" + "elvin", true}, // "\u212A" is the Kelvin temperature sign
		{"Kelvin", "\u212A" + "elvin", true},
		{"kelvin", "\u212A" + "el", true},
		{"Kelvin", "\u212A" + "el", true},
		{"\u212A" + "elvin", "Kelvin", true},
		{"\u212A" + "elvin", "kelvin", true},
		{"\u212A" + "elvin", "Kel", true},
		{"\u212A" + "elvin", "kel", true},
	}
	for _, tt := range tests {
		r := HasPrefixFold(tt.s, tt.prefix)
		if r != tt.result {
			t.Errorf("HasPrefixFold(%q, %q) returned %v", tt.s, tt.prefix, r)
		}
	}
}

func TestHasSuffixFold(t *testing.T) {
	tests := []struct {
		s, suffix string
		result    bool
	}{
		{"camli", "AMLI", true},
		{"CAMLI", "amli", true},
		{"mli", "MLI", true},
		{"camli", "ali", false},
		{"amli", "camli", false},
		{"asas Hello, 世界", "HeLlO, 世界", true},
		{"Hello, 世界", "HeLlO, 世界-", false},
		{"KkkkKKkelvin", "\u212A" + "elvin", true}, // "\u212A" is the Kelvin temperature sign

		{"kelvin", "\u212A" + "elvin", true}, // "\u212A" is the Kelvin temperature sign
		{"Kelvin", "\u212A" + "elvin", true},
		{"\u212A" + "elvin", "Kelvin", true},
		{"\u212A" + "elvin", "kelvin", true},
		{"\u212A" + "elvin", "vin", true},
		{"\u212A" + "elvin", "viN", true},
	}
	for _, tt := range tests {
		r := HasSuffixFold(tt.s, tt.suffix)
		if r != tt.result {
			t.Errorf("HasSuffixFold(%q, %q) returned %v", tt.s, tt.suffix, r)
		}
	}
}

func TestContainsFold(t *testing.T) {
	// TODO: more tests, more languages.
	tests := []struct {
		s, substr string
		result    bool
	}{
		{"camli", "CAML", true},
		{"CAMLI", "caml", true},
		{"cam", "Cam", true},
		{"мир", "ми", true},
		{"МИP", "ми", true},
		{"КАМЛИЙСТОР", "камлийс", true},
		{"КаМлИйСтОр", "КаМлИйС", true},
		{"camli", "car", false},
		{"caml", "camli", false},

		{"camli", "AMLI", true},
		{"CAMLI", "amli", true},
		{"mli", "MLI", true},
		{"мир", "ир", true},
		{"МИP", "ми", true},
		{"КАМЛИЙСТОР", "лийстор", true},
		{"КаМлИйСтОр", "лИйСтОр", true},
		{"мир", "р", true},
		{"camli", "ali", false},
		{"amli", "camli", false},

		{"МИP", "и", true},
		{"мир", "и", true},
		{"КАМЛИЙСТОР", "лийс", true},
		{"КаМлИйСтОр", "лИйС", true},

		{"árvíztűrő tükörfúrógép", "árvíztŰrŐ", true},
		{"I love ☕", "i love ☕", true},

		{"k", "\u212A", true}, // "\u212A" is the Kelvin temperature sign
		{"\u212A" + "elvin", "k", true},
		{"kelvin", "\u212A" + "elvin", true},
		{"Kelvin", "\u212A" + "elvin", true},
		{"\u212A" + "elvin", "Kelvin", true},
		{"\u212A" + "elvin", "kelvin", true},
		{"273.15 kelvin", "\u212A" + "elvin", true},
		{"273.15 Kelvin", "\u212A" + "elvin", true},
		{"273.15 \u212A" + "elvin", "Kelvin", true},
		{"273.15 \u212A" + "elvin", "kelvin", true},
	}
	for _, tt := range tests {
		r := ContainsFold(tt.s, tt.substr)
		if r != tt.result {
			t.Errorf("ContainsFold(%q, %q) returned %v", tt.s, tt.substr, r)
		}
	}
}

func TestIsPlausibleJSON(t *testing.T) {
	tests := []struct {
		in   string
		want bool
	}{
		{"{}", true},
		{" {}", true},
		{"{} ", true},
		{"\n\r\t {}\t \r \n", true},

		{"\n\r\t {x\t \r \n", false},
		{"{x", false},
		{"x}", false},
		{"x", false},
		{"", false},
	}
	for _, tt := range tests {
		got := IsPlausibleJSON(tt.in)
		if got != tt.want {
			t.Errorf("IsPlausibleJSON(%q) = %v; want %v", tt.in, got, tt.want)
		}
	}
}

func BenchmarkHasSuffixFoldToLower(tb *testing.B) {
	a, b := "camlik", "AMLI\u212A"
	for i := 0; i < tb.N; i++ {
		if !strings.HasSuffix(strings.ToLower(a), strings.ToLower(b)) {
			tb.Fatalf("%q should have the same suffix as %q", a, b)
		}
	}
}
func BenchmarkHasSuffixFold(tb *testing.B) {
	a, b := "camlik", "AMLI\u212A"
	for i := 0; i < tb.N; i++ {
		if !HasSuffixFold(a, b) {
			tb.Fatalf("%q should have the same suffix as %q", a, b)
		}
	}
}

func BenchmarkHasPrefixFoldToLower(tb *testing.B) {
	a, b := "kamlistore", "\u212AAMLI"
	for i := 0; i < tb.N; i++ {
		if !strings.HasPrefix(strings.ToLower(a), strings.ToLower(b)) {
			tb.Fatalf("%q should have the same suffix as %q", a, b)
		}
	}
}
func BenchmarkHasPrefixFold(tb *testing.B) {
	a, b := "kamlistore", "\u212AAMLI"
	for i := 0; i < tb.N; i++ {
		if !HasPrefixFold(a, b) {
			tb.Fatalf("%q should have the same suffix as %q", a, b)
		}
	}
}
