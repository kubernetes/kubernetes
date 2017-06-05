/*
Copyright 2015 The Kubernetes Authors.

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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_updateMacroBlock(t *testing.T) {
	token := "TOKEN"
	BEGIN := beginMungeTag(token)
	END := endMungeTag(token)

	var cases = []struct {
		in  string
		out string
	}{
		{"", ""},
		{"Lorem ipsum\ndolor sit amet\n",
			"Lorem ipsum\ndolor sit amet\n"},
		{"Lorem ipsum \n" + BEGIN + "\ndolor\n" + END + "\nsit amet\n",
			"Lorem ipsum \n" + BEGIN + "\nfoo\n" + END + "\nsit amet\n"},
	}
	for _, c := range cases {
		in := getMungeLines(c.in)
		expected := getMungeLines(c.out)
		actual, err := updateMacroBlock(in, token, getMungeLines("foo"))
		assert.NoError(t, err)
		if !expected.Equal(actual) {
			t.Errorf("Expected '%v' but got '%v'", expected.String(), expected.String())
		}
	}
}

func Test_updateMacroBlock_errors(t *testing.T) {
	token := "TOKEN"
	b := beginMungeTag(token)
	e := endMungeTag(token)

	var cases = []struct {
		in string
	}{
		{b + "\n"},
		{"blah\n" + b + "\nblah"},
		{e + "\n"},
		{"blah\n" + e + "\nblah\n"},
		{e + "\n" + b},
		{b + "\n" + e + "\n" + e},
		{b + "\n" + b + "\n" + e},
		{b + "\n" + b + "\n" + e + "\n" + e},
	}
	for _, c := range cases {
		in := getMungeLines(c.in)
		_, err := updateMacroBlock(in, token, getMungeLines("foo"))
		assert.Error(t, err)
	}
}

func TestHasLine(t *testing.T) {
	cases := []struct {
		haystack string
		needle   string
		expected bool
	}{
		{"abc\ndef\nghi", "abc", true},
		{"  abc\ndef\nghi", "abc", true},
		{"abc  \ndef\nghi", "abc", true},
		{"\n abc\ndef\nghi", "abc", true},
		{"abc \n\ndef\nghi", "abc", true},
		{"abc\ndef\nghi", "def", true},
		{"abc\ndef\nghi", "ghi", true},
		{"abc\ndef\nghi", "xyz", false},
	}

	for i, c := range cases {
		in := getMungeLines(c.haystack)
		if hasLine(in, c.needle) != c.expected {
			t.Errorf("case[%d]: %q, expected %t, got %t", i, c.needle, c.expected, !c.expected)
		}
	}
}

func TestHasMacroBlock(t *testing.T) {
	token := "<<<"
	b := beginMungeTag(token)
	e := endMungeTag(token)
	cases := []struct {
		lines    []string
		expected bool
	}{
		{[]string{b, e}, true},
		{[]string{b, "abc", e}, true},
		{[]string{b, b, "abc", e}, true},
		{[]string{b, "abc", e, e}, true},
		{[]string{b, e, b, e}, true},
		{[]string{b}, false},
		{[]string{e}, false},
		{[]string{b, "abc"}, false},
		{[]string{"abc", e}, false},
	}

	for i, c := range cases {
		in := getMungeLines(strings.Join(c.lines, "\n"))
		if hasMacroBlock(in, token) != c.expected {
			t.Errorf("case[%d]: expected %t, got %t", i, c.expected, !c.expected)
		}
	}
}

func TestAppendMacroBlock(t *testing.T) {
	token := "<<<"
	b := beginMungeTag(token)
	e := endMungeTag(token)
	cases := []struct {
		in       []string
		expected []string
	}{
		{[]string{}, []string{b, e}},
		{[]string{"bob"}, []string{"bob", "", b, e}},
		{[]string{b, e}, []string{b, e, "", b, e}},
	}
	for i, c := range cases {
		in := getMungeLines(strings.Join(c.in, "\n"))
		expected := getMungeLines(strings.Join(c.expected, "\n"))
		out := appendMacroBlock(in, token)
		if !out.Equal(expected) {
			t.Errorf("Case[%d]: expected '%q' but got '%q'", i, expected.String(), out.String())
		}
	}
}

func TestPrependMacroBlock(t *testing.T) {
	token := "<<<"
	b := beginMungeTag(token)
	e := endMungeTag(token)
	cases := []struct {
		in       []string
		expected []string
	}{
		{[]string{}, []string{b, e}},
		{[]string{"bob"}, []string{b, e, "", "bob"}},
		{[]string{b, e}, []string{b, e, "", b, e}},
	}
	for i, c := range cases {
		in := getMungeLines(strings.Join(c.in, "\n"))
		expected := getMungeLines(strings.Join(c.expected, "\n"))
		out := prependMacroBlock(token, in)
		if !out.Equal(expected) {
			t.Errorf("Case[%d]: expected '%q' but got '%q'", i, expected.String(), out.String())
		}
	}
}
