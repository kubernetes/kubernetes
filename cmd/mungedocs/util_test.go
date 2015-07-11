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

	"github.com/stretchr/testify/assert"
)

func Test_updateMacroBlock(t *testing.T) {
	var cases = []struct {
		in  string
		out string
	}{
		{"", ""},
		{"Lorem ipsum\ndolor sit amet\n",
			"Lorem ipsum\ndolor sit amet\n"},
		{"Lorem ipsum \n BEGIN\ndolor\nEND\nsit amet\n",
			"Lorem ipsum \n BEGIN\nfoo\n\nEND\nsit amet\n"},
	}
	for _, c := range cases {
		actual, err := updateMacroBlock(splitLines([]byte(c.in)), "BEGIN", "END", "foo\n")
		assert.NoError(t, err)
		if c.out != string(actual) {
			t.Errorf("Expected '%v' but got '%v'", c.out, string(actual))
		}
	}
}

func Test_updateMacroBlock_errors(t *testing.T) {
	var cases = []struct {
		in string
	}{
		{"BEGIN\n"},
		{"blah\nBEGIN\nblah"},
		{"END\n"},
		{"blah\nEND\nblah\n"},
		{"END\nBEGIN"},
		{"BEGIN\nEND\nEND"},
		{"BEGIN\nBEGIN\nEND"},
		{"BEGIN\nBEGIN\nEND\nEND"},
	}
	for _, c := range cases {
		_, err := updateMacroBlock(splitLines([]byte(c.in)), "BEGIN", "END", "foo")
		assert.Error(t, err)
	}
}

func TestHasLine(t *testing.T) {
	cases := []struct {
		lines    []string
		needle   string
		expected bool
	}{
		{[]string{"abc", "def", "ghi"}, "abc", true},
		{[]string{"  abc", "def", "ghi"}, "abc", true},
		{[]string{"abc  ", "def", "ghi"}, "abc", true},
		{[]string{"\n abc", "def", "ghi"}, "abc", true},
		{[]string{"abc \n", "def", "ghi"}, "abc", true},
		{[]string{"abc", "def", "ghi"}, "def", true},
		{[]string{"abc", "def", "ghi"}, "ghi", true},
		{[]string{"abc", "def", "ghi"}, "xyz", false},
	}

	for i, c := range cases {
		if hasLine(c.lines, c.needle) != c.expected {
			t.Errorf("case[%d]: %q, expected %t, got %t", i, c.needle, c.expected, !c.expected)
		}
	}
}

func TestHasMacroBlock(t *testing.T) {
	cases := []struct {
		lines    []string
		begin    string
		end      string
		expected bool
	}{
		{[]string{"<<<", ">>>"}, "<<<", ">>>", true},
		{[]string{"<<<", "abc", ">>>"}, "<<<", ">>>", true},
		{[]string{"<<<", "<<<", "abc", ">>>"}, "<<<", ">>>", true},
		{[]string{"<<<", "abc", ">>>", ">>>"}, "<<<", ">>>", true},
		{[]string{"<<<", ">>>", "<<<", ">>>"}, "<<<", ">>>", true},
		{[]string{"<<<"}, "<<<", ">>>", false},
		{[]string{">>>"}, "<<<", ">>>", false},
		{[]string{"<<<", "abc"}, "<<<", ">>>", false},
		{[]string{"abc", ">>>"}, "<<<", ">>>", false},
	}

	for i, c := range cases {
		if hasMacroBlock(c.lines, c.begin, c.end) != c.expected {
			t.Errorf("case[%d]: %q,%q, expected %t, got %t", i, c.begin, c.end, c.expected, !c.expected)
		}
	}
}
