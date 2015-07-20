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
	"reflect"
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
		{"Lorem ipsum \n " + BEGIN + "\ndolor\n" + END + "\nsit amet\n",
			"Lorem ipsum \n " + BEGIN + "\nfoo\n\n" + END + "\nsit amet\n"},
	}
	for _, c := range cases {
		actual, err := updateMacroBlock(splitLines([]byte(c.in)), "TOKEN", "foo\n")
		assert.NoError(t, err)
		if c.out != string(actual) {
			t.Errorf("Expected '%v' but got '%v'", c.out, string(actual))
		}
	}
}

func Test_updateMacroBlock_errors(t *testing.T) {
	b := beginMungeTag("TOKEN")
	e := endMungeTag("TOKEN")

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
		_, err := updateMacroBlock(splitLines([]byte(c.in)), "TOKEN", "foo")
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
		if hasMacroBlock(c.lines, token) != c.expected {
			t.Errorf("case[%d]: expected %t, got %t", i, c.expected, !c.expected)
		}
	}
}

func TestReplaceNonPreformatted(t *testing.T) {
	cases := []struct {
		in  string
		out string
	}{
		{"aoeu", ""},
		{"aoeu\n```\naoeu\n```\naoeu", "```\naoeu\n```\n"},
		{"ao\neu\n```\naoeu\n\n\n", "```\naoeu\n\n\n"},
		{"aoeu ```aoeu``` aoeu", ""},
	}

	for i, c := range cases {
		out := string(replaceNonPreformatted([]byte(c.in), func([]byte) []byte { return nil }))
		if out != c.out {
			t.Errorf("%v: got %q, wanted %q", i, out, c.out)
		}
	}
}

func TestReplaceNonPreformattedNoChange(t *testing.T) {
	cases := []struct {
		in string
	}{
		{"aoeu"},
		{"aoeu\n```\naoeu\n```\naoeu"},
		{"aoeu\n\n```\n\naoeu\n\n```\n\naoeu"},
		{"ao\neu\n```\naoeu\n\n\n"},
		{"aoeu ```aoeu``` aoeu"},
		{"aoeu\n```\naoeu\n```"},
		{"aoeu\n```\naoeu\n```\n"},
		{"aoeu\n```\naoeu\n```\n\n"},
	}

	for i, c := range cases {
		out := string(replaceNonPreformatted([]byte(c.in), func(in []byte) []byte { return in }))
		if out != c.in {
			t.Errorf("%v: got %q, wanted %q", i, out, c.in)
		}
	}
}

func TestReplaceNonPreformattedCallOrder(t *testing.T) {
	cases := []struct {
		in     string
		expect []string
	}{
		{"aoeu", []string{"aoeu"}},
		{"aoeu\n```\naoeu\n```\naoeu", []string{"aoeu\n", "aoeu"}},
		{"aoeu\n\n```\n\naoeu\n\n```\n\naoeu", []string{"aoeu\n\n", "\naoeu"}},
		{"ao\neu\n```\naoeu\n\n\n", []string{"ao\neu\n"}},
		{"aoeu ```aoeu``` aoeu", []string{"aoeu ```aoeu``` aoeu"}},
		{"aoeu\n```\naoeu\n```", []string{"aoeu\n"}},
		{"aoeu\n```\naoeu\n```\n", []string{"aoeu\n"}},
		{"aoeu\n```\naoeu\n```\n\n", []string{"aoeu\n", "\n"}},
	}

	for i, c := range cases {
		got := []string{}
		replaceNonPreformatted([]byte(c.in), func(in []byte) []byte {
			got = append(got, string(in))
			return in
		})
		if e, a := c.expect, got; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: got %q, wanted %q", i, a, e)
		}
	}
}
