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
		actual, err := updateMacroBlock([]byte(c.in), "BEGIN", "END", "foo\n")
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
		_, err := updateMacroBlock([]byte(c.in), "BEGIN", "END", "foo")
		assert.Error(t, err)
	}
}

func Test_buildTOC(t *testing.T) {
	var cases = []struct {
		in  string
		out string
	}{
		{"", ""},
		{"Lorem ipsum\ndolor sit amet\n", ""},
		{"# Title\nLorem ipsum \n## Section Heading\ndolor sit amet\n",
			"- [Title](#title)\n  - [Section Heading](#section-heading)\n"},
	}
	for _, c := range cases {
		actual, err := buildTOC([]byte(c.in))
		assert.NoError(t, err)
		if c.out != string(actual) {
			t.Errorf("Expected TOC '%v' but got '%v'", c.out, string(actual))
		}
	}
}

func Test_updateTOC(t *testing.T) {
	var cases = []struct {
		in  string
		out string
	}{
		{"", ""},
		{"Lorem ipsum\ndolor sit amet\n",
			"Lorem ipsum\ndolor sit amet\n"},
		{"# Title\nLorem ipsum \n**table of contents**\n<!-- BEGIN GENERATED TOC -->\nold cruft\n<!-- END GENERATED TOC -->\n## Section Heading\ndolor sit amet\n",
			"# Title\nLorem ipsum \n**table of contents**\n<!-- BEGIN GENERATED TOC -->\n- [Title](#title)\n  - [Section Heading](#section-heading)\n\n<!-- END GENERATED TOC -->\n## Section Heading\ndolor sit amet\n"},
	}
	for _, c := range cases {
		actual, err := updateTOC([]byte(c.in))
		assert.NoError(t, err)
		if c.out != string(actual) {
			t.Errorf("Expected TOC '%v' but got '%v'", c.out, string(actual))
		}
	}
}
