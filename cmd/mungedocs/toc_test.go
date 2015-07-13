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

func Test_buildTOC(t *testing.T) {
	var cases = []struct {
		in  string
		out string
	}{
		{"", ""},
		{"Lorem ipsum\ndolor sit amet\n", ""},
		{
			"# Title\nLorem ipsum \n## Section Heading\ndolor sit amet\n",
			"- [Title](#title)\n  - [Section Heading](#section-heading)\n",
		},
		{
			"# Title\nLorem ipsum \n## Section Heading\ndolor sit amet\n```bash\n#!/bin/sh\n```",
			"- [Title](#title)\n  - [Section Heading](#section-heading)\n",
		},
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
		{
			"Lorem ipsum\ndolor sit amet\n",
			"Lorem ipsum\ndolor sit amet\n",
		},
		{
			"# Title\nLorem ipsum \n**table of contents**\n<!-- BEGIN MUNGE: GENERATED_TOC -->\nold cruft\n<!-- END MUNGE: GENERATED_TOC -->\n## Section Heading\ndolor sit amet\n",
			"# Title\nLorem ipsum \n**table of contents**\n<!-- BEGIN MUNGE: GENERATED_TOC -->\n- [Title](#title)\n  - [Section Heading](#section-heading)\n\n<!-- END MUNGE: GENERATED_TOC -->\n## Section Heading\ndolor sit amet\n",
		},
	}
	for _, c := range cases {
		actual, err := updateTOC("filename.md", []byte(c.in))
		assert.NoError(t, err)
		if c.out != string(actual) {
			t.Errorf("Expected TOC '%v' but got '%v'", c.out, string(actual))
		}
	}
}
