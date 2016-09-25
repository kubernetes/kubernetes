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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPreformatted(t *testing.T) {
	var cases = []struct {
		in       string
		expected string
	}{
		{"", ""},
		{
			"```\nbob\n```",
			"\n```\nbob\n```\n\n",
		},
		{
			"```\nbob\n```\n```\nnotbob\n```\n",
			"\n```\nbob\n```\n\n```\nnotbob\n```\n\n",
		},
		{
			"```bob```\n",
			"```bob```\n",
		},
		{
			"    ```\n    bob\n    ```",
			"\n    ```\n    bob\n    ```\n\n",
		},
	}
	for i, c := range cases {
		in := getMungeLines(c.in)
		expected := getMungeLines(c.expected)
		actual, err := updatePreformatted("filename.md", in)
		assert.NoError(t, err)
		if !actual.Equal(expected) {
			t.Errorf("case[%d]: expected %q got %q", i, c.expected, actual.String())
		}
	}
}

func TestPreformattedImbalance(t *testing.T) {
	var cases = []struct {
		in string
		ok bool
	}{
		{"", true},
		{"```\nin\n```", true},
		{"```\nin\n```\nout", true},
		{"```", false},
		{"```\nin\n```\nout\n```", false},
	}
	for i, c := range cases {
		in := getMungeLines(c.in)
		out, err := checkPreformatBalance("filename.md", in)
		if err != nil && c.ok {
			t.Errorf("case[%d]: expected success", i)
		}
		if err == nil && !c.ok {
			t.Errorf("case[%d]: expected failure", i)
		}
		// Even in case of misformat, return all the text,
		// so that the user's work is not lost.
		if !equalMungeLines(out, in) {
			t.Errorf("case[%d]: expected munged text to be identical to input text", i)
		}
	}
}

func equalMungeLines(a, b mungeLines) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
