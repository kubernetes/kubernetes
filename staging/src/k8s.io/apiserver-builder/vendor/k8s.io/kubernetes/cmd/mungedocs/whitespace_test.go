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

func Test_updateWhiteSpace(t *testing.T) {
	var cases = []struct {
		in       string
		expected string
	}{
		{"", ""},
		{"\n", "\n"},
		{"  \t   \t \n", "\n"},
		{"bob  \t", "bob"},
		{"```\n   \n```\n", "```\n   \n```\n"},
	}
	for i, c := range cases {
		in := getMungeLines(c.in)
		expected := getMungeLines(c.expected)
		actual, err := updateWhitespace("filename.md", in)
		assert.NoError(t, err)
		if !expected.Equal(actual) {
			t.Errorf("Case[%d] Expected Whitespace '%v' but got '%v'", i, string(expected.Bytes()), string(actual.Bytes()))
		}
	}
}
