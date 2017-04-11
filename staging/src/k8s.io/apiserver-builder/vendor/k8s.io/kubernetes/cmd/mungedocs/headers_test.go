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

func TestHeaderLines(t *testing.T) {
	var cases = []struct {
		in       string
		expected string
	}{
		{"", ""},
		{
			"# ok",
			"# ok",
		},
		{
			"## ok",
			"## ok",
		},
		{
			"##### ok",
			"##### ok",
		},
		{
			"##fix",
			"## fix",
		},
		{
			"foo\n\n##fix\n\nbar",
			"foo\n\n## fix\n\nbar",
		},
		{
			"foo\n##fix\nbar",
			"foo\n\n## fix\n\nbar",
		},
		{
			"foo\n```\n##fix\n```\nbar",
			"foo\n```\n##fix\n```\nbar",
		},
		{
			"foo\n#fix1\n##fix2\nbar",
			"foo\n\n# fix1\n\n## fix2\n\nbar",
		},
	}
	for i, c := range cases {
		in := getMungeLines(c.in)
		expected := getMungeLines(c.expected)
		actual, err := updateHeaderLines("filename.md", in)
		assert.NoError(t, err)
		if !actual.Equal(expected) {
			t.Errorf("case[%d]: expected %q got %q", i, c.expected, actual.String())
		}
	}
}
