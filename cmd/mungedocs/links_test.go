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
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

var _ = fmt.Printf

func TestBadLinks(t *testing.T) {
	var cases = []struct {
		in string
	}{
		{"[NOTREADME](https://github.com/kubernetes/kubernetes/tree/master/NOTREADME.md)"},
		{"[NOTREADME](https://github.com/kubernetes/kubernetes/tree/master/docs/NOTREADME.md)"},
		{"[NOTREADME](../NOTREADME.md)"},
	}
	for _, c := range cases {
		in := getMungeLines(c.in)
		_, err := updateLinks("filename.md", in)
		assert.Error(t, err)
	}
}
func TestGoodLinks(t *testing.T) {
	var cases = []struct {
		in       string
		expected string
	}{
		{"", ""},
		{"[README](https://lwn.net)",
			"[README](https://lwn.net)"},
		// _ to -
		{"[README](https://github.com/kubernetes/kubernetes/tree/master/docs/devel/cli_roadmap.md)",
			"[README](../../docs/devel/cli-roadmap.md)"},
		// - to _
		{"[README](../../docs/devel/api-changes.md)",
			"[README](../../docs/devel/api_changes.md)"},

		// Does this even make sense?  i dunno
		{"[README](/docs/README.md)",
			"[README](https://github.com/docs/README.md)"},
		{"[README](/kubernetes/kubernetes/tree/master/docs/README.md)",
			"[README](../../docs/README.md)"},
	}
	for i, c := range cases {
		in := getMungeLines(c.in)
		expected := getMungeLines(c.expected)
		actual, err := updateLinks("filename.md", in)
		assert.NoError(t, err)
		if !actual.Equal(expected) {
			t.Errorf("case[%d]: expected %q got %q", i, c.expected, actual.String())
		}
	}
}
