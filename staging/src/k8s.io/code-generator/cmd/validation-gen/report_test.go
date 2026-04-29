/*
Copyright The Kubernetes Authors.

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

import "testing"

func TestJoinPath(t *testing.T) {
	cases := []struct {
		base, seg, want string
	}{
		{"", "", ""},
		{"spec", "", "spec"}, // empty seg is a no-op (inline-embedded struct)
		{"", "spec", "spec"}, // empty base
		{"spec", "name", "spec.name"},
		{"spec", "[*]", "spec[*]"}, // bracket-prefixed seg attaches without dot
		{"", "[*]", "[*]"},
	}
	for _, c := range cases {
		if got := joinPath(c.base, c.seg); got != c.want {
			t.Errorf("joinPath(%q, %q) = %q, want %q", c.base, c.seg, got, c.want)
		}
	}
}
