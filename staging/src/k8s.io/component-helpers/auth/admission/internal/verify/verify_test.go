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

package verify

import "testing"

func TestAllowsAPIGroup(t *testing.T) {
	for _, tc := range []struct {
		name   string
		groups []string
		group  string
		want   bool
	}{
		{name: "exact", groups: []string{"apps"}, group: "apps", want: true},
		{name: "wildcard", groups: []string{"*"}, group: "example.com", want: true},
		{name: "wildcard core", groups: []string{"*"}, group: "", want: true},
		{name: "empty", groups: nil, group: "apps"},
		{name: "other", groups: []string{"extensions"}, group: "apps"},
		{name: "multiple unrelated", groups: []string{"extensions", "batch"}, group: "apps"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := AllowsAPIGroup(tc.groups, tc.group); got != tc.want {
				t.Fatalf("AllowsAPIGroup(%q, %q) = %t, want %t", tc.groups, tc.group, got, tc.want)
			}
		})
	}
}
