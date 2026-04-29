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

package cmd

import (
	"slices"
	"testing"

	"k8s.io/code-generator/pkg/guardrails"
)

func TestAllowEntryMatches(t *testing.T) {
	rule := guardrails.Rule{ErrorType: "FieldValueInvalid", Origin: "minimum"}
	cases := []struct {
		name  string
		entry allowEntry
		want  bool
	}{
		{"empty entry matches everything", allowEntry{}, true},
		{"matching group", allowEntry{Group: "foo"}, true},
		{"non-matching group", allowEntry{Group: "bar"}, false},
		{"matching origin", allowEntry{Origin: "minimum"}, true},
		{"non-matching origin", allowEntry{Origin: "format"}, false},
		{"all fields match", allowEntry{
			Group: "foo", Version: "v1", Kind: "Bar",
			Path: "spec.x", ErrorType: "FieldValueInvalid", Origin: "minimum",
		}, true},
		{"one field mismatches", allowEntry{
			Group: "foo", Version: "v1", Kind: "Bar",
			Path: "spec.x", ErrorType: "FieldValueRequired", // wrong errorType
		}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := c.entry.matches("foo", "v1", "Bar", "spec.x", rule)
			if got != c.want {
				t.Errorf("matches() = %v, want %v", got, c.want)
			}
		})
	}
}

// makeReport is a tiny constructor for reports used by Diff/FilterAllowed tests.
func makeReport(group, version, kind string, paths map[string][]guardrails.Rule) guardrails.Report {
	return guardrails.Report{
		Group: group, Version: version,
		Kinds: map[string]map[string][]guardrails.Rule{kind: paths},
	}
}

func TestDiff(t *testing.T) {
	cases := []struct {
		name              string
		declared          map[string][]guardrails.Rule
		observed          map[string][]guardrails.Rule
		wantUncoveredKeys []string // paths remaining after diff (empty = fully covered)
	}{
		{
			name: "covered rule removed, uncovered remains",
			declared: map[string][]guardrails.Rule{
				"spec.a": {{ErrorType: "X"}},
				"spec.b": {{ErrorType: "Y", Origin: "minimum"}},
			},
			observed: map[string][]guardrails.Rule{
				"spec.a": {{ErrorType: "X"}},
			},
			wantUncoveredKeys: []string{"spec.b"},
		},
		{
			name: "concrete index normalizes to [*]",
			declared: map[string][]guardrails.Rule{
				"spec.items[*].name": {{ErrorType: "X"}},
			},
			observed: map[string][]guardrails.Rule{
				"spec.items[3].name": {{ErrorType: "X"}}, // runtime path
			},
		},
		{
			name: "slice-level rule declared at foo matches runtime foo[2]",
			declared: map[string][]guardrails.Rule{
				"spec.items": {{ErrorType: "Duplicate"}},
			},
			observed: map[string][]guardrails.Rule{
				"spec.items[2]": {{ErrorType: "Duplicate"}},
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := diff(
				[]guardrails.Report{makeReport("foo", "v1", "Bar", c.declared)},
				[]guardrails.Report{makeReport("foo", "v1", "Bar", c.observed)},
			)
			var gotKeys []string
			for _, r := range got {
				for path := range r.Kinds["Bar"] {
					gotKeys = append(gotKeys, path)
				}
			}
			slices.Sort(gotKeys)
			slices.Sort(c.wantUncoveredKeys)
			if !slices.Equal(gotKeys, c.wantUncoveredKeys) {
				t.Errorf("uncovered = %v, want %v", gotKeys, c.wantUncoveredKeys)
			}
		})
	}
}

func TestFilterAllowed(t *testing.T) {
	rules := map[string][]guardrails.Rule{
		"spec.a": {{ErrorType: "X"}},
		"spec.b": {{ErrorType: "Y", Origin: "minimum"}},
	}
	cases := []struct {
		name          string
		allow         []allowEntry
		wantRemaining []string // paths surviving the filter
	}{
		{
			name:          "nil allowlist is no-op",
			allow:         nil,
			wantRemaining: []string{"spec.a", "spec.b"},
		},
		{
			name:          "suppress by Path",
			allow:         []allowEntry{{Path: "spec.a"}},
			wantRemaining: []string{"spec.b"},
		},
		{
			name:          "suppress by ErrorType",
			allow:         []allowEntry{{ErrorType: "Y"}},
			wantRemaining: []string{"spec.a"},
		},
		{
			name:          "suppress by Origin",
			allow:         []allowEntry{{Origin: "minimum"}},
			wantRemaining: []string{"spec.a"},
		},
		{
			name:          "non-matching entry leaves everything",
			allow:         []allowEntry{{Group: "other"}},
			wantRemaining: []string{"spec.a", "spec.b"},
		},
		{
			name: "multiple entries — any match suppresses",
			allow: []allowEntry{
				{Path: "spec.a"},
				{ErrorType: "Y"},
			},
			wantRemaining: nil,
		},
		{
			name:          "Group match drops the whole report",
			allow:         []allowEntry{{Group: "foo"}},
			wantRemaining: nil,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			reports := []guardrails.Report{makeReport("foo", "v1", "Bar", rules)}
			got := filterAllowed(reports, c.allow)
			var gotKeys []string
			for _, r := range got {
				for path := range r.Kinds["Bar"] {
					gotKeys = append(gotKeys, path)
				}
			}
			slices.Sort(gotKeys)
			slices.Sort(c.wantRemaining)
			if !slices.Equal(gotKeys, c.wantRemaining) {
				t.Errorf("remaining = %v, want %v", gotKeys, c.wantRemaining)
			}
		})
	}
}
