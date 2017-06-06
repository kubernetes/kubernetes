/*
Copyright 2017 The Kubernetes Authors.

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

package validation

import (
	"reflect"
	"sort"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

func TestCompactRules(t *testing.T) {
	testcases := map[string]struct {
		Rules    []rbac.PolicyRule
		Expected []rbac.PolicyRule
	}{
		"empty": {
			Rules:    []rbac.PolicyRule{},
			Expected: []rbac.PolicyRule{},
		},
		"simple": {
			Rules: []rbac.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}},
				{Verbs: []string{"list"}, APIGroups: []string{""}, Resources: []string{"builds"}},
				{Verbs: []string{"update", "patch"}, APIGroups: []string{""}, Resources: []string{"builds"}},

				{Verbs: []string{"create"}, APIGroups: []string{"extensions"}, Resources: []string{"daemonsets"}},
				{Verbs: []string{"delete"}, APIGroups: []string{"extensions"}, Resources: []string{"daemonsets"}},

				{Verbs: []string{"educate"}, APIGroups: []string{""}, Resources: []string{"dolphins"}},

				// nil verbs are preserved in non-merge cases.
				// these are the pirates who don't do anything.
				{Verbs: nil, APIGroups: []string{""}, Resources: []string{"pirates"}},

				// Test merging into a nil Verbs string set
				{Verbs: nil, APIGroups: []string{""}, Resources: []string{"pods"}},
				{Verbs: []string{"create"}, APIGroups: []string{""}, Resources: []string{"pods"}},
			},
			Expected: []rbac.PolicyRule{
				{Verbs: []string{"create", "delete"}, APIGroups: []string{"extensions"}, Resources: []string{"daemonsets"}},
				{Verbs: []string{"get", "list", "update", "patch"}, APIGroups: []string{""}, Resources: []string{"builds"}},
				{Verbs: []string{"educate"}, APIGroups: []string{""}, Resources: []string{"dolphins"}},
				{Verbs: nil, APIGroups: []string{""}, Resources: []string{"pirates"}},
				{Verbs: []string{"create"}, APIGroups: []string{""}, Resources: []string{"pods"}},
			},
		},
		"complex multi-group": {
			Rules: []rbac.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{"", "builds.openshift.io"}, Resources: []string{"builds"}},
				{Verbs: []string{"list"}, APIGroups: []string{"", "builds.openshift.io"}, Resources: []string{"builds"}},
			},
			Expected: []rbac.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{"", "builds.openshift.io"}, Resources: []string{"builds"}},
				{Verbs: []string{"list"}, APIGroups: []string{"", "builds.openshift.io"}, Resources: []string{"builds"}},
			},
		},

		"complex multi-resource": {
			Rules: []rbac.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds", "images"}},
				{Verbs: []string{"list"}, APIGroups: []string{""}, Resources: []string{"builds", "images"}},
			},
			Expected: []rbac.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds", "images"}},
				{Verbs: []string{"list"}, APIGroups: []string{""}, Resources: []string{"builds", "images"}},
			},
		},

		"complex named-resource": {
			Rules: []rbac.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, ResourceNames: []string{"mybuild"}},
				{Verbs: []string{"list"}, APIGroups: []string{""}, Resources: []string{"builds"}, ResourceNames: []string{"mybuild2"}},
			},
			Expected: []rbac.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, ResourceNames: []string{"mybuild"}},
				{Verbs: []string{"list"}, APIGroups: []string{""}, Resources: []string{"builds"}, ResourceNames: []string{"mybuild2"}},
			},
		},

		"complex non-resource": {
			Rules: []rbac.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, NonResourceURLs: []string{"/"}},
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, NonResourceURLs: []string{"/foo"}},
			},
			Expected: []rbac.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, NonResourceURLs: []string{"/"}},
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, NonResourceURLs: []string{"/foo"}},
			},
		},
	}

	for k, tc := range testcases {
		rules := tc.Rules
		originalRules, err := api.Scheme.DeepCopy(tc.Rules)
		if err != nil {
			t.Errorf("%s: couldn't copy rules: %v", k, err)
			continue
		}
		compacted, err := CompactRules(tc.Rules)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", k, err)
			continue
		}
		if !reflect.DeepEqual(rules, originalRules) {
			t.Errorf("%s: CompactRules mutated rules. Expected\n%#v\ngot\n%#v", k, originalRules, rules)
			continue
		}
		if covers, missing := Covers(compacted, rules); !covers {
			t.Errorf("%s: compacted rules did not cover original rules. missing: %#v", k, missing)
			continue
		}
		if covers, missing := Covers(rules, compacted); !covers {
			t.Errorf("%s: original rules did not cover compacted rules. missing: %#v", k, missing)
			continue
		}

		sort.Stable(rbac.SortableRuleSlice(compacted))
		sort.Stable(rbac.SortableRuleSlice(tc.Expected))
		if !reflect.DeepEqual(compacted, tc.Expected) {
			t.Errorf("%s: Expected\n%#v\ngot\n%#v", k, tc.Expected, compacted)
			continue
		}
	}
}

func TestIsSimpleResourceRule(t *testing.T) {
	testcases := map[string]struct {
		Rule     rbac.PolicyRule
		Simple   bool
		Resource schema.GroupResource
	}{
		"simple, no verbs": {
			Rule:     rbac.PolicyRule{Verbs: []string{}, APIGroups: []string{""}, Resources: []string{"builds"}},
			Simple:   true,
			Resource: schema.GroupResource{Group: "", Resource: "builds"},
		},
		"simple, one verb": {
			Rule:     rbac.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}},
			Simple:   true,
			Resource: schema.GroupResource{Group: "", Resource: "builds"},
		},
		"simple, multi verb": {
			Rule:     rbac.PolicyRule{Verbs: []string{"get", "list"}, APIGroups: []string{""}, Resources: []string{"builds"}},
			Simple:   true,
			Resource: schema.GroupResource{Group: "", Resource: "builds"},
		},

		"complex, empty": {
			Rule:     rbac.PolicyRule{},
			Simple:   false,
			Resource: schema.GroupResource{},
		},
		"complex, no group": {
			Rule:     rbac.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{}, Resources: []string{"builds"}},
			Simple:   false,
			Resource: schema.GroupResource{},
		},
		"complex, multi group": {
			Rule:     rbac.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{"a", "b"}, Resources: []string{"builds"}},
			Simple:   false,
			Resource: schema.GroupResource{},
		},
		"complex, no resource": {
			Rule:     rbac.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{}},
			Simple:   false,
			Resource: schema.GroupResource{},
		},
		"complex, multi resource": {
			Rule:     rbac.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds", "images"}},
			Simple:   false,
			Resource: schema.GroupResource{},
		},
		"complex, resource names": {
			Rule:     rbac.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, ResourceNames: []string{"foo"}},
			Simple:   false,
			Resource: schema.GroupResource{},
		},
		"complex, non-resource urls": {
			Rule:     rbac.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, NonResourceURLs: []string{"/"}},
			Simple:   false,
			Resource: schema.GroupResource{},
		},
	}

	for k, tc := range testcases {
		resource, simple := isSimpleResourceRule(&tc.Rule)
		if simple != tc.Simple {
			t.Errorf("%s: expected simple=%v, got simple=%v", k, tc.Simple, simple)
			continue
		}
		if resource != tc.Resource {
			t.Errorf("%s: expected resource=%v, got resource=%v", k, tc.Resource, resource)
			continue
		}
	}
}
