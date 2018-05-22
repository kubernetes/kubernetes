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

	rbacv1 "k8s.io/api/rbac/v1"
	rbacv1helpers "k8s.io/kubernetes/pkg/apis/rbac/v1"
)

func TestCompactRules(t *testing.T) {
	testcases := map[string]struct {
		Rules    []rbacv1.PolicyRule
		Expected []rbacv1.PolicyRule
	}{
		"empty": {
			Rules:    []rbacv1.PolicyRule{},
			Expected: []rbacv1.PolicyRule{},
		},
		"simple": {
			Rules: []rbacv1.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}},
				{Verbs: []string{"list"}, APIGroups: []string{""}, Resources: []string{"builds"}},
				{Verbs: []string{"update", "patch"}, APIGroups: []string{""}, Resources: []string{"builds"}},

				{Verbs: []string{"create"}, APIGroups: []string{"extensions"}, Resources: []string{"daemonsets"}},
				{Verbs: []string{"delete"}, APIGroups: []string{"extensions"}, Resources: []string{"daemonsets"}},
				{Verbs: []string{"patch"}, APIGroups: []string{"extensions"}, Resources: []string{"daemonsets"}, ResourceNames: []string{""}},
				{Verbs: []string{"get"}, APIGroups: []string{"extensions"}, Resources: []string{"daemonsets"}, ResourceNames: []string{"foo"}},
				{Verbs: []string{"list"}, APIGroups: []string{"extensions"}, Resources: []string{"daemonsets"}, ResourceNames: []string{"foo"}},

				{Verbs: []string{"educate"}, APIGroups: []string{""}, Resources: []string{"dolphins"}},

				// nil verbs are preserved in non-merge cases.
				// these are the pirates who don't do anything.
				{Verbs: nil, APIGroups: []string{""}, Resources: []string{"pirates"}},

				// Test merging into a nil Verbs string set
				{Verbs: nil, APIGroups: []string{""}, Resources: []string{"pods"}},
				{Verbs: []string{"create"}, APIGroups: []string{""}, Resources: []string{"pods"}},
			},
			Expected: []rbacv1.PolicyRule{
				{Verbs: []string{"create", "delete"}, APIGroups: []string{"extensions"}, Resources: []string{"daemonsets"}},
				{Verbs: []string{"patch"}, APIGroups: []string{"extensions"}, Resources: []string{"daemonsets"}, ResourceNames: []string{""}},
				{Verbs: []string{"get", "list"}, APIGroups: []string{"extensions"}, Resources: []string{"daemonsets"}, ResourceNames: []string{"foo"}},
				{Verbs: []string{"get", "list", "update", "patch"}, APIGroups: []string{""}, Resources: []string{"builds"}},
				{Verbs: []string{"educate"}, APIGroups: []string{""}, Resources: []string{"dolphins"}},
				{Verbs: nil, APIGroups: []string{""}, Resources: []string{"pirates"}},
				{Verbs: []string{"create"}, APIGroups: []string{""}, Resources: []string{"pods"}},
			},
		},
		"complex multi-group": {
			Rules: []rbacv1.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{"", "builds.openshift.io"}, Resources: []string{"builds"}},
				{Verbs: []string{"list"}, APIGroups: []string{"", "builds.openshift.io"}, Resources: []string{"builds"}},
			},
			Expected: []rbacv1.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{"", "builds.openshift.io"}, Resources: []string{"builds"}},
				{Verbs: []string{"list"}, APIGroups: []string{"", "builds.openshift.io"}, Resources: []string{"builds"}},
			},
		},

		"complex multi-resource": {
			Rules: []rbacv1.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds", "images"}},
				{Verbs: []string{"list"}, APIGroups: []string{""}, Resources: []string{"builds", "images"}},
			},
			Expected: []rbacv1.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds", "images"}},
				{Verbs: []string{"list"}, APIGroups: []string{""}, Resources: []string{"builds", "images"}},
			},
		},

		"complex named-resource": {
			Rules: []rbacv1.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, ResourceNames: []string{"mybuild"}},
				{Verbs: []string{"list"}, APIGroups: []string{""}, Resources: []string{"builds"}, ResourceNames: []string{"mybuild2"}},
			},
			Expected: []rbacv1.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, ResourceNames: []string{"mybuild"}},
				{Verbs: []string{"list"}, APIGroups: []string{""}, Resources: []string{"builds"}, ResourceNames: []string{"mybuild2"}},
			},
		},

		"complex non-resource": {
			Rules: []rbacv1.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, NonResourceURLs: []string{"/"}},
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, NonResourceURLs: []string{"/foo"}},
			},
			Expected: []rbacv1.PolicyRule{
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, NonResourceURLs: []string{"/"}},
				{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, NonResourceURLs: []string{"/foo"}},
			},
		},
	}

	for k, tc := range testcases {
		rules := tc.Rules
		originalRules := make([]rbacv1.PolicyRule, len(tc.Rules))
		for i := range tc.Rules {
			originalRules[i] = *tc.Rules[i].DeepCopy()
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

		sort.Stable(rbacv1helpers.SortableRuleSlice(compacted))
		sort.Stable(rbacv1helpers.SortableRuleSlice(tc.Expected))
		if !reflect.DeepEqual(compacted, tc.Expected) {
			t.Errorf("%s: Expected\n%#v\ngot\n%#v", k, tc.Expected, compacted)
			continue
		}
	}
}

func TestIsSimpleResourceRule(t *testing.T) {
	testcases := map[string]struct {
		Rule     rbacv1.PolicyRule
		Simple   bool
		Resource simpleResource
	}{
		"simple, no verbs": {
			Rule:     rbacv1.PolicyRule{Verbs: []string{}, APIGroups: []string{""}, Resources: []string{"builds"}},
			Simple:   true,
			Resource: simpleResource{Group: "", Resource: "builds"},
		},
		"simple, one verb": {
			Rule:     rbacv1.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}},
			Simple:   true,
			Resource: simpleResource{Group: "", Resource: "builds"},
		},
		"simple, one empty resource name": {
			Rule:     rbacv1.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, ResourceNames: []string{""}},
			Simple:   true,
			Resource: simpleResource{Group: "", Resource: "builds", ResourceNameExist: true, ResourceName: ""},
		},
		"simple, one resource name": {
			Rule:     rbacv1.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, ResourceNames: []string{"foo"}},
			Simple:   true,
			Resource: simpleResource{Group: "", Resource: "builds", ResourceNameExist: true, ResourceName: "foo"},
		},
		"simple, multi verb": {
			Rule:     rbacv1.PolicyRule{Verbs: []string{"get", "list"}, APIGroups: []string{""}, Resources: []string{"builds"}},
			Simple:   true,
			Resource: simpleResource{Group: "", Resource: "builds"},
		},

		"complex, empty": {
			Rule:     rbacv1.PolicyRule{},
			Simple:   false,
			Resource: simpleResource{},
		},
		"complex, no group": {
			Rule:     rbacv1.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{}, Resources: []string{"builds"}},
			Simple:   false,
			Resource: simpleResource{},
		},
		"complex, multi group": {
			Rule:     rbacv1.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{"a", "b"}, Resources: []string{"builds"}},
			Simple:   false,
			Resource: simpleResource{},
		},
		"complex, no resource": {
			Rule:     rbacv1.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{}},
			Simple:   false,
			Resource: simpleResource{},
		},
		"complex, multi resource": {
			Rule:     rbacv1.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds", "images"}},
			Simple:   false,
			Resource: simpleResource{},
		},
		"complex, resource names": {
			Rule:     rbacv1.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, ResourceNames: []string{"foo", "bar"}},
			Simple:   false,
			Resource: simpleResource{},
		},
		"complex, non-resource urls": {
			Rule:     rbacv1.PolicyRule{Verbs: []string{"get"}, APIGroups: []string{""}, Resources: []string{"builds"}, NonResourceURLs: []string{"/"}},
			Simple:   false,
			Resource: simpleResource{},
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
