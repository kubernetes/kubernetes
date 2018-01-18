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

package rbac_test

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/rbac/v1"

	// install RBAC types
	_ "k8s.io/kubernetes/pkg/apis/rbac/install"
)

// TestHelpersRoundTrip confirms that the rbac.New* helper functions produce RBAC objects that match objects
// that have gone through conversion and defaulting.  This is required because these helper functions are
// used to create the bootstrap RBAC policy which is used during reconciliation.  If they produced objects
// that did not match, reconciliation would incorrectly add duplicate data to the cluster's RBAC policy.
func TestHelpersRoundTrip(t *testing.T) {
	rb := rbac.NewRoleBinding("role", "ns").Groups("g").SAs("ns", "sa").Users("u").BindingOrDie()
	rbcr := rbac.NewRoleBindingForClusterRole("role", "ns").Groups("g").SAs("ns", "sa").Users("u").BindingOrDie()
	crb := rbac.NewClusterBinding("role").Groups("g").SAs("ns", "sa").Users("u").BindingOrDie()

	role := &rbac.Role{
		Rules: []rbac.PolicyRule{
			rbac.NewRule("verb").Groups("g").Resources("foo").RuleOrDie(),
			rbac.NewRule("verb").URLs("/foo").RuleOrDie(),
		},
	}
	clusterRole := &rbac.ClusterRole{
		Rules: []rbac.PolicyRule{
			rbac.NewRule("verb").Groups("g").Resources("foo").RuleOrDie(),
			rbac.NewRule("verb").URLs("/foo").RuleOrDie(),
		},
	}

	for _, internalObj := range []runtime.Object{&rb, &rbcr, &crb, role, clusterRole} {
		v1Obj, err := legacyscheme.Scheme.ConvertToVersion(internalObj, v1.SchemeGroupVersion)
		if err != nil {
			t.Errorf("err on %T: %v", internalObj, err)
			continue
		}
		legacyscheme.Scheme.Default(v1Obj)
		roundTrippedObj, err := legacyscheme.Scheme.ConvertToVersion(v1Obj, rbac.SchemeGroupVersion)
		if err != nil {
			t.Errorf("err on %T: %v", internalObj, err)
			continue
		}
		if !reflect.DeepEqual(internalObj, roundTrippedObj) {
			t.Errorf("err on %T: got difference:\n%s", internalObj, diff.ObjectDiff(internalObj, roundTrippedObj))
			continue
		}
	}
}

func TestResourceMatches(t *testing.T) {
	tests := []struct {
		name                      string
		ruleResources             []string
		combinedRequestedResource string
		requestedSubresource      string
		expected                  bool
	}{
		{
			name:                      "all matches 01",
			ruleResources:             []string{"*"},
			combinedRequestedResource: "foo",
			expected:                  true,
		},
		{
			name:                      "checks all rules",
			ruleResources:             []string{"doesn't match", "*"},
			combinedRequestedResource: "foo",
			expected:                  true,
		},
		{
			name:                      "matches exact rule",
			ruleResources:             []string{"foo/bar"},
			combinedRequestedResource: "foo/bar",
			requestedSubresource:      "bar",
			expected:                  true,
		},
		{
			name:                      "matches exact rule 02",
			ruleResources:             []string{"foo/bar"},
			combinedRequestedResource: "foo",
			expected:                  false,
		},
		{
			name:                      "matches subresource",
			ruleResources:             []string{"*/scale"},
			combinedRequestedResource: "foo/scale",
			requestedSubresource:      "scale",
			expected:                  true,
		},
		{
			name:                      "doesn't match partial subresource hit",
			ruleResources:             []string{"foo/bar", "*/other"},
			combinedRequestedResource: "foo/other/segment",
			requestedSubresource:      "other/segment",
			expected:                  false,
		},
		{
			name:                      "matches subresource with multiple slashes",
			ruleResources:             []string{"*/other/segment"},
			combinedRequestedResource: "foo/other/segment",
			requestedSubresource:      "other/segment",
			expected:                  true,
		},
		{
			name:                      "doesn't fail on empty",
			ruleResources:             []string{""},
			combinedRequestedResource: "foo/other/segment",
			requestedSubresource:      "other/segment",
			expected:                  false,
		},
		{
			name:                      "doesn't fail on slash",
			ruleResources:             []string{"/"},
			combinedRequestedResource: "foo/other/segment",
			requestedSubresource:      "other/segment",
			expected:                  false,
		},
		{
			name:                      "doesn't fail on missing subresource",
			ruleResources:             []string{"*/"},
			combinedRequestedResource: "foo/other/segment",
			requestedSubresource:      "other/segment",
			expected:                  false,
		},
		{
			name:                      "doesn't match on not star",
			ruleResources:             []string{"*something/other/segment"},
			combinedRequestedResource: "foo/other/segment",
			requestedSubresource:      "other/segment",
			expected:                  false,
		},
		{
			name:                      "doesn't match on something else",
			ruleResources:             []string{"something/other/segment"},
			combinedRequestedResource: "foo/other/segment",
			requestedSubresource:      "other/segment",
			expected:                  false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rule := &rbac.PolicyRule{
				Resources: tc.ruleResources,
			}
			actual := rbac.ResourceMatches(rule, tc.combinedRequestedResource, tc.requestedSubresource)
			if tc.expected != actual {
				t.Errorf("expected %v, got %v", tc.expected, actual)
			}

		})
	}
}
